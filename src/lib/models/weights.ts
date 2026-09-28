import { blockUntilReady, numpy as np } from '@jax-js/jax';
import { cachedFetch, opfs } from '@jax-js/loaders';

import { modelUrl, type ModelDef } from './registry';

/**
 * Streams a safetensors checkpoint onto the GPU one tensor at a time using
 * HTTP Range requests. The file is stored in R2 as fixed-size parts (see
 * registry.ts), so a byte range is read from whichever part(s) it falls in.
 *
 * The stock `safetensors.parse` needs the whole file in one ArrayBuffer and
 * doesn't know BF16. Streaming keeps peak JS memory at one tensor (the 1.5B
 * judge is a 3 GB file) and lets us convert BF16 -> f32/f16 on the fly.
 * Raw tensor bytes are cached in OPFS so the second visit is offline.
 */

export type Linear = { w: np.Array; b?: np.Array };

export type Layer = {
	inNorm: np.Array;
	postNorm: np.Array;
	q: Linear;
	k: Linear;
	v: Linear;
	o: Linear;
	gate: Linear;
	up: Linear;
	down: Linear;
};

export type Weights = { embed: np.Array; layers: Layer[]; norm: np.Array };

export type LoadProgress = {
	loaded: number;
	total: number;
	/** Bytes that had to come over the network (the rest came from the OPFS cache). */
	downloaded: number;
};

type HeaderEntry = { dtype: string; shape: number[]; data_offsets: [number, number] };

const CONCURRENCY = 6;

type Manifest = { file: string; size: number; partSize: number; parts: number };

/** A model file split into parts: `url` names the whole file (and keys the cache). */
type Source = { url: string; manifest: Manifest };

async function openSource(def: ModelDef): Promise<Source> {
	const manifest = JSON.parse(new TextDecoder().decode(await cachedFetch(modelUrl(def, 'model.manifest.json'))));
	return { url: modelUrl(def, manifest.file), manifest };
}

async function fetchPart(url: string, start: number, end: number): Promise<Uint8Array<ArrayBuffer>> {
	const resp = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` } });
	if (resp.status !== 206 && resp.status !== 200) {
		throw new Error(`Failed to fetch ${url}: ${resp.status} ${resp.statusText}`);
	}
	const buf = new Uint8Array(await resp.arrayBuffer());
	if (resp.status === 200) return buf.slice(start, end); // server ignored Range
	if (buf.byteLength !== end - start) throw new Error(`Short read from ${url}`);
	return buf;
}

/** Bytes [start, end) of the whole file, read across its parts. */
async function fetchRange(src: Source, start: number, end: number): Promise<Uint8Array<ArrayBuffer>> {
	const { partSize } = src.manifest;
	const out = new Uint8Array(end - start);
	const reads = [];
	for (let k = Math.floor(start / partSize); k * partSize < end; k++) {
		const partStart = k * partSize;
		const from = Math.max(start, partStart);
		const to = Math.min(end, partStart + partSize);
		const partUrl = `${src.url}.part${String(k).padStart(2, '0')}`;
		reads.push(fetchPart(partUrl, from - partStart, to - partStart).then((bytes) => out.set(bytes, from - start)));
	}
	await Promise.all(reads);
	return out;
}

/** A byte range of a remote file, cached in OPFS. `fromCache` says where it came from. */
async function cachedRange(src: Source, key: string, start: number, end: number) {
	const cacheKey = `${src.url}#${key}`;
	const hit = await opfs.read(cacheKey);
	if (hit && hit.byteLength === end - start) return { data: hit, fromCache: true };
	const data = await fetchRange(src, start, end);
	await opfs.write(cacheKey, data);
	return { data, fromCache: false };
}

/**
 * Ask the browser not to evict our cached weights under storage pressure.
 * Without this, OPFS data is "best effort" and can silently disappear.
 */
export async function persistStorage(): Promise<boolean> {
	try {
		if (await navigator.storage.persisted()) return true;
		return await navigator.storage.persist();
	} catch {
		return false;
	}
}

async function readHeader(src: Source): Promise<{ header: Record<string, HeaderEntry>; dataStart: number }> {
	const { data: lenBytes } = await cachedRange(src, '__len__', 0, 8);
	const headerLen = Number(new DataView(lenBytes.buffer, lenBytes.byteOffset, 8).getBigUint64(0, true));
	const { data: headerBytes } = await cachedRange(src, '__header__', 8, 8 + headerLen);
	const header = JSON.parse(new TextDecoder().decode(headerBytes));
	delete header.__metadata__;
	return { header, dataStart: 8 + headerLen };
}

/** BF16 is the top half of an f32, so conversion is a 16-bit shift. */
function bf16ToF32(raw: Uint8Array<ArrayBuffer>): Float32Array<ArrayBuffer> {
	const src = new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2);
	const out = new Uint32Array(src.length);
	for (let i = 0; i < src.length; i++) out[i] = src[i] << 16;
	return new Float32Array(out.buffer);
}

function bf16ToF16(raw: Uint8Array<ArrayBuffer>): Float16Array<ArrayBuffer> {
	const src = new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2);
	const out = new Float16Array(src.length);
	const CHUNK = 1 << 20;
	const tmp = new Uint32Array(CHUNK);
	const tmpF = new Float32Array(tmp.buffer);
	for (let off = 0; off < src.length; off += CHUNK) {
		const n = Math.min(CHUNK, src.length - off);
		for (let i = 0; i < n; i++) tmp[i] = src[off + i] << 16;
		out.set(n === CHUNK ? tmpF : tmpF.subarray(0, n), off);
	}
	return out;
}

function toArray(entry: HeaderEntry, raw: Uint8Array<ArrayBuffer>, dtype: np.DType): np.Array {
	if (entry.dtype !== 'BF16') throw new Error(`Expected BF16 weights, got ${entry.dtype}`);
	if (dtype === np.float16) return np.array(bf16ToF16(raw), { shape: entry.shape, dtype: np.float16 });
	return np.array(bf16ToF32(raw), { shape: entry.shape, dtype: np.float32 });
}

/** HF tensor name -> path in our `Weights` tree. */
function mapName(name: string): string[] | null {
	if (name === 'model.embed_tokens.weight') return ['embed'];
	if (name === 'model.norm.weight') return ['norm'];
	if (name === 'lm_head.weight') return null; // tied to embed_tokens
	const m = name.match(/^model\.layers\.(\d+)\.(.+)$/);
	if (!m) throw new Error(`Unexpected tensor ${name}`);
	const rest: Record<string, string[]> = {
		'input_layernorm.weight': ['inNorm'],
		'post_attention_layernorm.weight': ['postNorm'],
		'self_attn.q_proj.weight': ['q', 'w'],
		'self_attn.q_proj.bias': ['q', 'b'],
		'self_attn.k_proj.weight': ['k', 'w'],
		'self_attn.k_proj.bias': ['k', 'b'],
		'self_attn.v_proj.weight': ['v', 'w'],
		'self_attn.v_proj.bias': ['v', 'b'],
		'self_attn.o_proj.weight': ['o', 'w'],
		'mlp.gate_proj.weight': ['gate', 'w'],
		'mlp.up_proj.weight': ['up', 'w'],
		'mlp.down_proj.weight': ['down', 'w']
	};
	const path = rest[m[2]];
	if (!path) throw new Error(`Unexpected tensor ${name}`);
	return ['layers', m[1], ...path];
}

function setPath(root: any, path: string[], value: np.Array) {
	let obj = root;
	for (let i = 0; i < path.length - 1; i++) {
		const key = path[i];
		const nextIsIndex = /^\d+$/.test(path[i + 1]);
		obj[key] ??= nextIsIndex ? [] : {};
		obj = obj[key];
	}
	obj[path[path.length - 1]] = value;
}

/**
 * Download (or read from cache) a model's weights and upload them to the
 * default device in `dtype`. Biases and norms are always kept in f32.
 */
export async function loadWeights(
	def: ModelDef,
	dtype: np.DType,
	onProgress?: (p: LoadProgress) => void
): Promise<Weights> {
	const src = await openSource(def);
	const { header, dataStart } = await readHeader(src);
	const entries = Object.entries(header).sort((a, b) => a[1].data_offsets[0] - b[1].data_offsets[0]);
	const total = entries.reduce((s, [, e]) => s + e.data_offsets[1] - e.data_offsets[0], 0);
	let loaded = 0;
	let downloaded = 0;
	onProgress?.({ loaded, total, downloaded });

	const root: any = {};
	let next = 0;
	const worker = async () => {
		while (next < entries.length) {
			const [name, entry] = entries[next++];
			const path = mapName(name);
			const [s, e] = entry.data_offsets;
			const { data: raw, fromCache } = await cachedRange(src, name, dataStart + s, dataStart + e);
			if (!fromCache) downloaded += e - s;
			if (path) {
				const isMatrix = entry.shape.length === 2;
				const arr = toArray(entry, raw, isMatrix ? dtype : np.float32);
				await arr.blockUntilReady();
				setPath(root, path, arr);
			}
			loaded += e - s;
			onProgress?.({ loaded, total, downloaded });
		}
	};
	await Promise.all(Array.from({ length: CONCURRENCY }, worker));

	const weights = root as Weights;
	if (weights.layers?.length !== def.config.layers) {
		throw new Error(`Expected ${def.config.layers} layers, found ${weights.layers?.length}`);
	}
	return blockUntilReady(weights);
}

/** Is every tensor of this model already in the OPFS cache? */
export async function isCached(def: ModelDef): Promise<boolean> {
	try {
		const url = modelUrl(def, 'model.safetensors');
		const files = await opfs.list();
		const prefix = `${url}#`;
		const bytes = files.filter((f) => f.name.startsWith(prefix)).reduce((s, f) => s + f.size, 0);
		return bytes >= def.downloadBytes * 0.99;
	} catch {
		return false;
	}
}

/** Fetch every tensor (and the tokenizer) into the OPFS cache without loading it anywhere. */
export async function downloadWeights(def: ModelDef, onProgress?: (p: LoadProgress) => void): Promise<void> {
	const src = await openSource(def);
	await cachedFetch(modelUrl(def, 'tokenizer.json'));
	const { header, dataStart } = await readHeader(src);
	const entries = Object.entries(header);
	const total = entries.reduce((s, [, e]) => s + e.data_offsets[1] - e.data_offsets[0], 0);
	let loaded = 0;
	let downloaded = 0;
	let next = 0;
	const worker = async () => {
		while (next < entries.length) {
			const [name, entry] = entries[next++];
			const [s, e] = entry.data_offsets;
			const { fromCache } = await cachedRange(src, name, dataStart + s, dataStart + e);
			if (!fromCache) downloaded += e - s;
			loaded += e - s;
			onProgress?.({ loaded, total, downloaded });
		}
	};
	await Promise.all(Array.from({ length: CONCURRENCY }, worker));
}
