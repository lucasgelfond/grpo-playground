import { numpy as np } from '@jax-js/jax';
import { safetensors } from '@jax-js/loaders';

import { layerTargets, type Lora } from './models/llama';
import { globalTensorNames, hfModule, layerTensorNames } from './models/names';
import type { ModelConfig, ModelDef } from './models/registry';
import type { Weights } from './models/weights';

type NamedTensor = { name: string; shape: number[]; data: Float32Array };

/** Serialize f32 tensors to the safetensors format. */
export function writeSafetensors(tensors: NamedTensor[], metadata: Record<string, string> = {}): Uint8Array<ArrayBuffer> {
	const header: Record<string, unknown> = { __metadata__: metadata };
	let offset = 0;
	for (const t of tensors) {
		header[t.name] = { dtype: 'F32', shape: t.shape, data_offsets: [offset, offset + t.data.byteLength] };
		offset += t.data.byteLength;
	}
	let json = JSON.stringify(header);
	json += ' '.repeat((8 - ((json.length + 8) % 8)) % 8); // keep data 8-byte aligned
	const headerBytes = new TextEncoder().encode(json);
	const out = new Uint8Array(8 + headerBytes.length + offset);
	new DataView(out.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
	out.set(headerBytes, 8);
	let pos = 8 + headerBytes.length;
	for (const t of tensors) {
		out.set(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength), pos);
		pos += t.data.byteLength;
	}
	return out;
}

async function read(x: np.Array): Promise<Float32Array> {
	return (await x.ref.astype(np.float32).data()) as Float32Array;
}

/** Follow a path like ['layers', '3', 'q', 'w'] into a tree. */
function at(root: unknown, path: string[]): np.Array | undefined {
	let x = root as Record<string, unknown> | undefined;
	for (const key of path) x = x?.[key] as Record<string, unknown> | undefined;
	return x as np.Array | undefined;
}

function setAt(root: Record<string, unknown>, path: string[], value: np.Array) {
	let x = root;
	for (const key of path.slice(0, -1)) x = (x[key] ??= {}) as Record<string, unknown>;
	x[path[path.length - 1]] = value;
}

/** LoRA adapters in PEFT naming, loadable with `PeftModel.from_pretrained`. */
export async function loraTensors(lora: Lora, cfg: ModelConfig): Promise<NamedTensor[]> {
	const out: NamedTensor[] = [];
	for (let i = 0; i < lora.length; i++) {
		for (const t of layerTargets(cfg, i)) {
			const prefix = `base_model.model.model.layers.${i}.${hfModule(cfg, i, t)}`;
			const { a, b } = lora[i][t]!;
			out.push({ name: `${prefix}.lora_A.weight`, shape: a.shape, data: await read(a) });
			out.push({ name: `${prefix}.lora_B.weight`, shape: b.shape, data: await read(b) });
		}
	}
	return out;
}

/** Full weights in HF naming (embeddings tied, so no lm_head). */
export async function weightTensors(w: Weights, cfg: ModelConfig): Promise<NamedTensor[]> {
	const out: NamedTensor[] = [];
	const add = async (name: string, x: np.Array | undefined) => x && out.push({ name, shape: x.shape, data: await read(x) });
	for (const [name, path] of globalTensorNames(cfg)) await add(name, at(w, path));
	for (let i = 0; i < w.layers.length; i++) {
		for (const [name, path] of layerTensorNames(cfg, i)) await add(`model.layers.${i}.${name}`, at(w.layers[i], path));
	}
	return out;
}

export function peftConfig(def: ModelDef, rank: number, alpha: number): string {
	return JSON.stringify(
		{
			peft_type: 'LORA',
			task_type: 'CAUSAL_LM',
			base_model_name_or_path: def.repo,
			r: rank,
			lora_alpha: alpha,
			lora_dropout: 0,
			bias: 'none',
			target_modules: [
				...new Set(
					Array.from({ length: def.config.layers }, (_, i) =>
						layerTargets(def.config, i).map((t) => hfModule(def.config, i, t).split('.').pop()!)
					).flat()
				)
			],
			fan_in_fan_out: false,
			inference_mode: true
		},
		null,
		2
	);
}

/** Parse our own safetensors back into a LoRA list or a weights tree. */
export function parseLora(bytes: Uint8Array<ArrayBuffer>, cfg: ModelConfig): Lora {
	const file = safetensors.parse(bytes);
	const get = (name: string) => {
		const t = file.tensors[name];
		return np.array(t.data as Float32Array<ArrayBuffer>, { shape: t.shape });
	};
	return Array.from({ length: cfg.layers }, (_, i) => {
		const layer: Lora[number] = {};
		for (const t of layerTargets(cfg, i)) {
			const prefix = `base_model.model.model.layers.${i}.${hfModule(cfg, i, t)}`;
			layer[t] = { a: get(`${prefix}.lora_A.weight`), b: get(`${prefix}.lora_B.weight`) };
		}
		return layer;
	});
}

export function parseWeights(bytes: Uint8Array<ArrayBuffer>, cfg: ModelConfig): Weights {
	const file = safetensors.parse(bytes);
	const root: Record<string, unknown> = { layers: Array.from({ length: cfg.layers }, () => ({})) };
	const put = (name: string, path: string[]) => {
		const t = file.tensors[name];
		if (t) setAt(root, path, np.array(t.data as Float32Array<ArrayBuffer>, { shape: t.shape }));
	};
	for (const [name, path] of globalTensorNames(cfg)) put(name, path);
	for (let i = 0; i < cfg.layers; i++) {
		for (const [name, path] of layerTensorNames(cfg, i)) put(`model.layers.${i}.${name}`, ['layers', String(i), ...path]);
	}
	return root as Weights;
}

// --- a minimal store-only zip writer, for downloading adapter + config together ---

const CRC_TABLE = (() => {
	const table = new Uint32Array(256);
	for (let n = 0; n < 256; n++) {
		let c = n;
		for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
		table[n] = c >>> 0;
	}
	return table;
})();

function crc32(data: Uint8Array): number {
	let c = 0xffffffff;
	for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
	return (c ^ 0xffffffff) >>> 0;
}

export function zip(files: { name: string; data: Uint8Array }[]): Blob {
	const parts: Uint8Array[] = [];
	const central: Uint8Array[] = [];
	let offset = 0;
	for (const f of files) {
		const name = new TextEncoder().encode(f.name);
		const crc = crc32(f.data);
		const local = new Uint8Array(30 + name.length);
		const lv = new DataView(local.buffer);
		lv.setUint32(0, 0x04034b50, true);
		lv.setUint16(4, 20, true);
		lv.setUint32(14, crc, true);
		lv.setUint32(18, f.data.length, true);
		lv.setUint32(22, f.data.length, true);
		lv.setUint16(26, name.length, true);
		local.set(name, 30);
		const entry = new Uint8Array(46 + name.length);
		const cv = new DataView(entry.buffer);
		cv.setUint32(0, 0x02014b50, true);
		cv.setUint16(4, 20, true);
		cv.setUint16(6, 20, true);
		cv.setUint32(16, crc, true);
		cv.setUint32(20, f.data.length, true);
		cv.setUint32(24, f.data.length, true);
		cv.setUint16(28, name.length, true);
		cv.setUint32(42, offset, true);
		entry.set(name, 46);
		parts.push(local, f.data);
		central.push(entry);
		offset += local.length + f.data.length;
	}
	const centralSize = central.reduce((s, c) => s + c.length, 0);
	const end = new Uint8Array(22);
	const ev = new DataView(end.buffer);
	ev.setUint32(0, 0x06054b50, true);
	ev.setUint16(8, files.length, true);
	ev.setUint16(10, files.length, true);
	ev.setUint32(12, centralSize, true);
	ev.setUint32(16, offset, true);
	return new Blob([...parts, ...central, end] as BlobPart[], { type: 'application/zip' });
}

export function downloadBlob(blob: Blob, filename: string) {
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = filename;
	a.click();
	setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
