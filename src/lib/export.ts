import { numpy as np } from '@jax-js/jax';
import { safetensors } from '@jax-js/loaders';

import { LORA_TARGETS, type Lora, type Target } from './models/llama';
import type { ModelDef } from './models/registry';
import type { Weights } from './models/weights';

/** HF module path for each adapted projection. */
const HF_MODULE: Record<Target, string> = {
	q: 'self_attn.q_proj',
	k: 'self_attn.k_proj',
	v: 'self_attn.v_proj',
	o: 'self_attn.o_proj',
	gate: 'mlp.gate_proj',
	up: 'mlp.up_proj',
	down: 'mlp.down_proj'
};

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

/** LoRA adapters in PEFT naming, loadable with `PeftModel.from_pretrained`. */
export async function loraTensors(lora: Lora): Promise<NamedTensor[]> {
	const out: NamedTensor[] = [];
	for (let i = 0; i < lora.length; i++) {
		for (const t of LORA_TARGETS) {
			const prefix = `base_model.model.model.layers.${i}.${HF_MODULE[t]}`;
			const { a, b } = lora[i][t];
			out.push({ name: `${prefix}.lora_A.weight`, shape: a.shape, data: await read(a) });
			out.push({ name: `${prefix}.lora_B.weight`, shape: b.shape, data: await read(b) });
		}
	}
	return out;
}

/** Full weights in HF naming (embeddings tied, so no lm_head). */
export async function weightTensors(w: Weights): Promise<NamedTensor[]> {
	const out: NamedTensor[] = [
		{ name: 'model.embed_tokens.weight', shape: w.embed.shape, data: await read(w.embed) },
		{ name: 'model.norm.weight', shape: w.norm.shape, data: await read(w.norm) }
	];
	for (let i = 0; i < w.layers.length; i++) {
		const l = w.layers[i];
		const p = `model.layers.${i}`;
		out.push({ name: `${p}.input_layernorm.weight`, shape: l.inNorm.shape, data: await read(l.inNorm) });
		out.push({ name: `${p}.post_attention_layernorm.weight`, shape: l.postNorm.shape, data: await read(l.postNorm) });
		for (const t of LORA_TARGETS) {
			out.push({ name: `${p}.${HF_MODULE[t]}.weight`, shape: l[t].w.shape, data: await read(l[t].w) });
			const bias = l[t].b;
			if (bias) out.push({ name: `${p}.${HF_MODULE[t]}.bias`, shape: bias.shape, data: await read(bias) });
		}
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
			target_modules: LORA_TARGETS.map((t) => HF_MODULE[t].split('.')[1]),
			fan_in_fan_out: false,
			inference_mode: true
		},
		null,
		2
	);
}

/** Parse our own safetensors back into a LoRA list or a weights tree. */
export function parseLora(bytes: Uint8Array<ArrayBuffer>, layers: number): Lora {
	const file = safetensors.parse(bytes);
	return Array.from({ length: layers }, (_, i) => {
		const layer = {} as Lora[number];
		for (const t of LORA_TARGETS) {
			const prefix = `base_model.model.model.layers.${i}.${HF_MODULE[t]}`;
			const a = file.tensors[`${prefix}.lora_A.weight`];
			const b = file.tensors[`${prefix}.lora_B.weight`];
			layer[t] = {
				a: np.array(a.data as Float32Array<ArrayBuffer>, { shape: a.shape }),
				b: np.array(b.data as Float32Array<ArrayBuffer>, { shape: b.shape })
			};
		}
		return layer;
	});
}

export function parseWeights(bytes: Uint8Array<ArrayBuffer>, layers: number): Weights {
	const file = safetensors.parse(bytes);
	const get = (name: string) => {
		const t = file.tensors[name];
		return t ? np.array(t.data as Float32Array<ArrayBuffer>, { shape: t.shape }) : undefined;
	};
	return {
		embed: get('model.embed_tokens.weight')!,
		norm: get('model.norm.weight')!,
		layers: Array.from({ length: layers }, (_, i) => {
			const p = `model.layers.${i}`;
			const layer = {
				inNorm: get(`${p}.input_layernorm.weight`)!,
				postNorm: get(`${p}.post_attention_layernorm.weight`)!
			} as Weights['layers'][number];
			for (const t of LORA_TARGETS) {
				const b = get(`${p}.${HF_MODULE[t]}.bias`);
				layer[t] = b ? { w: get(`${p}.${HF_MODULE[t]}.weight`)!, b } : { w: get(`${p}.${HF_MODULE[t]}.weight`)! };
			}
			return layer;
		})
	};
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
