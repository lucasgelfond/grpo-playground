import { numpy as np, tree } from '@jax-js/jax';

import type { CompiledModel, LayerCache, Lora } from '../models/llama';
import type { Weights } from '../models/weights';

export const PROMPT_BUCKET = 32;

export function bucket(n: number, step: number): number {
	return Math.max(step, Math.ceil(n / step) * step);
}

export type GenOptions = {
	rows: number;
	maxNew: number;
	/** 0 means greedy. */
	temperature: number;
	stopTokens: number[];
	padToken: number;
	signal?: AbortSignal;
	/** Called after every decode step with the tokens so far. */
	onStep?: (tokens: number[][], done: boolean[]) => void;
};

export type Generation = {
	tokens: number[][]; // includes the stop token when one was produced
	logps: number[][]; // log-prob of each sampled token under the sampling distribution
	stopped: boolean[];
};

/** Left-pad `ids` to `len`. Returns the flat row and the pad count. */
export function leftPad(ids: number[], len: number, padToken: number): { row: number[]; pad: number } {
	const pad = len - ids.length;
	return { row: [...Array(pad).fill(padToken), ...ids], pad };
}

/** Sample from `logits / temperature`; returns [token, logprob]. */
function sampleRow(logits: Float32Array, temperature: number): [number, number] {
	let max = -Infinity;
	let argmax = 0;
	for (let i = 0; i < logits.length; i++) {
		if (logits[i] > max) {
			max = logits[i];
			argmax = i;
		}
	}
	const t = temperature > 0 ? temperature : 1;
	let total = 0;
	for (let i = 0; i < logits.length; i++) total += Math.exp((logits[i] - max) / t);
	const logZ = Math.log(total);
	if (temperature <= 0) return [argmax, -logZ];

	let r = Math.random() * total;
	for (let i = 0; i < logits.length; i++) {
		const p = Math.exp((logits[i] - max) / t);
		r -= p;
		if (r <= 0) return [i, (logits[i] - max) / t - logZ];
	}
	return [argmax, -logZ];
}

/**
 * Generate `rows` completions of the same prompt in one batch, with a KV
 * cache. Every row shares the prompt, so the prompt is left-padded once.
 */
export async function generate(
	model: CompiledModel,
	weights: Weights,
	lora: Lora | null,
	promptIds: number[],
	opts: GenOptions
): Promise<Generation> {
	const { rows: B, maxNew } = opts;
	const P = bucket(promptIds.length, PROMPT_BUCKET);
	const { row, pad } = leftPad(promptIds, P, opts.padToken);
	const ids = np.array(new Int32Array(Array.from({ length: B }, () => row).flat()), {
		shape: [B, P],
		dtype: np.int32
	});
	const pads = new Int32Array(B).fill(pad);
	const capacity = P + maxNew;

	const tokens: number[][] = Array.from({ length: B }, () => []);
	const logps: number[][] = Array.from({ length: B }, () => []);
	const stopped = new Array<boolean>(B).fill(false);

	let logits: np.Array | null;
	let caches: LayerCache[];
	[logits, caches] = model.prefill(
		tree.ref(weights),
		lora ? tree.ref(lora) : null,
		ids,
		np.array(pads, { dtype: np.int32 }),
		capacity
	);

	try {
		for (let t = 0; t < maxNew; t++) {
			const current = logits!;
			logits = null; // data() consumes the array
			const data = (await current.data()) as Float32Array;
			const V = data.length / B;
			const next = new Int32Array(B);
			for (let b = 0; b < B; b++) {
				if (stopped[b]) {
					next[b] = opts.padToken;
					continue;
				}
				const [tok, lp] = sampleRow(data.subarray(b * V, (b + 1) * V), opts.temperature);
				tokens[b].push(tok);
				logps[b].push(lp);
				next[b] = tok;
				if (opts.stopTokens.includes(tok)) stopped[b] = true;
			}
			opts.onStep?.(tokens, stopped);
			if (stopped.every(Boolean) || t === maxNew - 1 || opts.signal?.aborted) break;
			[logits, caches] = model.decode(
				tree.ref(weights),
				lora ? tree.ref(lora) : null,
				caches,
				np.array(next, { dtype: np.int32 }),
				np.array(P + t, { dtype: np.int32 }),
				np.array(pads, { dtype: np.int32 })
			);
		}
	} finally {
		logits?.dispose();
		tree.dispose(caches);
	}
	return { tokens, logps, stopped };
}
