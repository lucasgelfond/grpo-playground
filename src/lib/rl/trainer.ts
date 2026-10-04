import { blockUntilReady, jit, numpy as np, tree } from '@jax-js/jax';
import { adamw, applyUpdates, chain, clipByGlobalNorm, type GradientTransformation, type OptState } from '@jax-js/optax';

import { compileModel, layerTargets, LORA_TARGETS, type CompiledModel, type Lora, type Target } from '../models/llama';
import type { ModelConfig, ModelDef } from '../models/registry';
import type { Weights } from '../models/weights';
import { bucket, leftPad, PROMPT_BUCKET } from './generate';

export type Mode = 'lora' | 'full';

export type TrainSettings = {
	mode: Mode;
	learningRate: number;
	loraRank: number;
	loraAlpha: number;
	temperature: number;
	/** Rows per forward/backward pass; gradients are accumulated across them. */
	microbatch: number;
};

export type UpdateResult = {
	skipped: boolean;
	advantages: number[];
	loss: number;
	/** Per-token log-probs of each completion before and after the update. */
	logpBefore: number[][];
	logpAfter: number[][];
	/** Mean per-token KL(π ‖ π_ref) of each answer, before the update. */
	kl: number[];
};

export type DpoMargin = { chosen: number; rejected: number; p: number; before: number; after: number };

export type DpoResult = {
	skipped: boolean;
	loss: number;
	/** Each pair's implicit reward margin before and after the step. */
	margins: DpoMargin[];
	/** Per-token log-probs of answers that were in a pair (empty otherwise). */
	logpBefore: number[][];
	logpAfter: number[][];
	kl: number[];
};

/** Per-token KL estimate (k3), matching the loss: exp(r) - r - 1, r = ref - policy. */
export function klPerToken(logp: number, ref: number): number {
	const r = ref - logp;
	return Math.exp(r) - r - 1;
}

const COMPLETION_BUCKET = 32;
/** Skip groups whose best and worst rewards are closer than this. */
const MIN_SPREAD = 0.05;

/**
 * Rows of [prompt (left-padded) | completion (right-padded)] for one prompt,
 * with the completion tokens as targets, bucketed to few shapes.
 */
function packRows(promptIds: number[], completions: number[][], padToken: number) {
	const n = completions.length;
	const P = bucket(promptIds.length, PROMPT_BUCKET);
	const L = bucket(Math.max(1, ...completions.map((c) => c.length)), COMPLETION_BUCKET);
	const { row: promptRow, pad } = leftPad(promptIds, P, padToken);
	const ids = new Int32Array(n * (P + L));
	const targets = new Int32Array(n * L).fill(padToken);
	completions.forEach((c, g) => {
		ids.set(promptRow, g * (P + L));
		ids.fill(padToken, g * (P + L) + P, (g + 1) * (P + L));
		ids.set(c, g * (P + L) + P);
		targets.set(c, g * L);
	});
	return {
		L,
		batch: (start: number, end: number) => ({
			ids: np.array(ids.subarray(start * (P + L), end * (P + L)), { shape: [end - start, P + L], dtype: np.int32 }),
			pad: np.array(new Int32Array(end - start).fill(pad), { dtype: np.int32 }),
			targets: np.array(targets.subarray(start * L, end * L), { shape: [end - start, L], dtype: np.int32 })
		}),
		/** Per-row values [rows, L] -> one array per completion, trimmed to its length. */
		unflatten: (data: Float32Array, start: number, end: number, into: number[][]) => {
			for (let g = start; g < end; g++) {
				const off = (g - start) * L;
				into[g] = Array.from(data.subarray(off, off + completions[g].length));
			}
		}
	};
}

/** [in, out] of each adapted projection. */
export function targetDims(cfg: ModelConfig, t: Target): [number, number] {
	const q = cfg.heads * cfg.headDim;
	const kv = cfg.kvHeads * cfg.headDim;
	switch (t) {
		case 'q':
			return [cfg.hidden, q];
		case 'k':
		case 'v':
			return [cfg.hidden, kv];
		case 'in':
			return [cfg.hidden, 3 * cfg.hidden];
		case 'o':
			// A short-conv layer's output projection is [D, D]; attention's is [heads * headDim, D].
			return [cfg.layerTypes ? cfg.hidden : q, cfg.hidden];
		case 'gate':
		case 'up':
			return [cfg.hidden, cfg.intermediate];
		case 'down':
			return [cfg.intermediate, cfg.hidden];
	}
}

function gaussian(n: number, std: number): Float32Array<ArrayBuffer> {
	const out = new Float32Array(n);
	for (let i = 0; i < n; i += 2) {
		const u = 1 - Math.random();
		const v = Math.random();
		const r = Math.sqrt(-2 * Math.log(u)) * std;
		out[i] = r * Math.cos(2 * Math.PI * v);
		if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
	}
	return out;
}

/** Standard LoRA init: A ~ N(0, 1/in), B = 0, so the adapter starts as a no-op. */
export function initLora(cfg: ModelConfig, rank: number): Lora {
	return Array.from({ length: cfg.layers }, (_, i) => {
		const layer: Lora[number] = {};
		for (const t of layerTargets(cfg, i)) {
			const [inDim, outDim] = targetDims(cfg, t);
			layer[t] = {
				a: np.array(gaussian(rank * inDim, 1 / Math.sqrt(inDim)), { shape: [rank, inDim] }),
				b: np.zeros([outDim, rank])
			};
		}
		return layer;
	});
}

/**
 * Owns the trainable parameters (LoRA adapters, or every weight) and the Adam
 * state, and runs one GRPO update per call to `update`.
 */
export class PolicyTrainer {
	readonly model: CompiledModel;
	readonly cfg: ModelConfig;
	/** Current weights. In full mode these are what we train. */
	weights: Weights;
	/** Full mode only: a frozen copy of the original weights (for base chat + deltas). */
	readonly base: Weights | null;
	lora: Lora | null;
	readonly settings: TrainSettings;
	#opt: GradientTransformation;
	#optState: OptState;
	#loraNorms;
	#fullNorms;

	constructor(def: ModelDef, weights: Weights, base: Weights | null, settings: TrainSettings, lora?: Lora) {
		this.cfg = def.config;
		this.settings = settings;
		this.model = compileModel(def.config, { loraRank: settings.loraRank, loraAlpha: settings.loraAlpha });
		this.weights = weights;
		this.base = base;
		this.lora = settings.mode === 'lora' ? (lora ?? initLora(def.config, settings.loraRank)) : null;
		this.#opt = chain(
			clipByGlobalNorm(1.0),
			adamw(settings.learningRate, { b1: 0.9, b2: 0.99, weightDecay: 0 })
		);
		this.#optState = this.#opt.init(tree.ref(this.#trainable()));

		const scale = this.model.loraScale;
		// ||scale * B A||_F computed via r x r Gram matrices: tr(BᵀB · AAᵀ).
		this.#loraNorms = jit((lora: Lora) =>
			np.stack(
				lora.map((layer) =>
					np.stack(
						LORA_TARGETS.map((t) => {
							if (!layer[t]) return np.zeros([]);
							const { a, b } = layer[t];
							const aat = np.dot(a.ref, a.transpose());
							const btb = np.dot(b.ref.transpose(), b);
							return np.sqrt(np.maximum(aat.mul(btb).sum(), 0)).mul(scale);
						})
					)
				)
			)
		);
		this.#fullNorms = jit((w: Weights, w0: Weights) =>
			np.stack(
				w.layers.map((layer, i) =>
					np.stack(
						LORA_TARGETS.map((t) => {
							if (!layer[t]) return np.zeros([]);
							const d = layer[t].w.sub(w0.layers[i][t]!.w);
							return np.sqrt(d.ref.mul(d).sum());
						})
					)
				)
			)
		);
	}

	#trainable(): Weights | Lora {
		return this.settings.mode === 'lora' ? this.lora! : this.weights;
	}

	/** Parameters for sampling from the current policy. */
	policy(): { weights: Weights; lora: Lora | null } {
		return { weights: this.weights, lora: this.lora };
	}

	/** Parameters of the untouched starting model. */
	original(): { weights: Weights; lora: null } {
		return { weights: this.base ?? this.weights, lora: null };
	}

	/**
	 * One GRPO step (DeepSeekMath) on a group of G completions of one prompt:
	 *
	 *   J = 1/G Σ_i 1/|o_i| Σ_t [ Â_i · log π(o_i,t)  −  β · KL_t(π ‖ π_ref) ]
	 *
	 * with Â_i = (r_i − mean(r)) / std(r), the same for every token of answer i.
	 * Each answer counts equally however long it is (per-answer token mean,
	 * then group mean). We take one gradient step per group of samples, so the
	 * PPO ratio π/π_old is exactly 1 and its clipping never activates; the
	 * gradient is exactly Â · ∇log π. Groups whose rewards are within MIN_SPREAD
	 * of each other are skipped: dividing by their tiny std would turn judge
	 * noise into a full-strength update.
	 */
	async update(
		promptIds: number[],
		completions: number[][],
		rewards: number[],
		padToken: number,
		klBeta: number
	): Promise<UpdateResult> {
		const G = completions.length;
		const mean = rewards.reduce((s, r) => s + r, 0) / G;
		const std = Math.sqrt(rewards.reduce((s, r) => s + (r - mean) ** 2, 0) / G);
		const tied = Math.max(...rewards) - Math.min(...rewards) < MIN_SPREAD;
		const advantages = rewards.map((r) => (tied ? 0 : (r - mean) / (std + 1e-4)));
		const empty = completions.map((c) => c.map(() => 0));
		if (tied || completions.every((c) => c.length === 0)) {
			return { skipped: true, advantages, loss: 0, logpBefore: empty, logpAfter: empty, kl: completions.map(() => 0) };
		}

		const { L, batch, unflatten } = packRows(promptIds, completions, padToken);
		// Per-token weights 1 / (|o_i| · G): mean over each answer's tokens, then
		// over the group. Padding gets weight 0. Microbatch gradients add up.
		const weights = new Float32Array(G * L);
		completions.forEach((c, g) => c.length && weights.fill(1 / (c.length * G), g * L, g * L + c.length));
		const invTemp = 1 / this.settings.temperature;

		const mb = Math.max(1, Math.min(this.settings.microbatch, G));

		// Reference log-probs: the original model (LoRA off, or the frozen copy).
		const original = this.original();
		const logpRef: number[][] = [];
		const refFlat = new Float32Array(G * L);
		for (let start = 0; start < G; start += mb) {
			const end = Math.min(G, start + mb);
			const b = batch(start, end);
			const lp = this.model.logprobs(tree.ref(original.weights), null, b.ids, b.pad, b.targets, invTemp);
			const data = (await lp.data()) as Float32Array;
			refFlat.set(data, start * L);
			unflatten(data, start, end, logpRef);
		}

		const logpBefore: number[][] = [];
		let loss = 0;
		let grads: Weights | Lora | null = null;
		for (let start = 0; start < G; start += mb) {
			const end = Math.min(G, start + mb);
			const b = batch(start, end);
			const [[lossVal, logp], g] = this.model.trainStep(
				tree.ref(this.#trainable()),
				this.settings.mode === 'lora' ? tree.ref(this.weights) : null,
				b.ids,
				b.pad,
				b.targets,
				np.array(weights.subarray(start * L, end * L), { shape: [end - start, L] }),
				np.array(new Float32Array(advantages.slice(start, end))),
				np.array(1),
				np.array(refFlat.subarray(start * L, end * L), { shape: [end - start, L] }),
				np.array(klBeta),
				invTemp
			) as unknown as [[np.Array, np.Array], Weights | Lora];
			loss += await lossVal.jsAsync();
			unflatten((await logp.data()) as Float32Array, start, end, logpBefore);
			grads = grads === null ? g : (tree.map((x: np.Array, y: np.Array) => x.add(y), grads, g) as typeof g);
		}

		const [updates, optState] = this.#opt.update(grads!, this.#optState, tree.ref(this.#trainable()));
		this.#optState = optState;
		if (this.settings.mode === 'lora') this.lora = applyUpdates(this.lora!, updates as Lora);
		else this.weights = applyUpdates(this.weights, updates as Weights);
		await blockUntilReady(this.#trainable());

		const logpAfter: number[][] = [];
		for (let start = 0; start < G; start += mb) {
			const end = Math.min(G, start + mb);
			const b = batch(start, end);
			const logp = this.model.logprobs(tree.ref(this.weights), this.lora ? tree.ref(this.lora) : null, b.ids, b.pad, b.targets, invTemp);
			unflatten((await logp.data()) as Float32Array, start, end, logpAfter);
		}

		const kl = logpBefore.map((lps, g) =>
			lps.length ? lps.reduce((s, lp, i) => s + klPerToken(lp, logpRef[g][i]), 0) / lps.length : 0
		);
		return { skipped: false, advantages, loss, logpBefore, logpAfter, kl };
	}

	// DPO works but isn't exposed in the UI (mostly because GRPO playground was a better name, LOL). if you do ?mode=dpo or go to dpo.lucasgelfond.online you can try the DPO version
	/**
	 * One DPO step on preference pairs between a prompt's completions. `pairs` index into
	 * `completions`; p is the judge's confidence that `chosen` is better.
	 * Margins m = β·((s_c − ref_c) − (s_r − ref_r)) are reported before and
	 * after the step, so the view can show whether each pair moved the right way.
	 */
	async updateDpo(
		promptIds: number[],
		completions: number[][],
		pairs: { chosen: number; rejected: number; p: number }[],
		padToken: number,
		beta: number
	): Promise<DpoResult> {
		const empty = completions.map(() => [] as number[]);
		if (!pairs.length) return { skipped: true, loss: 0, margins: [], logpBefore: empty, logpAfter: empty, kl: completions.map(() => 0) };
		const K = pairs.length;
		// Rows [chosen_0, rejected_0, chosen_1, ...]; an answer can appear in several pairs.
		const rowOf = pairs.flatMap((q) => [q.chosen, q.rejected]);
		const rows = rowOf.map((i) => completions[i]);
		const { L, batch, unflatten } = packRows(promptIds, rows, padToken);
		const mask = new Float32Array(2 * K * L);
		rows.forEach((c, r) => mask.fill(1, r * L, r * L + c.length));
		const seqSums = (lps: number[][]) => lps.map((x) => x.reduce((a, b) => a + b, 0));
		// Pairs per microbatch: two rows each.
		const mbPairs = Math.max(1, Math.floor(this.settings.microbatch / 2));

		const logprobs = async (weights: Weights, lora: Lora | null) => {
			const out: number[][] = [];
			for (let k = 0; k < K; k += mbPairs) {
				const [s, e] = [2 * k, 2 * Math.min(K, k + mbPairs)];
				const b = batch(s, e);
				unflatten((await this.model.logprobs(tree.ref(weights), lora ? tree.ref(lora) : null, b.ids, b.pad, b.targets, 1).data()) as Float32Array, s, e, out);
			}
			return out;
		};
		const original = this.original();
		const logpRef = await logprobs(original.weights, null);
		const refSeq = seqSums(logpRef);

		const logpRows: number[][] = [];
		const before: number[] = [];
		let loss = 0;
		let grads: Weights | Lora | null = null;
		for (let k = 0; k < K; k += mbPairs) {
			const kEnd = Math.min(K, k + mbPairs);
			const [s, e] = [2 * k, 2 * kEnd];
			const b = batch(s, e);
			const [[lossVal, [logp, m]], g] = this.model.dpoStep(
				tree.ref(this.#trainable()),
				this.settings.mode === 'lora' ? tree.ref(this.weights) : null,
				b.ids,
				b.pad,
				b.targets,
				np.array(mask.subarray(s * L, e * L), { shape: [e - s, L] }),
				np.array(new Float32Array(refSeq.slice(s, e))),
				np.array(new Float32Array(pairs.slice(k, kEnd).map((q) => q.p))),
				np.array(beta),
				np.array(1 / K)
			) as unknown as [[np.Array, [np.Array, np.Array]], Weights | Lora];
			loss += await lossVal.jsAsync();
			unflatten((await logp.data()) as Float32Array, s, e, logpRows);
			before.push(...((await m.data()) as Float32Array));
			grads = grads === null ? g : (tree.map((x: np.Array, y: np.Array) => x.add(y), grads, g) as typeof g);
		}

		const [updates, optState] = this.#opt.update(grads!, this.#optState, tree.ref(this.#trainable()));
		this.#optState = optState;
		if (this.settings.mode === 'lora') this.lora = applyUpdates(this.lora!, updates as Lora);
		else this.weights = applyUpdates(this.weights, updates as Weights);
		await blockUntilReady(this.#trainable());

		const afterRows = await logprobs(this.weights, this.lora);
		const afterSeq = seqSums(afterRows);
		const margins = pairs.map((q, k) => ({
			...q,
			before: before[k],
			after: beta * (afterSeq[2 * k] - refSeq[2 * k] - (afterSeq[2 * k + 1] - refSeq[2 * k + 1]))
		}));

		// Per-answer views (first row each answer appears in) for the token tints and KL.
		const logpBefore = completions.map(() => [] as number[]);
		const logpAfter = completions.map(() => [] as number[]);
		const kl = completions.map(() => 0);
		rowOf.forEach((i, r) => {
			if (logpBefore[i].length || !rows[r].length) return;
			logpBefore[i] = logpRows[r];
			logpAfter[i] = afterRows[r];
			kl[i] = logpRows[r].reduce((s, lp, t) => s + klPerToken(lp, logpRef[r][t]), 0) / logpRows[r].length;
		});
		return { skipped: false, loss, margins, logpBefore, logpAfter, kl };
	}

	/** How far each adapted matrix has moved from the original: [layer][target] Frobenius norm. */
	async layerNorms(): Promise<number[][]> {
		const out =
			this.settings.mode === 'lora'
				? this.#loraNorms(tree.ref(this.lora!))
				: this.#fullNorms(tree.ref(this.weights), tree.ref(this.base!));
		return (await out.jsAsync()) as number[][];
	}

	/** Free everything; `keepWeights` leaves the (frozen, LoRA-mode) base weights for reuse. */
	dispose(keepWeights = false) {
		tree.dispose(this.#optState);
		if (this.lora) tree.dispose(this.lora);
		if (!keepWeights) tree.dispose(this.weights);
		if (this.base) tree.dispose(this.base);
	}
}

/** Rough cost model shown on the setup screen. */
export function estimateCost(cfg: ModelConfig, mode: Mode, rank: number) {
	let matrices = 0;
	let loraParams = 0;
	for (let l = 0; l < cfg.layers; l++) {
		for (const t of layerTargets(cfg, l)) {
			const [i, o] = targetDims(cfg, t);
			matrices += i * o;
			loraParams += rank * (i + o);
		}
	}
	const totalParams = matrices + cfg.vocab * cfg.hidden;
	const trainable = mode === 'lora' ? loraParams : totalParams;
	const f32 = 4;
	// weights + (full: frozen copy for comparisons) + grads + Adam m and v.
	const memory = totalParams * f32 + (mode === 'full' ? totalParams * f32 : 0) + trainable * f32 * 3;
	return { totalParams, trainable, memoryBytes: memory, fraction: trainable / totalParams };
}
