import { jit, nn, numpy as np, valueAndGrad } from '@jax-js/jax';

import type { ModelConfig } from './registry';
import type { Layer, Linear, Weights } from './weights';

/**
 * A Llama-style decoder (SmolLM2 / Qwen2.5 / Qwen3, and LFM2's mix of attention
 * and short-convolution layers) written against jax-js, plus the
 * three compiled entry points the app needs:
 *
 * - `prefill` / `decode`: batched KV-cache generation for sampling answers.
 * - `extend`: the judge's A/B logits for many suffixes of one cached prefix.
 * - `trainStep`: GRPO policy-gradient loss + gradients w.r.t. LoRA adapters
 *   (or all weights for full fine-tuning), returning per-token log-probs.
 *
 * Sequences are left-padded: `pad[b]` is how many pad tokens precede row b.
 * RoPE positions are shifted by `pad` and pad keys are masked, so a padded
 * row computes exactly what it would unpadded. Padding to a few fixed bucket
 * lengths keeps the number of distinct shapes (and jit compiles) small.
 *
 * Every function follows jax-js ownership rules: arguments are consumed, and
 * `.ref` is taken wherever a value is used more than once.
 */

/** Every projection an adapter can target. A layer has the ones its kind uses (see layerTargets). */
export const LORA_TARGETS = ['q', 'k', 'v', 'in', 'o', 'gate', 'up', 'down'] as const;
export type Target = (typeof LORA_TARGETS)[number];
export type LoraPair = { a: np.Array; b: np.Array }; // a: [r, in], b: [out, r]
export type LoraLayer = Partial<Record<Target, LoraPair>>;
export type Lora = LoraLayer[];

export type KV = { k: np.Array; v: np.Array }; // [B, cap, kvHeads, headDim]
/** Short-conv layers carry the last K-1 gated inputs instead of keys and values: [B, K-1, D]. */
export type ConvState = { conv: np.Array };
export type LayerCache = KV | ConvState;

export type LayerKind = 'attention' | 'conv';
export function layerKind(cfg: ModelConfig, i: number): LayerKind {
	return cfg.layerTypes?.[i] ?? 'attention';
}
/** The projections a layer has: attention q/k/v/o, or a short conv's in/out, plus the MLP. */
export function layerTargets(cfg: ModelConfig, i: number): Target[] {
	return layerKind(cfg, i) === 'conv' ? ['in', 'o', 'gate', 'up', 'down'] : ['q', 'k', 'v', 'o', 'gate', 'up', 'down'];
}

function f32(x: np.Array): np.Array {
	return x.dtype === np.float32 ? x : x.astype(np.float32);
}

/** y = x W^T (+ b) (+ scale * x A^T B^T). Weights may be stored in f16. */
function linear(lin: Linear, x: np.Array, lora: LoraPair | undefined, loraScale: number): np.Array {
	let y = np.dot(lora ? x.ref : x, f32(lin.w).transpose());
	if (lin.b) y = y.add(lin.b);
	if (lora) {
		const low = np.dot(x, lora.a.transpose());
		y = y.add(np.dot(low, lora.b.transpose()).mul(loraScale));
	}
	return y;
}

function rmsNorm(x: np.Array, w: np.Array, eps: number): np.Array {
	const ms = x.ref.mul(x.ref).mean(-1, { keepdims: true });
	return x.div(np.sqrt(ms.add(eps))).mul(w);
}

/** cos/sin tables for rotate-half RoPE; positions: [B, T] float. -> [B, T, 1, D] */
function ropeTables(positions: np.Array, headDim: number, theta: number): [np.Array, np.Array] {
	const [B, T] = positions.shape;
	const half = headDim / 2;
	const invFreq = np.exp(
		np.arange(half, undefined, undefined, { dtype: np.float32 }).mul((-Math.log(theta) * 2) / headDim)
	);
	const freqs = positions.reshape([B, T, 1]).mul(invFreq);
	const c = np.cos(freqs.ref);
	const s = np.sin(freqs);
	const cos = np.concatenate([c.ref, c], -1).reshape([B, T, 1, headDim]);
	const sin = np.concatenate([s.ref, s], -1).reshape([B, T, 1, headDim]);
	return [cos, sin];
}

function applyRope(x: np.Array, cos: np.Array, sin: np.Array): np.Array {
	const [x1, x2] = np.split(x.ref, 2, -1);
	const rotated = np.concatenate([x2.mul(-1), x1], -1);
	return x.mul(cos).add(rotated.mul(sin));
}

type Ctx = { cfg: ModelConfig; loraScale: number };

/** Project q/k/v for x: [B, T, D], normalize q/k per head (Qwen3), and apply RoPE. */
function qkv(ctx: Ctx, lw: Layer, ll: LoraLayer | undefined, x: np.Array, cos: np.Array, sin: np.Array) {
	const { heads, kvHeads, headDim } = ctx.cfg;
	const [B, T] = x.shape;
	let q = linear(lw.q!, x.ref, ll?.q, ctx.loraScale).reshape([B, T, heads, headDim]);
	let k = linear(lw.k!, x.ref, ll?.k, ctx.loraScale).reshape([B, T, kvHeads, headDim]);
	const v = linear(lw.v!, x, ll?.v, ctx.loraScale).reshape([B, T, kvHeads, headDim]);
	if (lw.qNorm && lw.kNorm) {
		q = rmsNorm(q, lw.qNorm, ctx.cfg.rmsEps);
		k = rmsNorm(k, lw.kNorm, ctx.cfg.rmsEps);
	}
	q = applyRope(q, cos.ref, sin.ref);
	k = applyRope(k, cos, sin);
	return { q, k, v };
}

function mlp(ctx: Ctx, lw: Layer, ll: LoraLayer | undefined, x: np.Array): np.Array {
	const g = linear(lw.gate, x.ref, ll?.gate, ctx.loraScale);
	const u = linear(lw.up, x, ll?.up, ctx.loraScale);
	return linear(lw.down, nn.silu(g).mul(u), ll?.down, ctx.loraScale);
}

/**
 * LFM2's short convolution over a whole sequence. x: [B, T, D], already
 * normalized, with padding rows zeroed (so padding feeds the convolution the
 * same zeros an unpadded sequence starts with). in_proj splits into gates
 * (b, c) and input u; y = c * conv(b * u), a causal depthwise convolution of
 * width K. Returns the projected output and the last K-1 gated inputs.
 */
function shortConv(ctx: Ctx, lw: Layer, ll: LoraLayer | undefined, x: np.Array): [np.Array, np.Array] {
	const { hidden: D, convKernel: K = 3 } = ctx.cfg;
	const T = x.shape[1];
	const [b, c, u] = np.split(linear(lw.in!, x, ll?.in, ctx.loraScale), 3, -1);
	const padded = np.pad(b.mul(u), { 1: [K - 1, 0] }); // [B, K-1+T, D]
	const w = f32(lw.conv!).reshape([D, K]);
	let y = padded.ref.slice([], [0, T]).mul(w.ref.slice([], 0));
	for (let j = 1; j < K; j++) y = y.add(padded.ref.slice([], [j, j + T]).mul(w.ref.slice([], j)));
	w.dispose();
	const state = padded.slice([], [T, T + K - 1]);
	return [linear(lw.o, c.mul(y), ll?.o, ctx.loraScale), state];
}

/** One decode step of the short convolution. x: [B, 1, D]; state: [B, K-1, D]. */
function shortConvStep(ctx: Ctx, lw: Layer, ll: LoraLayer | undefined, x: np.Array, state: np.Array): [np.Array, np.Array] {
	const { hidden: D, convKernel: K = 3 } = ctx.cfg;
	const [b, c, u] = np.split(linear(lw.in!, x, ll?.in, ctx.loraScale), 3, -1);
	const window = np.concatenate([state, b.mul(u)], 1); // [B, K, D]
	const w = f32(lw.conv!).reshape([D, K]).transpose().reshape([1, K, D]);
	const y = window.ref.mul(w).sum(1, { keepdims: true });
	return [linear(lw.o, c.mul(y), ll?.o, ctx.loraScale), window.slice([], [1, K])];
}

function embed(table: np.Array, ids: np.Array): np.Array {
	return f32(table.slice(ids));
}

/** (ids == arange(V)) as f32, shape [...ids, V]. Fused into its consumer. */
function oneHot(ids: np.Array, V: number): np.Array {
	return np.arange(V).equal(ids.reshape([...ids.shape, 1])).astype(np.float32);
}

/**
 * Embedding lookup as a one-hot matmul. Used only when the table itself is
 * being trained: jax-js can't yet jit the scatter-add that differentiating a
 * gather produces, while a matmul's gradient is another matmul.
 */
function embedDifferentiable(table: np.Array, ids: np.Array): np.Array {
	return np.dot(oneHot(ids, table.shape[0]), f32(table));
}

/**
 * Full-sequence pass. ids: [B, T] int32, pad: [B] int32.
 * Returns final-norm hidden states [B, T, D] and optionally each layer's K/V.
 */
function runSequence(
	ctx: Ctx,
	W: Omit<Weights, 'embed'>,
	x: np.Array,
	lora: Lora,
	pad: np.Array,
	keepKV: boolean
): { h: np.Array; kvs: LayerCache[] } {
	const { cfg } = ctx;
	const [B, T] = x.shape;
	const idx = np.arange(T);
	const positions = idx.ref
		.astype(np.float32)
		.reshape([1, T])
		.sub(pad.ref.astype(np.float32).reshape([B, 1]));
	// Real queries see only real keys; pad queries see earlier pads, which
	// keeps their softmax finite (they're never read by real tokens).
	// Short-conv layers see padding as zeros: [B, T, 1], 1 for real tokens.
	const real = cfg.layerTypes ? idx.ref.reshape([1, T, 1]).greaterEqual(pad.ref.reshape([B, 1, 1])).astype(np.float32) : null;
	const keyReal = idx.ref.reshape([1, 1, 1, T]).greaterEqual(pad.ref.reshape([B, 1, 1, 1]));
	const queryPad = idx.reshape([1, 1, T, 1]).less(pad.reshape([B, 1, 1, 1]));
	const mask = np.logicalOr(keyReal, queryPad);
	const [cos, sin] = ropeTables(positions, cfg.headDim, cfg.ropeTheta);

	const kvs: LayerCache[] = [];
	for (let i = 0; i < cfg.layers; i++) {
		const lw = W.layers[i];
		const ll = lora[i];
		const h = rmsNorm(x.ref, lw.inNorm, cfg.rmsEps);
		if (layerKind(cfg, i) === 'conv') {
			const [out, state] = shortConv(ctx, lw, ll, h.mul(real!.ref));
			if (keepKV) kvs.push({ conv: state });
			else state.dispose();
			x = x.add(out);
		} else {
			const { q, k, v } = qkv(ctx, lw, ll, h, cos.ref, sin.ref);
			const attn = nn.dotProductAttention(q, keepKV ? k.ref : k, keepKV ? v.ref : v, {
				mask: mask.ref,
				isCausal: true
			});
			if (keepKV) kvs.push({ k, v });
			x = x.add(linear(lw.o, attn.reshape([B, T, cfg.heads * cfg.headDim]), ll?.o, ctx.loraScale));
		}
		x = x.add(mlp(ctx, lw, ll, rmsNorm(x.ref, lw.postNorm, cfg.rmsEps)));
	}
	real?.dispose();
	mask.dispose();
	cos.dispose();
	sin.dispose();
	return { h: rmsNorm(x, W.norm, cfg.rmsEps), kvs };
}

function logitsOf(table: np.Array, h: np.Array): np.Array {
	return np.dot(h, f32(table).transpose());
}

export type CompiledModel = ReturnType<typeof compileModel>;

export function compileModel(cfg: ModelConfig, opts: { loraRank?: number; loraAlpha?: number } = {}) {
	const ctx: Ctx = { cfg, loraScale: (opts.loraAlpha ?? 32) / (opts.loraRank ?? 16) };

	/** Prompt pass that also returns K/V caches padded to `capacity`. */
	const prefillJit = jit(
		function prefill(W: Weights, lora: Lora, ids: np.Array, pad: np.Array, capacity: number) {
			const { embed: table, ...rest } = W;
			const T = ids.shape[1];
			const x = embed(table.ref, ids);
			const { h, kvs } = runSequence(ctx, rest, x, lora, pad, true);
			const caches = kvs.map((c) =>
				'conv' in c ? c : { k: np.pad(c.k, { 1: [0, capacity - T] }), v: np.pad(c.v, { 1: [0, capacity - T] }) }
			);
			return [logitsOf(table, h.slice([], -1)), caches] as [np.Array, LayerCache[]];
		},
		{ staticArgnums: [4] }
	);

	/**
	 * Prefix caching: many suffixes continue one shared, already-prefilled
	 * prefix. `prefix` holds each layer's K/V for a single row [1, P, kv, hd],
	 * left-padded by `prefixPad` (scalar). Each row of `ids` [B, S] is a
	 * different suffix, left-padded by pad[b] between prefix and suffix; those
	 * pad keys are masked. Returns last-token logits [B, V].
	 *
	 * Attention-only models (the judges); LFM2's conv layers aren't handled here.
	 * The judge uses this to read its long, shared instructions once and then
	 * only pay for each pair of answers. Only the logits of `pick` (token ids,
	 * e.g. "A" and "B") are computed: -> [B, pick.length], not [B, vocab].
	 */
	const extendJit = jit(function extend(
		W: Weights,
		prefix: KV[],
		prefixPad: np.Array,
		ids: np.Array,
		pad: np.Array,
		pick: np.Array
	) {
		const { embed: table, layers, norm } = W;
		const [B, S] = ids.shape;
		const P = prefix[0].k.shape[1];
		let x = embed(table.ref, ids);
		const sIdx = np.arange(S);
		// Suffix positions continue from the prefix's real length (P - prefixPad).
		const positions = sIdx.ref
			.astype(np.float32)
			.reshape([1, S])
			.sub(pad.ref.astype(np.float32).reshape([B, 1]))
			.add(P)
			.sub(prefixPad.ref.astype(np.float32));
		const [cos, sin] = ropeTables(positions, cfg.headDim, cfg.ropeTheta);

		// Keys = [prefix | suffix]. Real prefix keys are visible to everyone; suffix
		// keys are causal, and pad queries may see earlier pads (keeps softmax finite).
		const prefixVisible = np
			.arange(P)
			.greaterEqual(prefixPad)
			.reshape([1, 1, 1, P])
			.astype(np.int32)
			.add(np.zeros([B, 1, S, 1], { dtype: np.int32 }))
			.greater(0);
		const causal = np.tri(S, S, 0, { dtype: np.bool }).reshape([1, 1, S, S]);
		const keyReal = sIdx.ref.reshape([1, 1, 1, S]).greaterEqual(pad.ref.reshape([B, 1, 1, 1]));
		const queryPad = sIdx.reshape([1, 1, S, 1]).less(pad.reshape([B, 1, 1, 1]));
		const suffixVisible = np.logicalAnd(causal, np.logicalOr(keyReal, queryPad));
		const mask = np.concatenate([prefixVisible, suffixVisible], -1);

		for (let i = 0; i < cfg.layers; i++) {
			const lw = layers[i];
			const h = rmsNorm(x.ref, lw.inNorm, cfg.rmsEps);
			const { q, k, v } = qkv(ctx, lw, undefined, h, cos.ref, sin.ref);
			const kAll = np.concatenate([np.tile(prefix[i].k, [B, 1, 1, 1]), k], 1);
			const vAll = np.concatenate([np.tile(prefix[i].v, [B, 1, 1, 1]), v], 1);
			const attn = nn.dotProductAttention(q, kAll, vAll, { mask: mask.ref });
			x = x.add(linear(lw.o, attn.reshape([B, S, cfg.heads * cfg.headDim]), undefined, ctx.loraScale));
			x = x.add(mlp(ctx, lw, undefined, rmsNorm(x.ref, lw.postNorm, cfg.rmsEps)));
		}
		mask.dispose();
		cos.dispose();
		sin.dispose();
		const h = rmsNorm(x, norm, cfg.rmsEps).slice([], -1);
		return np.dot(h, f32(table.slice(pick)).transpose());
	});

	/**
	 * One decode step for every row. token: [B] int32, cur: scalar int32 (the
	 * cache slot being written, same for all rows), pad: [B] int32.
	 */
	const decodeJit = jit(function decode(
		W: Weights,
		lora: Lora,
		caches: LayerCache[],
		token: np.Array,
		cur: np.Array,
		pad: np.Array
	) {
		const { embed: table, layers, norm } = W;
		const B = token.shape[0];
		// Every model has at least one attention layer; its cache sets the capacity.
		const cap = caches.find((c): c is KV => 'k' in c)!.k.shape[1];
		let x = embed(table.ref, token).reshape([B, 1, cfg.hidden]);
		const positions = cur.ref.astype(np.float32).sub(pad.ref.astype(np.float32)).reshape([B, 1]);
		const [cos, sin] = ropeTables(positions, cfg.headDim, cfg.ropeTheta);
		const slots = np.arange(cap);
		const slotMask = slots.ref.equal(cur.ref).reshape([1, cap, 1, 1]);
		const valid = np.logicalAnd(
			slots.ref.reshape([1, 1, 1, cap]).greaterEqual(pad.reshape([B, 1, 1, 1])),
			slots.reshape([1, 1, 1, cap]).lessEqual(cur)
		);

		const newCaches: LayerCache[] = [];
		for (let i = 0; i < cfg.layers; i++) {
			const lw = layers[i];
			const ll = lora[i];
			const h = rmsNorm(x.ref, lw.inNorm, cfg.rmsEps);
			const cache = caches[i];
			if ('conv' in cache) {
				const [out, state] = shortConvStep(ctx, lw, ll, h, cache.conv);
				newCaches.push({ conv: state });
				x = x.add(out);
			} else {
				const { q, k, v } = qkv(ctx, lw, ll, h, cos.ref, sin.ref);
				const kc = np.where(slotMask.ref, k, cache.k);
				const vc = np.where(slotMask.ref, v, cache.v);
				const attn = nn.dotProductAttention(q, kc.ref, vc.ref, { mask: valid.ref });
				newCaches.push({ k: kc, v: vc });
				x = x.add(linear(lw.o, attn.reshape([B, 1, cfg.heads * cfg.headDim]), ll?.o, ctx.loraScale));
			}
			x = x.add(mlp(ctx, lw, ll, rmsNorm(x.ref, lw.postNorm, cfg.rmsEps)));
		}
		slotMask.dispose();
		valid.dispose();
		cos.dispose();
		sin.dispose();
		const h = rmsNorm(x, norm, cfg.rmsEps).reshape([B, cfg.hidden]);
		return [logitsOf(table, h), newCaches] as [np.Array, LayerCache[]];
	});

	/**
	 * Log-probs of the completion tokens under the (temperature-scaled) policy.
	 * ids: [B, P + L] (prompt left-padded to P, completion right-padded to L),
	 * targets: [B, L] = the completion tokens. -> [B, L]
	 */
	function completionLogprobs(
		W: Weights,
		lora: Lora,
		ids: np.Array,
		pad: np.Array,
		targets: np.Array,
		invTemp: number
	): np.Array {
		const { embed: table, ...rest } = W;
		const [T, L] = [ids.shape[1], targets.shape[1]];
		// Without an adapter we're in full fine-tuning, where the table trains too.
		const x = lora.length ? embed(table.ref, ids) : embedDifferentiable(table.ref, ids);
		const { h } = runSequence(ctx, rest, x, lora, pad, false);
		const hc = h.slice([], [T - L - 1, T - 1]); // positions that predict the completion
		const logp = nn.logSoftmax(logitsOf(table, hc).mul(invTemp), -1);
		// Pick each target's log-prob with a mask rather than a gather (see above).
		return logp.mul(oneHot(targets, cfg.vocab)).sum(-1);
	}

	const logprobsJit = jit(completionLogprobs, { staticArgnums: [5] });

	/**
	 * GRPO loss for one (micro)batch:
	 *
	 *   loss = −Σ_{i,t} w_it [ Â_i · log π(t)  −  β · KL_t(π ‖ π_ref) ] / norm
	 *
	 * `mask` holds per-token weights w_it (1 / (|o_i| · G) for GRPO's
	 * per-answer mean, 0 on padding), so microbatch gradients simply add up.
	 * The KL term is the per-token "k3" estimator from GRPO (DeepSeekMath),
	 * exp(r) - r - 1 with r = log π_ref - log π: always >= 0, zero when the
	 * policy matches the reference. It's the classic RLHF leash that keeps the
	 * model near the original (and so, fluent) while it chases reward.
	 * Returns [loss, per-token log-probs] with gradients w.r.t. `trainable`.
	 */
	function grpoLoss(
		trainable: Weights | Lora,
		frozen: Weights | Record<string, never>,
		ids: np.Array,
		pad: np.Array,
		targets: np.Array,
		mask: np.Array,
		adv: np.Array,
		norm: np.Array,
		refLogp: np.Array,
		beta: np.Array,
		invTemp: number
	): [np.Array, np.Array] {
		// LoRA mode trains the adapter list against frozen weights; full mode
		// trains the weights themselves (and `frozen` is an empty placeholder).
		const [W, lora] = Array.isArray(trainable)
			? [frozen as Weights, trainable]
			: [trainable, [] as Lora];
		const B = targets.shape[0];
		const logp = completionLogprobs(W, lora, ids, pad, targets, invTemp);
		const r = refLogp.sub(logp.ref);
		const kl = np.exp(r.ref).sub(r).sub(1);
		const objective = logp.ref.mul(adv.reshape([B, 1])).sub(kl.mul(beta));
		const loss = objective.mul(mask).sum().div(norm).mul(-1);
		return [loss, logp];
	}

	const trainStepJit = jit(valueAndGrad(grpoLoss, { hasAux: true }), { staticArgnums: [10] });

	// jit trees can't contain null, so "no adapter" is an empty list inside.
	return {
		cfg,
		loraScale: ctx.loraScale,
		prefill: (W: Weights, lora: Lora | null, ids: np.Array, pad: np.Array, capacity: number) =>
			prefillJit(W, lora ?? [], ids, pad, capacity),
		extend: (W: Weights, prefix: KV[], prefixPad: np.Array, ids: np.Array, pad: np.Array, pick: np.Array) =>
			extendJit(W, prefix, prefixPad, ids, pad, pick),
		decode: (W: Weights, lora: Lora | null, caches: LayerCache[], token: np.Array, cur: np.Array, pad: np.Array) =>
			decodeJit(W, lora ?? [], caches, token, cur, pad),
		logprobs: (W: Weights, lora: Lora | null, ids: np.Array, pad: np.Array, targets: np.Array, invTemp: number) =>
			logprobsJit(W, lora ?? [], ids, pad, targets, invTemp),
		trainStep: (
			trainable: Weights | Lora,
			frozen: Weights | null,
			ids: np.Array,
			pad: np.Array,
			targets: np.Array,
			mask: np.Array,
			adv: np.Array,
			norm: np.Array,
			refLogp: np.Array,
			beta: np.Array,
			invTemp: number
		) => trainStepJit(trainable, frozen ?? {}, ids, pad, targets, mask, adv, norm, refLogp, beta, invTemp)
	};
}
