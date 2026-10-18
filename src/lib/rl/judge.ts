import { numpy as np, tree } from '@jax-js/jax';

import type { CompiledModel, KV } from '../models/llama';
import type { Tokenizer } from '../models/tokenizer';
import type { Weights } from '../models/weights';
import { bucket, leftPad } from './generate';

/** An answer's reward, in [0, 1]. */
export type Verdict = { score: number };

const JUDGE_BUCKET = 64;

/*
 * The judge never writes text. We read its next-token logits for a fixed pair
 * of tokens, in one of two modes:
 *
 * - Pointwise: it sees one answer and is asked "does it follow the
 *   guidelines? Yes or No": score = P(Yes) = sigmoid(logit Yes − logit No).
 * - Pairwise: it sees two answers and is asked "which follows the guidelines
 *   better? A or B": P(A) = sigmoid(logit A − logit B).
 *
 * Which is sharper depends on the judge. Picking the answer that follows the
 * guidelines from a pair where only one does, Qwen3 1.7B scored 70–100% pointwise against
 * 52–90% pairwise; Qwen2.5 1.5B was the other way around. Pointwise also costs
 * one row per answer rather than two per pair. Small judges lean hard toward
 * one position (Qwen3 picked the second answer ~70% of the time), so pairs are
 * asked in a random order (bias cancels on average) or, better, in both.
 * The instructions, guidelines and question are the same for every answer, so
 * they're prefilled once and each row only pays for its answer(s).
 */

export type JudgeMode = 'pointwise' | 'pairwise';

const SYSTEM: Record<JudgeMode, string> = {
	pointwise: 'You check AI answers against guidelines.',
	pairwise: 'You compare AI answers against guidelines.'
};

function judgePrompt(mode: JudgeMode, constitution: string, question: string): string {
	return (
		`<|im_start|>system\n${SYSTEM[mode]}<|im_end|>\n<|im_start|>user\n` +
		`GUIDELINES:\n${constitution.trim()}\n\nQUESTION:\n${question.trim()}\n\n`
	);
}

function checkSuffix(answer: string, assistantPrefix: string): string {
	return (
		`ANSWER:\n${answer.trim() || '(empty)'}\n\n` +
		'Does the ANSWER follow the GUIDELINES? Reply with only Yes or No.<|im_end|>\n<|im_start|>assistant\n' +
		assistantPrefix
	);
}

function compareSuffix(a: string, b: string, assistantPrefix: string): string {
	return (
		`ANSWER A:\n${a.trim() || '(empty)'}\n\nANSWER B:\n${b.trim() || '(empty)'}\n\n` +
		'Which answer follows the GUIDELINES better? Reply with only A or B.<|im_end|>\n<|im_start|>assistant\n' +
		assistantPrefix
	);
}

export type Match = {
	a: number;
	b: number;
	/** P(judge picks a over b), averaged over the orders that were asked. */
	p: number;
	/** P(picks A) with a shown first, and with b shown first (whichever were asked). */
	aFirst?: number;
	bFirst?: number;
};

/** Pairs so every answer plays `perAnswer` matches (all pairs if perAnswer >= n - 1). */
export function schedulePairs(n: number, perAnswer: number): [number, number][] {
	const all: [number, number][] = [];
	for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) all.push([i, j]);
	if (perAnswer >= n - 1) return all;
	for (let i = all.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[all[i], all[j]] = [all[j], all[i]];
	}
	const played = new Array(n).fill(0);
	const out: [number, number][] = [];
	// Least-played answers first, so everyone ends up with about perAnswer matches.
	for (let round = 0; round < perAnswer; round++) {
		for (const [i, j] of all) {
			if (out.some(([x, y]) => x === i && y === j)) continue;
			if (played[i] <= round && played[j] <= round) {
				out.push([i, j]);
				played[i]++;
				played[j]++;
			}
		}
	}
	return out;
}

/**
 * Rows per judge call. Peak GPU memory grows ~1 GB per row with Qwen3 1.7B
 * (16 rows peaked at ~22 GB and swapped on a 16 GB Mac, 8x slower than 4).
 */
const COMPARE_BATCH = 4;
const SUFFIX_BUCKET = 32;

/** The judge's shared prompt (instructions, guidelines, question), prefilled once. */
export type JudgePrefix = { kv: KV[]; pad: number };

export async function judgePrefix(
	model: CompiledModel,
	weights: Weights,
	tok: Tokenizer,
	padToken: number,
	mode: JudgeMode,
	constitution: string,
	question: string
): Promise<JudgePrefix> {
	const ids = tok.encode(judgePrompt(mode, constitution, question));
	const P = bucket(ids.length, JUDGE_BUCKET);
	const { row, pad } = leftPad(ids, P, padToken);
	const [logits, caches] = model.prefill(
		tree.ref(weights),
		null,
		np.array(new Int32Array(row), { shape: [1, P], dtype: np.int32 }),
		np.array(new Int32Array([pad]), { dtype: np.int32 }),
		P
	);
	logits.dispose();
	const kv = caches as KV[]; // judges are attention-only
	await Promise.all(kv.map(({ k, v }) => Promise.all([k.blockUntilReady(), v.blockUntilReady()])));
	return { kv, pad };
}

/** Logits of the two `pick` tokens after each row continues the prefix: [rows, 2]. */
async function pickLogits(
	model: CompiledModel,
	weights: Weights,
	prefix: JudgePrefix,
	rows: number[][],
	padToken: number,
	pick: [number, number]
): Promise<Float32Array> {
	const S = bucket(Math.max(...rows.map((r) => r.length)), SUFFIX_BUCKET);
	const padded = rows.map((r) => leftPad(r, S, padToken));
	const logits = model.extend(
		tree.ref(weights),
		tree.ref(prefix.kv),
		np.array(prefix.pad, { dtype: np.int32 }),
		np.array(new Int32Array(padded.flatMap((r) => r.row)), { shape: [rows.length, S], dtype: np.int32 }),
		np.array(new Int32Array(padded.map((r) => r.pad)), { dtype: np.int32 }),
		np.array(new Int32Array(pick), { dtype: np.int32 })
	);
	return (await logits.data()) as Float32Array;
}

/**
 * Pointwise: each answer's log-odds that it follows the guidelines,
 * logit(Yes) − logit(No). P(Yes) is sigmoid of that.
 */
export async function scoreAnswers(
	model: CompiledModel,
	weights: Weights,
	tok: Tokenizer,
	padToken: number,
	constitution: string,
	question: string,
	answers: string[],
	opts: {
		assistantPrefix?: string;
		/** Called as each batch of answers is scored, with all log-odds so far. */
		onScores?: (logOdds: number[]) => void;
		/** A cached prefix for this question (kept by the caller), else one is built and freed. */
		prefix?: JudgePrefix;
	}
): Promise<number[]> {
	const pick: [number, number] = [tok.encode('Yes')[0], tok.encode('No')[0]];
	const prefix = opts.prefix ?? (await judgePrefix(model, weights, tok, padToken, 'pointwise', constitution, question));
	const rows = answers.map((a) => tok.encode(checkSuffix(a, opts.assistantPrefix ?? '')));
	const logOdds: number[] = [];
	try {
		for (let start = 0; start < rows.length; start += COMPARE_BATCH) {
			const data = await pickLogits(model, weights, prefix, rows.slice(start, start + COMPARE_BATCH), padToken, pick);
			for (let b = 0; 2 * b < data.length; b++) logOdds.push(data[2 * b] - data[2 * b + 1]);
			opts.onScores?.(logOdds);
		}
	} finally {
		if (!opts.prefix) tree.dispose(prefix.kv);
	}
	return logOdds;
}

export async function compareAnswers(
	model: CompiledModel,
	weights: Weights,
	tok: Tokenizer,
	padToken: number,
	constitution: string,
	question: string,
	answers: string[],
	pairs: [number, number][],
	opts: {
		bothOrders: boolean;
		/** Opens the judge's turn, e.g. Qwen3's empty think block. */
		assistantPrefix?: string;
		onMatch?: (match: Match) => void;
		/** A cached prefix for this question (kept by the caller), else one is built and freed. */
		prefix?: JudgePrefix;
	}
): Promise<Match[]> {
	const [aId, bId] = [tok.encode('A')[0], tok.encode('B')[0]];
	const owned = !opts.prefix;
	const { kv: prefix, pad: prefixPad } =
		opts.prefix ?? (await judgePrefix(model, weights, tok, padToken, 'pairwise', constitution, question));

	// Either two rows per pair, (a, b) and (b, a), or one row in a random order.
	// A random order cancels the judge's position bias on average at half the cost.
	const orders: boolean[][] = pairs.map(() => (opts.bothOrders ? [true, false] : [Math.random() < 0.5]));
	const rows = pairs.flatMap(([i, j], k) =>
		orders[k].map((aFirst) =>
			tok.encode(
				aFirst
					? compareSuffix(answers[i], answers[j], opts.assistantPrefix ?? '')
					: compareSuffix(answers[j], answers[i], opts.assistantPrefix ?? '')
			)
		)
	);
	const rowsBefore = orders.map((_, k) => orders.slice(0, k).reduce((s, o) => s + o.length, 0));
	const pA: number[] = [];
	const matches: Match[] = [];
	try {
		for (let start = 0; start < rows.length; start += COMPARE_BATCH) {
			const chunk = rows.slice(start, start + COMPARE_BATCH);
			const data = await pickLogits(model, weights, { kv: prefix, pad: prefixPad }, chunk, padToken, [aId, bId]); // logit(A), logit(B)
			for (let b = 0; b < chunk.length; b++) pA.push(1 / (1 + Math.exp(data[2 * b + 1] - data[2 * b])));
			// Emit every pair whose orderings are all done.
			while (matches.length < pairs.length && rowsBefore[matches.length] + orders[matches.length].length <= pA.length) {
				const k = matches.length;
				const [i, j] = pairs[k];
				const match: Match = { a: i, b: j, p: 0 };
				const wins: number[] = [];
				orders[k].forEach((aFirst, r) => {
					const pa = pA[rowsBefore[k] + r];
					if (aFirst) match.aFirst = pa;
					else match.bFirst = pa;
					wins.push(aFirst ? pa : 1 - pa);
				});
				match.p = wins.reduce((s, w) => s + w, 0) / wins.length;
				matches.push(match);
				opts.onMatch?.(match);
			}
		}
	} finally {
		if (owned) tree.dispose(prefix);
	}
	return matches;
}

/** Each answer's reward: its average probability of being picked across its matches. */
export function winRates(n: number, matches: Match[]): number[] {
	const won = new Array(n).fill(0);
	const played = new Array(n).fill(0);
	for (const m of matches) {
		won[m.a] += m.p;
		won[m.b] += 1 - m.p;
		played[m.a]++;
		played[m.b]++;
	}
	return won.map((w, i) => (played[i] ? w / played[i] : 0.5));
}
