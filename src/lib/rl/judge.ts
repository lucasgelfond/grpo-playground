import { numpy as np, tree } from '@jax-js/jax';

import type { CompiledModel, KV } from '../models/llama';
import { chatPrompt } from '../models/registry';
import type { Tokenizer } from '../models/tokenizer';
import type { Weights } from '../models/weights';
import { bucket, leftPad } from './generate';

/**
 * The judge never writes text. We show it the constitution, the question and
 * one answer, ask "does the answer follow the guidelines? Yes or No", and
 * read its next-token probabilities for "Yes" and "No". The reward is
 * P(Yes) renormalized over the two, which costs one forward pass.
 *
 * We A/B tested this against asking for a 0-9 digit: small judges rank
 * answers far more cleanly as a yes/no question (P(yes) of 1.00 / 0.73 / 0.10
 * / 0.00 for good / ok / rambling / broken answers, vs. muddled digit scores).
 */

export const JUDGE_SYSTEM = 'You grade AI answers against guidelines.';

export function judgePrompt(constitution: string, question: string, answer: string): string {
	return chatPrompt([
		{ role: 'system', content: JUDGE_SYSTEM },
		{
			role: 'user',
			content:
				`GUIDELINES:\n${constitution.trim()}\n\nQUESTION:\n${question.trim()}\n\n` +
				`ANSWER:\n${answer.trim() || '(empty)'}\n\n` +
				'Does the ANSWER follow the GUIDELINES well? Reply with only Yes or No.'
		}
	]);
}

export type Verdict = {
	/** P(Yes) renormalized over {Yes, No}: the reward, in [0, 1]. */
	score: number;
	/** Total probability the judge put on "Yes" or "No" at all (sanity check). */
	mass: number;
	/** When the reward mixes sources, the parts it came from. */
	judge?: number;
	rule?: number;
};

const JUDGE_BUCKET = 64;
/** Rows per judge forward pass; bounds attention/MLP activation memory. */
const JUDGE_BATCH = 4;

export async function judgeAnswers(
	model: CompiledModel,
	weights: Weights,
	tok: Tokenizer,
	padToken: number,
	constitution: string,
	question: string,
	answers: string[],
	onVerdict?: (index: number, verdict: Verdict) => void
): Promise<Verdict[]> {
	const [yesId, noId] = [tok.encode('Yes')[0], tok.encode('No')[0]];
	const encoded = answers.map((a) => tok.encode(judgePrompt(constitution, question, a)));
	const verdicts: Verdict[] = [];

	for (let start = 0; start < encoded.length; start += JUDGE_BATCH) {
		const chunk = encoded.slice(start, start + JUDGE_BATCH);
		const T = bucket(Math.max(...chunk.map((ids) => ids.length)), JUDGE_BUCKET);
		const rows = chunk.map((ids) => leftPad(ids, T, padToken));
		const ids = np.array(new Int32Array(rows.flatMap((r) => r.row)), {
			shape: [chunk.length, T],
			dtype: np.int32
		});
		const pad = np.array(new Int32Array(rows.map((r) => r.pad)), { dtype: np.int32 });
		const logits = model.lastLogits(tree.ref(weights), null, ids, pad);
		const data = (await logits.data()) as Float32Array;
		const V = data.length / chunk.length;

		for (let b = 0; b < chunk.length; b++) {
			const row = data.subarray(b * V, (b + 1) * V);
			let max = -Infinity;
			for (let i = 0; i < V; i++) if (row[i] > max) max = row[i];
			let total = 0;
			for (let i = 0; i < V; i++) total += Math.exp(row[i] - max);
			const yes = Math.exp(row[yesId] - max) / total;
			const no = Math.exp(row[noId] - max) / total;
			const verdict = { score: yes / (yes + no), mass: yes + no };
			verdicts.push(verdict);
			onVerdict?.(start + b, verdict);
		}
	}
	return verdicts;
}

// --- Choosing between answers ---------------------------------------------------------
//
// Pairwise judging is sharper than scoring one answer at a time (92% vs a 0.68
// AUC on our test sets with Qwen2.5 1.5B). Small judges lean toward whichever
// answer comes first: ask each pair in a random order (bias cancels on
// average) or, more expensively, in both orders and average.
// The instructions, guidelines and question are the same for every pair, so
// they're prefilled once and each comparison only pays for its two answers.

export const COMPARE_SYSTEM = 'You compare AI answers against guidelines.';

function comparePrefix(constitution: string, question: string): string {
	return (
		`<|im_start|>system\n${COMPARE_SYSTEM}<|im_end|>\n<|im_start|>user\n` +
		`GUIDELINES:\n${constitution.trim()}\n\nQUESTION:\n${question.trim()}\n\n`
	);
}

function compareSuffix(a: string, b: string): string {
	return (
		`ANSWER A:\n${a.trim() || '(empty)'}\n\nANSWER B:\n${b.trim() || '(empty)'}\n\n` +
		'Which answer follows the GUIDELINES better? Reply with only A or B.<|im_end|>\n<|im_start|>assistant\n'
	);
}

/** The full text the judge sees for one ordering of a pair (for display). */
export function comparePrompt(constitution: string, question: string, a: string, b: string): string {
	return comparePrefix(constitution, question) + compareSuffix(a, b);
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

/** All of a pass's comparisons (6 pairs + 4 vs original) fit in one batch. */
const COMPARE_BATCH = 16;
const SUFFIX_BUCKET = 32;

/** The judge's shared prompt (instructions, guidelines, question), prefilled once. */
export type JudgePrefix = { kv: KV[]; pad: number };

export async function judgePrefix(
	model: CompiledModel,
	weights: Weights,
	tok: Tokenizer,
	padToken: number,
	constitution: string,
	question: string
): Promise<JudgePrefix> {
	const ids = tok.encode(comparePrefix(constitution, question));
	const P = bucket(ids.length, JUDGE_BUCKET);
	const { row, pad } = leftPad(ids, P, padToken);
	const [logits, kv] = model.prefill(
		tree.ref(weights),
		null,
		np.array(new Int32Array(row), { shape: [1, P], dtype: np.int32 }),
		np.array(new Int32Array([pad]), { dtype: np.int32 }),
		P
	);
	logits.dispose();
	await Promise.all(kv.map(({ k, v }) => Promise.all([k.blockUntilReady(), v.blockUntilReady()])));
	return { kv, pad };
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
		onMatch?: (match: Match) => void;
		/** A cached prefix for this question (kept by the caller), else one is built and freed. */
		prefix?: JudgePrefix;
	}
): Promise<Match[]> {
	const [aId, bId] = [tok.encode('A')[0], tok.encode('B')[0]];
	const owned = !opts.prefix;
	const { kv: prefix, pad: prefixPad } =
		opts.prefix ?? (await judgePrefix(model, weights, tok, padToken, constitution, question));

	// Either two rows per pair, (a, b) and (b, a), or one row in a random order.
	// A random order cancels the judge's position bias on average at half the cost.
	const orders: boolean[][] = pairs.map(() => (opts.bothOrders ? [true, false] : [Math.random() < 0.5]));
	const rows = pairs.flatMap(([i, j], k) =>
		orders[k].map((aFirst) =>
			tok.encode(aFirst ? compareSuffix(answers[i], answers[j]) : compareSuffix(answers[j], answers[i]))
		)
	);
	const rowsBefore = orders.map((_, k) => orders.slice(0, k).reduce((s, o) => s + o.length, 0));
	const pA: number[] = [];
	const matches: Match[] = [];
	try {
		for (let start = 0; start < rows.length; start += COMPARE_BATCH) {
			const chunk = rows.slice(start, start + COMPARE_BATCH);
			const S = bucket(Math.max(...chunk.map((r) => r.length)), SUFFIX_BUCKET);
			const padded = chunk.map((r) => leftPad(r, S, padToken));
			const logits = model.extend(
				tree.ref(weights),
				tree.ref(prefix),
				np.array(prefixPad, { dtype: np.int32 }),
				np.array(new Int32Array(padded.flatMap((r) => r.row)), { shape: [chunk.length, S], dtype: np.int32 }),
				np.array(new Int32Array(padded.map((r) => r.pad)), { dtype: np.int32 }),
				np.array(new Int32Array([aId, bId]), { dtype: np.int32 })
			);
			const data = (await logits.data()) as Float32Array; // [rows, 2]: logit(A), logit(B)
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
