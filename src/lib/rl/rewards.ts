/**
 * Where a pass's rewards come from. The three kinds of feedback in RL
 * fine-tuning:
 *
 * - ai (RLAIF): a judge model scores each answer against a judge prompt.
 * - code (RLVR, "verifiable rewards"): a JavaScript function scores each answer.
 * - human (RLHF): you pick the best answer of each pass.
 */
export type RewardKind = 'ai' | 'code' | 'human';

export type RewardFn = (answer: string, prompt: string) => number;

/**
 * Compile `code`, the body of `function (answer, prompt)`, into a reward in
 * [0, 1]. true and false count as 1 and 0; numbers are clamped.
 */
export function compileReward(code: string): RewardFn {
	const fn = new Function('answer', 'prompt', code) as (answer: string, prompt: string) => unknown;
	return (answer, prompt) => {
		const value = fn(answer, prompt);
		const n = Number(value);
		if (typeof value !== 'boolean' && (typeof value !== 'number' || !Number.isFinite(n))) {
			throw new Error(`The reward code returned ${JSON.stringify(value)}; return a number or true / false.`);
		}
		return Math.min(1, Math.max(0, n));
	};
}

