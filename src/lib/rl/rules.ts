/**
 * Rule-based ("verifiable") rewards: code that checks a property of the answer
 * directly. A 135M model rarely nails these behaviors outright, so each rule
 * gives partial credit where it can, which keeps GRPO's within-group
 * differences informative even before any sample fully passes.
 */

export type RuleId = 'brevity' | 'yes-no' | 'plain-prose' | 'dont-know' | 'follow-up';

const MARKDOWN = /(^|\n)\s*([-*•]|\d+[.)])\s|\*\*|(^|\n)#/;
const DONT_KNOW =
	/(don't|do not|can't|cannot|can not) (know|tell|access|predict|say|see)|not sure|no way (to|of) know|not able to|unable to/i;

export const RULES: Record<RuleId, (text: string, finished: boolean) => number> = {
	// Finished, and shorter is better.
	brevity: (t, finished) => (finished ? Math.max(0, 1 - words(t) / 80) : 0),
	// First word is Yes or No; half credit if over 40 words.
	'yes-no': (t) => (/^(yes|no)\b/i.test(t.trim()) ? 1 : 0) * (words(t) <= 40 ? 1 : 0.5),
	// No lists, headings or bold; 30% credit if cut off.
	'plain-prose': (t, finished) => (MARKDOWN.test(t) ? 0 : 1) * (finished ? 1 : 0.3),
	'dont-know': (t) => (DONT_KNOW.test(t) ? 1 : 0),
	'follow-up': (t, finished) => (finished && t.trim().endsWith('?') ? 1 : 0)
};

function words(t: string): number {
	return t.trim().split(/\s+/).filter(Boolean).length;
}

/**
 * Checks that chart what training changes, matched to a preset's goal
 * (e.g. markdown use dropping, or answers starting to end with a question).
 */
export const CHECKS: { id: RuleId; label: string; test: (text: string, finished: boolean) => boolean }[] = [
	{ id: 'plain-prose', label: 'Lists / markdown presence', test: (t) => MARKDOWN.test(t) },
	{ id: 'follow-up', label: 'Ends with question', test: (t, finished) => finished && t.trim().endsWith('?') },
	{ id: 'yes-no', label: 'Starts with Yes/No', test: (t) => /^(yes|no)\b/i.test(t.trim()) },
	{ id: 'brevity', label: 'One sentence', test: (t, finished) => finished && (t.trim().match(/[.!?](\s|$)/g)?.length ?? 0) <= 1 }
];
