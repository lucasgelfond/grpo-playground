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

export const RULES: Record<RuleId, { label: string; score: (text: string, finished: boolean) => number }> = {
	brevity: {
		label: 'finished, and shorter is better (1 − words/80)',
		score: (t, finished) => (finished ? Math.max(0, 1 - words(t) / 80) : 0)
	},
	'yes-no': {
		label: 'first word is Yes or No (half credit if over 40 words)',
		score: (t) => (/^(yes|no)\b/i.test(t.trim()) ? 1 : 0) * (words(t) <= 40 ? 1 : 0.5)
	},
	'plain-prose': {
		label: 'no lists, headings or bold (30% credit if cut off)',
		score: (t, finished) => (MARKDOWN.test(t) ? 0 : 1) * (finished ? 1 : 0.3)
	},
	'dont-know': {
		label: 'says it doesn’t or can’t know',
		score: (t) => (DONT_KNOW.test(t) ? 1 : 0)
	},
	'follow-up': {
		label: 'finished, and the last character is "?"',
		score: (t, finished) => (finished && t.trim().endsWith('?') ? 1 : 0)
	}
};

function words(t: string): number {
	return t.trim().split(/\s+/).filter(Boolean).length;
}

/**
 * Yes/no checks run on every answer of every pass, whatever the judge prompt,
 * so the charts can show what training actually changes (e.g. markdown use
 * dropping, or answers starting to end with a question).
 */
export const HEURISTICS: { id: string; label: string; test: (text: string, finished: boolean) => boolean }[] = [
	{ id: 'markdown', label: 'Uses lists / markdown', test: (t) => MARKDOWN.test(t) },
	{ id: 'question', label: 'Ends with a question', test: (t, finished) => finished && t.trim().endsWith('?') },
	{ id: 'yes-no', label: 'Starts with Yes/No', test: (t) => /^(yes|no)\b/i.test(t.trim()) },
	{ id: 'dont-know', label: 'Admits not knowing', test: (t) => DONT_KNOW.test(t) },
	{
		id: 'one-sentence',
		label: 'One sentence',
		test: (t, finished) => finished && (t.trim().match(/[.!?](\s|$)/g)?.length ?? 0) <= 1
	},
	{ id: 'cut-off', label: 'Cut off', test: (_, finished) => !finished }
];
