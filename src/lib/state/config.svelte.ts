import { DEFAULT_TASK } from '$lib/presets';
import type { RuleId } from '$lib/rl/rules';
import type { Mode } from '$lib/rl/trainer';

/**
 * Everything the user configures before training. Persisted to localStorage
 * so a reload keeps the prompts and constitution.
 */
export type Config = {
	policyId: string;
	judgeId: string;
	mode: Mode;
	prompts: string[];
	promptOrder: 'sequential' | 'shuffle';
	constitution: string;
	/** Rule check for the loaded task preset, if it has one. */
	rule: RuleId | null;
	/** How many head-to-head matches each answer plays per pass. */
	matchesPerAnswer: number;
	/** Ask every pair in both A/B orders (2× judge cost) instead of one random order. */
	bothOrders: boolean;
	groupSize: number;
	maxNew: number;
	temperature: number;
	loraLearningRate: number;
	fullLearningRate: number;
	loraRank: number;
	loraAlpha: number;
	/** Weight of the KL penalty toward the original model (0 turns it off). */
	klBeta: number;
	// DPO works but isn't exposed in the UI (mostly because GRPO playground was a better name, LOL). if you do ?mode=dpo or go to dpo.lucasgelfond.online you can try the DPO version
	/** 'dpo' trains on the judge's preference pairs instead of GRPO's group advantages. */
	algorithm: 'grpo' | 'dpo';
	/** DPO's β: how strongly the implicit reward margin is pushed. */
	dpoBeta: number;
	/** DPO: pairs trained on per pass (the judge's most confident). */
	dpoPairs: number;
	/** DPO: skip pairs the judge called closer than this to a coin flip (|p − 0.5|). */
	dpoMinConfidence: number;
};

export const DEFAULT_CONFIG: Config = {
	policyId: 'lfm2-350m',
	judgeId: 'qwen3-1.7b',
	mode: 'lora',
	prompts: DEFAULT_TASK.prompts.slice(0, 5),
	promptOrder: 'shuffle',
	constitution: DEFAULT_TASK.constitution,
	rule: DEFAULT_TASK.rule ?? null,
	// Every pair, asked in both A/B orders: the judge's lean toward whichever
	// answer comes first cancels instead of adding noise to the reward.
	matchesPerAnswer: 4,
	bothOrders: true,
	// 5 answers: 10 pairs among them plus 5 against the original model.
	groupSize: 5,
	maxNew: 128,
	// Warm, not hot: enough variety within each group for GRPO to have something
	// to prefer, without the 135M model's incoherent samples at 1.0.
	temperature: 0.7,
	loraLearningRate: 3e-4,
	fullLearningRate: 1e-5,
	loraRank: 16,
	loraAlpha: 32,
	klBeta: 0.05,
	algorithm: 'grpo',
	dpoBeta: 0.1,
	dpoPairs: 4,
	dpoMinConfidence: 0.1
};

// v5: everyone restarts on the current defaults (LFM2 350M judged by Qwen3
// 1.7B, both-order judging, 5 answers at temperature 0.7).
const KEY = 'grpo-playground:config:v5';

function load(): Config {
	try {
		const raw = localStorage.getItem(KEY);
		if (raw) return { ...DEFAULT_CONFIG, ...JSON.parse(raw) };
	} catch {
		// Storage can be unavailable (private mode, blocked site data).
	}
	return structuredClone(DEFAULT_CONFIG);
}

export const config: Config = $state(load());

$effect.root(() => {
	$effect(() => {
		const snapshot = JSON.stringify(config);
		try {
			localStorage.setItem(KEY, snapshot);
		} catch {
			// Ignore: persistence is a convenience.
		}
	});
});

export function learningRate(c: Config): number {
	return c.mode === 'lora' ? c.loraLearningRate : c.fullLearningRate;
}
