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
};

export const DEFAULT_CONFIG: Config = {
	policyId: 'smollm2-135m',
	judgeId: 'qwen2.5-1.5b',
	mode: 'lora',
	prompts: DEFAULT_TASK.prompts.slice(0, 5),
	promptOrder: 'shuffle',
	constitution: DEFAULT_TASK.constitution,
	rule: DEFAULT_TASK.rule ?? null,
	matchesPerAnswer: 3,
	bothOrders: false,
	// Few answers per prompt keeps the judge cheap (a round robin of 4 is 6
	// pairs): more prompts per minute beats a finer ranking of each group.
	groupSize: 4,
	// Hot sampling: enough variety within each group for GRPO to have something to prefer.
	maxNew: 128,
	temperature: 1.0,
	loraLearningRate: 3e-4,
	fullLearningRate: 1e-5,
	loraRank: 16,
	loraAlpha: 32,
	klBeta: 0.05
};

// v3: start everyone on the "yes or no first" preset.
const KEY = 'grpo-playground:config:v3';
/** Where settings lived before the rename; read once so nothing is lost. */
const OLD_KEY = 'jax-rl-model:config:v3';

function load(): Config {
	try {
		const raw = localStorage.getItem(KEY) ?? localStorage.getItem(OLD_KEY);
		if (raw) {
			const saved = JSON.parse(raw);
			// Settings saved before the smaller-group default: move them to 4 answers.
			if (!('bothOrders' in saved)) saved.groupSize = DEFAULT_CONFIG.groupSize;
			// Answers are cut off at 128 tokens (it was 256 for a while).
			saved.maxNew = DEFAULT_CONFIG.maxNew;
			if (saved.judgeId !== DEFAULT_CONFIG.judgeId) saved.judgeId = DEFAULT_CONFIG.judgeId;
			// Only the 135M model is offered for training for now.
			if (saved.policyId !== DEFAULT_CONFIG.policyId) saved.policyId = DEFAULT_CONFIG.policyId;
			return { ...DEFAULT_CONFIG, ...saved };
		}
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
