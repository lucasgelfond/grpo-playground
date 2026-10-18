import { DEFAULT_PRESET, PRESETS, type Preset } from '$lib/presets';
import type { JudgeMode } from '$lib/rl/judge';
import type { RewardKind } from '$lib/rl/rewards';
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
	/** What the training is for, in a line (up to TASK_MAX characters): shown on the judge node. */
	task: string;
	/** Where rewards come from: an AI judge, reward code, or you. */
	reward: RewardKind;
	/** AI: what the judge rewards. */
	constitution: string;
	/** Code: the body of `function (answer, prompt)`, returning a number or true / false. */
	rewardCode: string;
	/** Charted on the train page (from the preset): the share of answers this code calls true. */
	check: Preset['check'] | null;
	/** Score each answer on its own (yes/no), or compare answers in pairs (A/B). */
	judgeMode: JudgeMode;
	/** Pairwise: how many head-to-head matches each answer plays per pass. */
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

export const TASK_MAX = 80;

// SmolLM2 360M, not LFM2 350M: LFM2 rambles past any short token limit
// (under 13% of its answers finish in 128 tokens), so nearly every answer
// was cut off and the judge had nothing to prefer.
export const DEFAULT_CONFIG: Config = {
	policyId: 'smollm2-360m',
	judgeId: 'qwen3-1.7b',
	mode: 'lora',
	prompts: DEFAULT_PRESET.prompts.slice(0, 5),
	promptOrder: 'shuffle',
	task: DEFAULT_PRESET.task,
	reward: DEFAULT_PRESET.reward,
	constitution: DEFAULT_PRESET.constitution ?? '',
	rewardCode: PRESETS.find((p) => p.reward === 'code')!.code!,
	check: DEFAULT_PRESET.check ?? null,
	// Qwen3 1.7B is more accurate scoring one answer at a time, and it's a
	// fifth of the judge rows (6 instead of 30).
	judgeMode: 'pointwise',
	// Pairwise: every pair, asked in both A/B orders, so the judge's lean toward
	// one position cancels instead of adding noise to the reward.
	matchesPerAnswer: 4,
	bothOrders: true,
	groupSize: 5,
	// Short answers finish, so the judge sees whole answers, and passes are fast.
	maxNew: 64,
	temperature: 1.0,
	// High for LoRA, so the behavior changes within a demo's 10-30 passes.
	loraLearningRate: 1e-3,
	fullLearningRate: 1e-5,
	loraRank: 16,
	loraAlpha: 32,
	klBeta: 0.05,
	algorithm: 'grpo',
	dpoBeta: 0.1,
	dpoPairs: 4,
	dpoMinConfidence: 0.1
};

// v7: everyone restarts on the current defaults (SmolLM2 360M, an AI judge
// rewarding answers without filler, 64 tokens at temperature 1.0, LoRA at 1e-3).
const KEY = 'grpo-playground:config:v7';

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

/** Load a preset: its reward and (the first five of) its prompts. */
export function applyPreset(c: Config, p: Preset) {
	c.task = p.task;
	c.reward = p.reward;
	c.prompts = p.prompts.slice(0, 5);
	if (p.constitution !== undefined) c.constitution = p.constitution;
	if (p.code !== undefined) c.rewardCode = p.code;
	// Code rewards chart their own pass rate.
	c.check = p.check ?? (p.code ? { label: p.label, code: p.code } : null);
}

/** DPO trains on preferences between answers, so it always judges in pairs. */
export function judgeModeOf(c: Config): JudgeMode {
	return c.algorithm === 'dpo' ? 'pairwise' : c.judgeMode;
}

export function learningRate(c: Config): number {
	return c.mode === 'lora' ? c.loraLearningRate : c.fullLearningRate;
}
