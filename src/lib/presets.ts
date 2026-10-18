/**
 * Presets for each kind of reward (see rl/rewards.ts). Each sets the prompts
 * and the reward: a judge prompt, reward code, or nothing (you pick). A
 * `check` is charted on the train page: how often answers do the thing.
 *
 * Measured on SmolLM2 360M with the defaults (LoRA at 1e-3, 5 answers of 64
 * tokens at temperature 1.0), on held-out prompts:
 *
 *   AI judge, no filler opener   33% -> 100% by pass 10 ("Sure! I'd be happy
 *                                to…" -> "Inflation is the general rise…").
 *
 * Tried and dropped as AI demos: yes or no first (the model learns the format,
 * not the facts: one-word answers, right half the time), optimist (answers
 * went flat, not cheerful), asks a clarifying question first (barely moved).
 */
import type { RewardKind } from './rl/rewards';

export type Preset = {
	id: string;
	label: string;
	reward: RewardKind;
	prompts: string[];
	/** AI: the judge prompt. */
	constitution?: string;
	/** Code: the body of `function (answer, prompt)`. */
	code?: string;
	/** What the train page charts: the share of answers for which this code returns true. */
	check?: { label: string; code: string };
};

const NO_FILLER = `return !/^\\W*(sure|certainly|absolutely|of course|great|good question|i'?d be (happy|glad|delighted)|i'?m (happy|glad)|yes|okay|ok|hello|hi|wow|what a|ah|oh|well)\\b/i.test(answer.trim());`;

const FILLER_PROMPTS = [
			'Can you explain how rainbows form?',
			'Can you give me a tip for studying?',
			'Can you help me write a birthday message for my sister?',
			'Can you tell me a fun fact about octopuses?',
			'Can you explain what a black hole is?',
			'Can you suggest a name for my cat?',
			'Can you explain why the sky is blue?',
			'Can you give me a quick dinner idea?',
			'Can you tell me how to boil an egg?',
			'Can you explain what DNA is?',
			'Can you give me a tip for sleeping better?',
			'Can you explain how vaccines work?',
			'Can you tell me a joke?',
			'Can you explain what inflation is?',
			'Can you suggest a weekend activity?',
			'Can you explain how plants grow?',
			'Can you give me a tip for public speaking?',
			'Can you tell me about the Moon?',
			'Can you explain what gravity is?',
			'Can you suggest a good habit to start?'
];

export const PRESETS: Preset[] = [
	{
		id: 'no-filler',
		label: 'No filler opener',
		reward: 'ai',
		constitution: `Reward answers that get straight to the point. The answer must NOT open with filler like "Sure!", "Certainly!", "Absolutely!", "Of course!", "Great question!" or "I'd be happy to help".
Penalize any answer whose first words are filler or enthusiasm instead of content.`,
		prompts: FILLER_PROMPTS,
		check: { label: 'No filler opener', code: NO_FILLER }
	},
	{
		id: 'no-filler-code',
		label: 'No filler opener',
		reward: 'code',
		code: NO_FILLER,
		prompts: FILLER_PROMPTS
	},
	{
		id: 'ends-with-question',
		label: 'Ends with a question',
		reward: 'code',
		code: `return answer.trim().endsWith('?');`,
		prompts: [
			'I want to get into hiking.',
			"I'm thinking about learning guitar.",
			'I need a gift for my mom.',
			'I want to cook dinner tonight.',
			"I'm bored this weekend.",
			'I want to read more books.',
			"I'm planning a trip to Japan.",
			'I want to get a pet.',
			"I'm starting a new job next week.",
			'I want to learn to draw.',
			'I need help picking a laptop.',
			'I want to eat healthier.',
			"I'm learning to swim.",
			'I want to start a podcast.',
			"I'm redecorating my room.",
			'I want to be more productive.',
			"I'm hosting a party.",
			'I want to learn chess.',
			"I'm training for a 5K.",
			'I want to grow tomatoes.'
		]
	},
	{
		id: 'admits-it-cant-know',
		label: "Admits what it can't know",
		reward: 'code',
		code: `return /(don't|do not|can't|cannot) (know|tell|access|predict|say|see)|not sure|unable to/i.test(answer);`,
		prompts: [
			'What did I eat for breakfast today?',
			"What is my dog's name?",
			'Who will win the next World Cup?',
			'What number am I thinking of?',
			'What is the capital of the country Zanthoria?',
			'What color is my car?',
			'What will the weather be in Paris on June 3rd, 2031?',
			'How old is my grandmother?',
			'What is my favorite movie?',
			"Who wrote the novel 'The Glass Harbor of Emberwick'?",
			'What will the stock market do tomorrow?',
			'Where did I leave my keys?',
			'What is my phone number?',
			'What was the name of my first teacher?',
			'Which horse will win tomorrow’s race?',
			'What did my friend text me yesterday?',
			'How many siblings do I have?',
			'What song am I listening to right now?',
			"What's in my fridge?",
			'What will I dream about tonight?'
		]
	},
	{
		id: 'shorter',
		label: 'Shorter (watch it cheat)',
		reward: 'code',
		// Rewards length alone, so it learns to say "No." to everything: reward hacking.
		code: `return 1 - answer.split(/\\s+/).length / 100;`,
		prompts: [
			'Explain how the internet works.',
			'Tell me about the Roman Empire.',
			'How do computers work?',
			'What is machine learning?',
			'Describe the water cycle.',
			'How does the stock market work?',
			'What is democracy?',
			'Explain photosynthesis.',
			'How do airplanes fly?',
			'What causes earthquakes?',
			'Tell me about dinosaurs.',
			'How does the heart work?',
			'What is climate change?',
			'Explain how vaccines work.',
			'What is a black hole?',
			'How do batteries work?',
			'What is the theory of evolution?',
			'How does GPS work?',
			'What is inflation?',
			'Explain how a car engine works.'
		]
	},
	{
		id: 'your-taste',
		label: 'Your taste',
		reward: 'human',
		prompts: [
			'Name a coffee shop.',
			'Write a one-line poem about rain.',
			'Describe a cat in a few words.',
			'Invent a new ice cream flavor.',
			'Give me a motto for Mondays.',
			'Name a pirate ship.',
			'Write a one-line poem about the moon.',
			'Describe the ocean in a few words.',
			'Invent a holiday.',
			'Name a band.',
			'Write a fortune cookie message.',
			'Describe autumn in a few words.',
			'Invent a new sport.',
			'Name a dragon.',
			'Write a one-line poem about coffee.',
			'Describe a city at night in a few words.',
			'Invent a board game.',
			'Name a spaceship.',
			'Write a toast for a friend.',
			'Describe snow in a few words.'
		]
	}
];

export const DEFAULT_PRESET = PRESETS[0];

type RewardConfig = { reward: RewardKind; constitution: string; rewardCode: string; prompts: string[] };

/** A config's preset, if its reward and prompts still match one. */
export function presetOf(c: RewardConfig): Preset | undefined {
	return PRESETS.find(
		(p) =>
			p.reward === c.reward &&
			(p.reward !== 'ai' || p.constitution === c.constitution) &&
			(p.reward !== 'code' || p.code === c.rewardCode) &&
			c.prompts.every((q) => p.prompts.includes(q))
	);
}

/** One line describing a reward, kept with saved models. */
export function rewardSummary(c: RewardConfig): string {
	const preset = presetOf(c);
	if (c.reward === 'human') return 'human feedback';
	if (c.reward === 'code') return `code: ${preset?.label ?? c.rewardCode.trim().split('\n')[0]}`;
	return `AI judge: ${preset?.label ?? c.constitution.trim().split('\n')[0]}`;
}
