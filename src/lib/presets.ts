/**
 * Tasks pair a prompt set with a constitution and, where possible, a rule
 * check. Tasks train when the model already does the thing some of the time,
 * so a group of 8 answers has variety to learn from. Ordered by how well they
 * trained in 20 passes on SmolLM2 135M (LoRA, temperature 1.0, 128 tokens):
 *
 *   End with a question   22% -> 82% (rule), 57% -> 86% (rule + judge)
 *   Yes or no first       85% -> 100%, but collapses to "Yes." for everything
 *                         (since rebalanced to half yes / half no, and the
 *                         judge now checks the answer is right; not re-measured)
 *   No lists or markdown  11% -> 51%, prose gets less coherent
 *   One sentence, Admit what it can't know: rarely happen at all, so little signal
 */
import type { RuleId } from './rl/rules';

export type Task = { id: string; label: string; prompts: string[]; constitution: string; rule?: RuleId };

export const TASKS: Task[] = [
	{
		id: 'follow-up',
		rule: 'follow-up',
		label: 'End with a question',
		constitution: `Reward answers that give one or two helpful sentences and then end with a short, friendly follow-up question to the user.
The very last character must be "?".
The sentences before the question must be correct and sensible.
Penalize wrong or made-up information most of all, then answers that do not end with a question, and answers longer than three sentences.`,
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
		id: 'yes-no',
		rule: 'yes-no',
		label: 'Starts with yes or no',
		constitution: `Every question has an obvious answer: yes or no.
Reward answers whose very first word is the CORRECT answer, "Yes" or "No", followed by at most one short sentence of explanation.
Penalize wrong answers most of all, even if they start with Yes or No. Also penalize answers that do not start with Yes or No, and long answers.`,
		// Facts a 1.5B judge reliably gets right (no trick questions like "is a
		// tomato a fruit?"), alternating yes and no so "Yes." to everything loses.
		prompts: [
			'Answer yes or no: Is the sun hot?',
			'Answer yes or no: Can a cow fly?',
			'Answer yes or no: Is ice cold?',
			'Answer yes or no: Is the sky green?',
			'Answer yes or no: Do birds have feathers?',
			'Answer yes or no: Is fire cold?',
			'Answer yes or no: Do bees make honey?',
			'Answer yes or no: Is a rock alive?',
			'Answer yes or no: Is snow white?',
			'Answer yes or no: Do fish live in trees?',
			'Answer yes or no: Is the ocean salty?',
			'Answer yes or no: Is the moon made of cheese?',
			'Answer yes or no: Do plants need water?',
			'Answer yes or no: Can humans breathe underwater?',
			'Answer yes or no: Is the Earth round?',
			'Answer yes or no: Are bananas blue?',
			'Answer yes or no: Do cats have four legs?',
			'Answer yes or no: Is two plus two five?',
			'Answer yes or no: Is the sun a star?',
			'Answer yes or no: Can penguins fly?',
			'Answer yes or no: Is milk white?',
			'Answer yes or no: Can a fish ride a bike?',
			'Answer yes or no: Do dogs bark?',
			'Answer yes or no: Is grass purple?',
			'Answer yes or no: Is the night sky dark?',
			'Answer yes or no: Do trees have wheels?',
			'Answer yes or no: Is a week seven days long?',
			'Answer yes or no: Can a person live on the sun?',
			'Answer yes or no: Do cars need fuel or electricity to run?',
			'Answer yes or no: Is a mouse bigger than an elephant?'
		]
	},
	{
		id: 'plain-prose',
		rule: 'plain-prose',
		label: 'No lists or markdown',
		constitution: `Reward answers written as one short paragraph of plain prose that gives correct, sensible instructions.
Penalize wrong or unsafe advice most of all, then any formatting: bullet points, numbered lists, "Step 1" style steps, headings, or bold text with **.`,
		prompts: [
			'How do I make a paper airplane?',
			'How do I start running?',
			'How do I bake bread?',
			'How can I learn to code?',
			'How do I plant a tree?',
			'How do I change a tire?',
			'How do I make pancakes?',
			'How can I save money?',
			'How do I study for an exam?',
			'How do I train a puppy?',
			'How do I write a resume?',
			'How do I start a garden?',
			'How can I sleep better?',
			'How do I brew coffee?',
			'How do I learn a new language?',
			'How do I set up a fish tank?',
			'How do I make a budget?',
			'How do I prepare for a job interview?',
			'How do I fix a flat bike tire?',
			'How do I clean my room quickly?'
		]
	},
	{
		id: 'brevity',
		rule: 'brevity',
		label: 'One sentence',
		constitution: `Reward answers that are ONE short sentence (under 25 words) that directly answers the question.
The sentence must be correct.
Penalize wrong facts most of all, then answers longer than one sentence, lists, headings, and answers that trail off unfinished.`,
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
		id: 'dont-know',
		rule: 'dont-know',
		label: 'Admit what it can’t know',
		constitution: `These questions ask about things an AI cannot know: the user's private life, the future, or made-up things.
Reward answers that clearly say "I don't know" or explain that it can't know, in one or two sentences.
Penalize answers that invent a specific answer most of all, since any specific answer is a made-up one.`,
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
		id: 'general',
		label: 'Concise & kind',
		constitution: `Reward answers that:
1. Answer the question directly and correctly in the first sentence.
2. Are short: 1-3 sentences. Shorter wins if nothing important is lost.
3. Are warm and friendly, but never gushing or flattering.
4. Say "I'm not sure" rather than making things up.
5. Use plain words a 10-year-old would understand.

Penalize wrong facts most of all, then: rambling, repeating itself, trailing off mid-sentence, invented facts, and ignoring the question.`,
		prompts: [
			'Why is the sky blue?',
			'How do I boil an egg?',
			'What is a black hole?',
			'Why do cats purr?',
			'How does a rainbow form?',
			'What should I name my goldfish?',
			'Why do we dream?',
			'How do airplanes stay in the air?',
			'What is the tallest mountain in the world?',
			'How can I fall asleep faster?',
			'Why is the ocean salty?',
			'What is photosynthesis?',
			'How do I make a paper airplane?',
			'Why do leaves change color in the fall?',
			'What is the difference between weather and climate?',
			'How many legs does a spider have?',
			'Why do onions make you cry?',
			'What is gravity?',
			'How do bees make honey?',
			'What is a good way to start learning to cook?'
		]
	}
];

export const DEFAULT_TASK = TASKS.find((t) => t.id === 'yes-no')!;
export const DEFAULT_PROMPTS = DEFAULT_TASK.prompts;
export const DEFAULT_CONSTITUTION = DEFAULT_TASK.constitution;

/** The gist of a judge prompt in one line: its preset's name, or its first line. */
export function judgeSummary(constitution: string): string {
	return TASKS.find((t) => t.constitution === constitution.trim())?.label ?? constitution.trim().split('\n')[0];
}
