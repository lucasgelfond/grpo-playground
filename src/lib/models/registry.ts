/**
 * The models this app knows how to load. All four are Llama-style decoders
 * (RMSNorm, RoPE, SwiGLU MLP, grouped-query attention, tied embeddings) that
 * differ only in shapes, rope base, and whether q/k/v have biases (Qwen2).
 */

export type ModelConfig = {
	hidden: number;
	layers: number;
	heads: number;
	kvHeads: number;
	headDim: number;
	intermediate: number;
	vocab: number;
	ropeTheta: number;
	rmsEps: number;
	qkvBias: boolean;
};

export type TokenizerKind = 'smollm2' | 'qwen2';

export type ModelDef = {
	id: string;
	label: string;
	repo: string;
	params: string;
	/** Size of the bf16 safetensors download, in bytes. */
	downloadBytes: number;
	config: ModelConfig;
	tokenizer: TokenizerKind;
	defaultSystemPrompt: string;
	/** Token ids that end an assistant turn. */
	stopTokens: number[];
	/** Token id used to right-pad finished sequences. */
	padToken: number;
	/** Can this model be the one we train? */
	trainable: boolean;
	/** Is full fine-tuning (all weights + Adam state) small enough to allow? */
	fullFinetune: 'ok' | 'heavy' | 'no';
	/** Can this model act as the judge? */
	judge: boolean;
};

export const MODELS: ModelDef[] = [
	{
		id: 'smollm2-135m',
		label: 'SmolLM2 135M Instruct',
		repo: 'HuggingFaceTB/SmolLM2-135M-Instruct',
		params: '135M',
		downloadBytes: 269_060_552,
		config: {
			hidden: 576,
			layers: 30,
			heads: 9,
			kvHeads: 3,
			headDim: 64,
			intermediate: 1536,
			vocab: 49152,
			ropeTheta: 100_000,
			rmsEps: 1e-5,
			qkvBias: false
		},
		tokenizer: 'smollm2',
		defaultSystemPrompt: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face',
		stopTokens: [2, 0],
		padToken: 2,
		trainable: true,
		fullFinetune: 'ok',
		judge: false
	},
	{
		id: 'smollm2-360m',
		label: 'SmolLM2 360M Instruct',
		repo: 'HuggingFaceTB/SmolLM2-360M-Instruct',
		params: '360M',
		downloadBytes: 723_674_912,
		config: {
			hidden: 960,
			layers: 32,
			heads: 15,
			kvHeads: 5,
			headDim: 64,
			intermediate: 2560,
			vocab: 49152,
			ropeTheta: 100_000,
			rmsEps: 1e-5,
			qkvBias: false
		},
		tokenizer: 'smollm2',
		defaultSystemPrompt: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face',
		stopTokens: [2, 0],
		padToken: 2,
		trainable: true,
		fullFinetune: 'heavy',
		judge: false
	},
	{
		id: 'qwen2.5-0.5b',
		label: 'Qwen2.5 0.5B Instruct',
		repo: 'Qwen/Qwen2.5-0.5B-Instruct',
		params: '494M',
		downloadBytes: 988_097_824,
		config: {
			hidden: 896,
			layers: 24,
			heads: 14,
			kvHeads: 2,
			headDim: 64,
			intermediate: 4864,
			vocab: 151936,
			ropeTheta: 1_000_000,
			rmsEps: 1e-6,
			qkvBias: true
		},
		tokenizer: 'qwen2',
		defaultSystemPrompt: 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.',
		stopTokens: [151645, 151643],
		padToken: 151643,
		trainable: true,
		fullFinetune: 'no',
		judge: true
	},
	{
		id: 'qwen2.5-1.5b',
		label: 'Qwen2.5 1.5B Instruct',
		repo: 'Qwen/Qwen2.5-1.5B-Instruct',
		params: '1.5B',
		downloadBytes: 3_087_467_144,
		config: {
			hidden: 1536,
			layers: 28,
			heads: 12,
			kvHeads: 2,
			headDim: 128,
			intermediate: 8960,
			vocab: 151936,
			ropeTheta: 1_000_000,
			rmsEps: 1e-6,
			qkvBias: true
		},
		tokenizer: 'qwen2',
		defaultSystemPrompt: 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.',
		stopTokens: [151645, 151643],
		padToken: 151643,
		trainable: false,
		fullFinetune: 'no',
		judge: true
	}
];

export function getModel(id: string): ModelDef {
	const def = MODELS.find((m) => m.id === id);
	if (!def) throw new Error(`Unknown model ${id}`);
	return def;
}

/**
 * Model files are mirrored in R2 (bucket rl-playground-models), laid out like
 * Hugging Face repos: <repo>/tokenizer.json, and each model.safetensors split
 * into 256 MB parts (R2 uploads are size-limited) with a manifest.
 */
export const WEIGHTS_BASE = 'https://models.lucasgelfond.online';

export function modelUrl(def: ModelDef, file: string): string {
	return `${WEIGHTS_BASE}/${def.repo}/${file}`;
}

export type ChatTurn = { role: 'system' | 'user' | 'assistant'; content: string };

/** ChatML, shared by SmolLM2 and Qwen2.5. Ends with an open assistant turn. */
export function chatPrompt(turns: ChatTurn[]): string {
	let text = '';
	for (const t of turns) text += `<|im_start|>${t.role}\n${t.content}<|im_end|>\n`;
	return text + '<|im_start|>assistant\n';
}

export function formatBytes(n: number): string {
	if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
	return `${Math.round(n / 1e6)} MB`;
}
