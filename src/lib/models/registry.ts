/**
 * The models this app knows how to load. All are Llama-style decoders
 * (RMSNorm, RoPE, SwiGLU MLP, grouped-query attention, tied embeddings) that
 * differ in shapes, rope base, whether q/k/v have biases (Qwen2), and whether
 * q and k are RMS-normalized per head (Qwen3). LFM2 also swaps most attention
 * layers for gated short convolutions.
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
	/** RMSNorm on each head of q and k before RoPE (Qwen3, LFM2's attention layers). */
	qkNorm?: boolean;
	/** Hugging Face tensor naming: Llama-style, or LFM2's. */
	arch?: 'llama' | 'lfm2';
	/** LFM2: which layers are attention and which are short convolutions. */
	layerTypes?: ('attention' | 'conv')[];
	/** LFM2: width of the short convolution. */
	convKernel?: number;
};

export type TokenizerKind = 'smollm2' | 'qwen2' | 'lfm2';

export type ModelDef = {
	id: string;
	label: string;
	repo: string;
	params: string;
	/** Size of the bf16 safetensors download, in bytes. */
	downloadBytes: number;
	config: ModelConfig;
	tokenizer: TokenizerKind;
	/** '' means no system turn at all (LFM2 has no default system prompt). */
	defaultSystemPrompt: string;
	/** Text before the first turn, e.g. LFM2's <|startoftext|>. */
	bos?: string;
	/** Opens every assistant turn, e.g. Qwen3's empty think block. */
	assistantPrefix?: string;
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
		id: 'lfm2-350m',
		label: 'LFM2 350M',
		repo: 'LiquidAI/LFM2-350M',
		params: '354M',
		downloadBytes: 708_984_464,
		config: {
			hidden: 1024,
			layers: 16,
			heads: 16,
			kvHeads: 8,
			headDim: 64,
			intermediate: 4608,
			vocab: 65536,
			ropeTheta: 1_000_000,
			rmsEps: 1e-5,
			qkvBias: false,
			qkNorm: true,
			arch: 'lfm2',
			// Attention at layers 2, 5, 8, 10, 12 and 14; short convolutions elsewhere.
			layerTypes: Array.from({ length: 16 }, (_, i) => ([2, 5, 8, 10, 12, 14].includes(i) ? 'attention' : 'conv')),
			convKernel: 3
		},
		tokenizer: 'lfm2',
		defaultSystemPrompt: '',
		bos: '<|startoftext|>',
		stopTokens: [7, 2],
		padToken: 0,
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
	},
	{
		id: 'qwen3-1.7b',
		label: 'Qwen3 1.7B',
		repo: 'Qwen/Qwen3-1.7B',
		params: '1.7B',
		downloadBytes: 4_063_515_528,
		config: {
			hidden: 2048,
			layers: 28,
			heads: 16,
			kvHeads: 8,
			headDim: 128,
			intermediate: 6144,
			vocab: 151936,
			ropeTheta: 1_000_000,
			rmsEps: 1e-6,
			qkvBias: false,
			qkNorm: true
		},
		tokenizer: 'qwen2',
		defaultSystemPrompt: 'You are a helpful assistant.',
		// Qwen3 thinks before answering unless its turn opens with an empty think block.
		assistantPrefix: '<think>\n\n</think>\n\n',
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
// VITE_WEIGHTS_BASE points a dev build at a local mirror (same layout) for testing.
export const WEIGHTS_BASE: string = import.meta.env.VITE_WEIGHTS_BASE ?? 'https://models.lucasgelfond.online';

export function modelUrl(def: ModelDef, file: string): string {
	return `${WEIGHTS_BASE}/${def.repo}/${file}`;
}

export type ChatTurn = { role: 'system' | 'user' | 'assistant'; content: string };

/** ChatML, shared by SmolLM2, Qwen and LFM2. Ends with an open assistant turn. */
export function chatPrompt(turns: ChatTurn[], assistantPrefix = '', bos = ''): string {
	let text = bos;
	for (const t of turns) {
		if (t.role === 'system' && !t.content) continue;
		text += `<|im_start|>${t.role}\n${t.content}<|im_end|>\n`;
	}
	return text + '<|im_start|>assistant\n' + assistantPrefix;
}

/** A model's own chat prompt: its system prompt (if any), the turns, and its BOS. */
export function promptFor(def: ModelDef, turns: ChatTurn[]): string {
	return chatPrompt([{ role: 'system', content: def.defaultSystemPrompt }, ...turns], def.assistantPrefix, def.bos);
}

export function formatBytes(n: number): string {
	if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
	return `${Math.round(n / 1e6)} MB`;
}
