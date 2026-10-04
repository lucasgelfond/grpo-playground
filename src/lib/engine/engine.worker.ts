/**
 * Everything that touches the GPU, in a worker so the page stays smooth:
 * loading weights, sampling, judging, updating and chatting. The page's
 * runtime (state/runtime.svelte.ts) drives it through engine.ts and mirrors
 * what it reports.
 */
import { defaultDevice, init, numpy as np, tree } from '@jax-js/jax';

import { loraTensors, parseLora, parseWeights, weightTensors, writeSafetensors } from '$lib/export';
import { compileModel, type CompiledModel, type Lora } from '$lib/models/llama';
import { getModel, promptFor, type ChatTurn, type ModelDef } from '$lib/models/registry';
import { loadTokenizer, type Tokenizer } from '$lib/models/tokenizer';
import { downloadWeights, isCached, loadWeights, type LoadProgress, type Weights } from '$lib/models/weights';
import { generate } from '$lib/rl/generate';
import { compareAnswers, judgePrefix, schedulePairs, winRates, type JudgePrefix, type Verdict } from '$lib/rl/judge';
import { RULES } from '$lib/rl/rules';
import { PolicyTrainer } from '$lib/rl/trainer';
import { loadSavedWeights, saveModel, type SavedMeta } from '$lib/saved';
import type { Config } from '$lib/state/config.svelte';
import type { Answer, LoadItem, Pass } from '$lib/state/runtime.svelte';

type ChatSide = { weights: Weights; lora: Lora | null };

export type LoadInput = {
	policyId: string;
	judgeId: string;
	mode: 'lora' | 'full';
	loraRank: number;
	loraAlpha: number;
	learningRate: number;
	temperature: number;
};
export type LoadUpdate = { status?: string; loads?: LoadItem[] };
export type PassUpdate = { status?: string; pass?: Pass };
/** Where the session autosaves after each pass. */
export type SaveTarget = { name: string; id: string; createdAt: string; passes: number; constitution: string };

/** Posts at most ~30 pass snapshots a second, however often the pass changes. */
function throttled(post: () => void): () => void {
	let pending = false;
	return () => {
		if (pending) return;
		pending = true;
		setTimeout(() => {
			pending = false;
			post();
		}, 33);
	};
}

/** Judge prefixes kept prefilled (each ~10–40 MB of K/V). */
const PREFIX_CACHE = 8;

export class Engine {
	#gpu: boolean | null = null;
	#policyDef: ModelDef | null = null;
	#judgeDef: ModelDef | null = null;
	#policyTok: Tokenizer | null = null;
	#judgeTok: Tokenizer | null = null;
	#trainer: PolicyTrainer | null = null;
	#judgeModel: CompiledModel | null = null;
	#judgeWeights: Weights | null = null;
	/** Judge prompt prefixes (guidelines + question), prefilled once and reused every pass. */
	#prefixes = new Map<string, JudgePrefix>();
	/**
	 * The original model's answer to each prompt ("GT"): its greedy, most
	 * likely answer. The original never changes, so this is computed once per
	 * prompt and is a steady yardstick (random samples from a 135M model are
	 * often just wrong, e.g. "a tomato is a vegetable").
	 */
	#references = new Map<string, Answer>();
	#chatSaved: { id: string; side: ChatSide } | null = null;

	async initGpu(): Promise<boolean> {
		if (this.#gpu === null) {
			this.#gpu = (await init('webgpu')).includes('webgpu');
			if (this.#gpu) defaultDevice('webgpu');
		}
		return this.#gpu;
	}

	async isCached(id: string): Promise<boolean> {
		return isCached(getModel(id));
	}

	async download(id: string, progress: (p: LoadProgress) => void): Promise<void> {
		await downloadWeights(getModel(id), progress);
	}

	/** Download (or read from cache) both models and set up a fresh trainer. */
	async load(input: LoadInput, progress: (u: LoadUpdate) => void): Promise<void> {
		if (!(await this.initGpu())) throw new Error('WebGPU is not available in this browser.');
		const loads: LoadItem[] = [];
		const track = (label: string, total: number) => {
			const item = { label, loaded: 0, total, downloaded: 0, done: false };
			loads.push(item);
			return (p: Partial<LoadItem>) => {
				Object.assign(item, p);
				progress({ loads });
			};
		};
		// Keep the (3 GB) judge in memory if it hasn't changed. In LoRA mode the
		// trainee's weights are frozen too, so a new project (new prompts or judge
		// prompt) only needs a fresh adapter, not a reload.
		const keepJudge = this.#judgeDef?.id === input.judgeId && !!this.#judgeWeights;
		const kept = input.mode === 'lora' && this.#trainer?.settings.mode === 'lora' && this.#policyDef?.id === input.policyId;
		const keptWeights = kept ? this.#trainer!.weights : null;
		this.#disposeAll(keepJudge, kept);
		const policyDef = getModel(input.policyId);
		const judgeDef = getModel(input.judgeId);

		progress({ status: 'Loading tokenizers…', loads });
		[this.#policyTok, this.#judgeTok] = await Promise.all([loadTokenizer(policyDef), loadTokenizer(judgeDef)]);

		let weights = keptWeights;
		if (!weights) {
			progress({ status: `Loading ${policyDef.label}…` });
			const p = track(`${policyDef.label} (trainee, f32)`, policyDef.downloadBytes);
			weights = await loadWeights(policyDef, np.float32, p);
			p({ done: true });
		}
		let base: Weights | null = null;
		if (input.mode === 'full') {
			// Full fine-tuning keeps an untouched copy for comparisons (read from cache).
			const b = track(`${policyDef.label} (frozen copy)`, policyDef.downloadBytes);
			base = await loadWeights(policyDef, np.float32, b);
			b({ done: true });
		}

		if (!keepJudge) {
			progress({ status: `Loading ${judgeDef.label}…` });
			const j = track(`${judgeDef.label} (judge, f16)`, judgeDef.downloadBytes);
			this.#judgeWeights = await loadWeights(judgeDef, np.float16, j);
			j({ done: true });
			this.#judgeModel = compileModel(judgeDef.config);
		}

		this.#trainer = new PolicyTrainer(policyDef, weights, base, {
			mode: input.mode,
			learningRate: input.learningRate,
			loraRank: input.loraRank,
			loraAlpha: input.loraAlpha,
			temperature: input.temperature,
			microbatch: policyDef.config.hidden > 600 ? 4 : 8
		});
		this.#policyDef = policyDef;
		this.#judgeDef = judgeDef;
	}

	/**
	 * Precompute what the first passes will need for these prompts, while the
	 * user is still on another page: the judge's prefilled instructions and the
	 * original model's answer for each. Both are cached, so a pass reuses them.
	 */
	async warm(input: { prompts: string[]; constitution: string; maxNew: number }): Promise<void> {
		if (!this.#trainer || !this.#policyDef || !this.#policyTok || !this.#judgeModel) return;
		for (const prompt of input.prompts.slice(0, PREFIX_CACHE)) {
			await this.#prefixFor(input.constitution, prompt);
			await this.#referenceFor(prompt, this.#policyTok.encode(promptFor(this.#policyDef, [{ role: 'user', content: prompt }])), input.maxNew);
		}
	}

	async #prefixFor(constitution: string, question: string): Promise<JudgePrefix> {
		const key = `${constitution}\u0000${question}`;
		let prefix = this.#prefixes.get(key);
		if (!prefix) {
			prefix = await judgePrefix(this.#judgeModel!, this.#judgeWeights!, this.#judgeTok!, this.#judgeDef!.padToken, constitution, question);
			this.#prefixes.set(key, prefix);
			// Bounded: each prefix is ~10 MB of K/V for Qwen2.5 1.5B, ~40 MB for Qwen3 1.7B.
			if (this.#prefixes.size > PREFIX_CACHE) {
				const [oldest, old] = this.#prefixes.entries().next().value!;
				tree.dispose(old.kv);
				this.#prefixes.delete(oldest);
			}
		}
		return prefix;
	}

	async #referenceFor(prompt: string, promptIds: number[], maxNew: number): Promise<Answer> {
		const key = `${maxNew}\u0000${prompt}`;
		let ref = this.#references.get(key);
		if (!ref) {
			const trainer = this.#trainer!;
			const def = this.#policyDef!;
			const tok = this.#policyTok!;
			const original = trainer.original();
			const gen = await generate(trainer.model, original.weights, original.lora, promptIds, {
				rows: 1,
				maxNew,
				temperature: 0,
				stopTokens: def.stopTokens,
				padToken: def.padToken
			});
			const ids = gen.tokens[0];
			ref = { tokens: ids, text: tok.decodeText(ids), stopped: gen.stopped[0], pieces: tok.pieces(ids) };
			this.#references.set(key, ref);
		}
		// A fresh copy per pass: each pass writes its own verdict onto it.
		return { ...ref, verdict: undefined };
	}

	/** Sample a group of answers, judge them, apply one GRPO update, and autosave. */
	async runPass(
		input: { index: number; prompt: string; config: Config; save: SaveTarget },
		progress: (u: PassUpdate) => void
	): Promise<Pass> {
		const trainer = this.#trainer;
		const def = this.#policyDef;
		const tok = this.#policyTok;
		if (!trainer || !def || !tok || !this.#judgeModel || !this.#judgeWeights || !this.#judgeTok) {
			throw new Error('Models are not loaded.');
		}
		const { index, prompt, config } = input;
		const G = config.groupSize;
		const pass: Pass = {
			index,
			prompt,
			phase: 'prefill',
			answers: Array.from({ length: G }, () => ({ tokens: [], text: '', stopped: false, pieces: [] }))
		};
		const sync = throttled(() => progress({ pass }));
		const status = (s: string) => progress({ status: s });
		const promptIds = tok.encode(promptFor(def, [{ role: 'user', content: prompt }]));

		// Prefill: the prompt goes into the policy model while the judge prefills
		// its own prompt for this question (cached after the first time). Then
		// the answers sample in parallel, alongside the original model's answers.
		status(`Pass ${index + 1}: prefilling`);
		const judgeReady = this.#prefixFor(config.constitution, prompt);
		const referenceReady = this.#referenceFor(prompt, promptIds, config.maxNew);
		const { weights, lora } = trainer.policy();
		const genReady = generate(trainer.model, weights, lora, promptIds, {
			rows: G,
			maxNew: config.maxNew,
			temperature: config.temperature,
			stopTokens: def.stopTokens,
			padToken: def.padToken,
			onStep: (tokens, stopped) => {
				if (pass.phase === 'prefill') {
					pass.phase = 'sampling';
					status(`Pass ${index + 1}: sampling ${G} answers`);
				}
				tokens.forEach((ids, i) => {
					const a = pass.answers[i];
					if (a.tokens.length === ids.length) return;
					a.tokens = [...ids];
					a.text = tok.decodeText(ids);
					a.stopped = stopped[i];
				});
				sync();
			}
		});
		const [gen, prefix, reference] = await Promise.all([genReady, judgeReady, referenceReady]);
		pass.reference = reference;
		gen.tokens.forEach((ids, i) => {
			pass.answers[i].pieces = tok.pieces(ids);
			pass.answers[i].stopped = gen.stopped[i];
		});

		pass.phase = 'judging';
		sync();
		// Presets with a rule check reward judge and rule together (averaged).
		const rule = config.rule ? RULES[config.rule] : null;
		const reward = (i: number, judged: number): Verdict => ({
			score: rule ? (judged + rule(pass.answers[i].text, pass.answers[i].stopped)) / 2 : judged
		});
		// Every pair of answers, plus each answer against the original model's.
		const pairs: [number, number][] = [
			...schedulePairs(G, Math.min(config.matchesPerAnswer, G - 1)),
			...pass.answers.map((_, i) => [i, G] as [number, number])
		];
		pass.matches = [];
		pass.matchesTotal = pairs.length;
		status(`Pass ${index + 1}: judge choosing between ${pairs.length} pairs`);
		const matches = await compareAnswers(
			this.#judgeModel,
			this.#judgeWeights,
			this.#judgeTok,
			this.#judgeDef!.padToken,
			config.constitution,
			prompt,
			[...pass.answers.map((a) => a.text), reference.text],
			pairs,
			{
				bothOrders: config.bothOrders,
				assistantPrefix: this.#judgeDef!.assistantPrefix,
				prefix,
				onMatch: (m) => {
					pass.matches!.push(m);
					// Live standings: win rates over the matches played so far.
					const rates = winRates(G + 1, pass.matches!);
					pass.answers.forEach((a, i) => {
						if (pass.matches!.some((x) => x.a === i || x.b === i)) a.verdict = reward(i, rates[i]);
					});
					reference.verdict = { score: rates[G] };
					sync();
				}
			}
		);
		const rates = winRates(G + 1, matches);
		const vsRef = matches.filter((m) => m.b === G);
		pass.vsOriginal = vsRef.reduce((s, m) => s + m.p, 0) / Math.max(1, vsRef.length);
		reference.verdict = { score: rates[G] };
		const verdicts = rates.slice(0, G).map((r, i) => reward(i, r));
		verdicts.forEach((v, i) => (pass.answers[i].verdict = v));
		const scores = verdicts.map((v) => v.score);
		pass.meanScore = scores.reduce((s, x) => s + x, 0) / scores.length;
		pass.best = scores.indexOf(Math.max(...scores));

		pass.phase = 'updating';
		sync();
		status(`Pass ${index + 1}: updating weights`);
		const res = await trainer.update(promptIds, gen.tokens, scores, def.padToken, config.klBeta);
		res.advantages.forEach((adv, i) => {
			const a = pass.answers[i];
			a.advantage = adv;
			a.logpBefore = res.logpBefore[i];
			a.logpAfter = res.logpAfter[i];
		});
		pass.klBeta = config.klBeta;
		if (!res.skipped) pass.kl = res.kl.reduce((s, x) => s + x, 0) / res.kl.length;
		pass.loss = res.loss;
		pass.skipped = res.skipped;
		pass.layerNorms = await trainer.layerNorms();
		pass.phase = 'done';
		await this.save(input.save).catch((e) => console.warn('Autosave failed', e));
		return pass;
	}

	async #sessionBytes(): Promise<Uint8Array<ArrayBuffer>> {
		const t = this.#trainer!;
		return writeSafetensors(t.lora ? await loraTensors(t.lora, t.cfg) : await weightTensors(t.weights, t.cfg), {
			base_model: this.#policyDef!.repo
		});
	}

	/** The current adapter (or weights), as safetensors. */
	async sessionBytes(): Promise<Uint8Array<ArrayBuffer>> {
		if (!this.#trainer || !this.#policyDef) throw new Error('Nothing to save yet.');
		return this.#sessionBytes();
	}

	/** Save the current adapter (or weights) in the browser; a new id when `id` is empty. */
	async save(target: SaveTarget): Promise<SavedMeta> {
		if (!this.#trainer || !this.#policyDef) throw new Error('Nothing to save yet.');
		const { settings } = this.#trainer;
		return saveModel(
			{
				name: target.name,
				baseId: this.#policyDef.id,
				mode: settings.mode,
				loraRank: settings.loraRank,
				loraAlpha: settings.loraAlpha,
				passes: target.passes,
				constitution: target.constitution
			},
			await this.#sessionBytes(),
			target.id || undefined,
			target.createdAt || undefined
		);
	}

	/**
	 * Parameters for one side of the chat. `source` is 'session', a saved
	 * model id, or 'original' (the untouched base of the given model).
	 */
	async #chatSide(source: string, baseId: string, saved: SavedMeta[]) {
		if (source === 'session') {
			if (!this.#trainer || !this.#policyDef || !this.#policyTok) throw new Error('No training session loaded.');
			return { def: this.#policyDef, tok: this.#policyTok, side: this.#trainer.policy() };
		}
		const def = getModel(baseId);
		const tok = this.#policyTok!;
		const original = this.#trainer!.original();
		if (source === 'original') return { def, tok, side: original };

		if (this.#chatSaved?.id !== source) {
			if (this.#chatSaved) this.#disposeSide(this.#chatSaved.side);
			const meta = saved.find((m) => m.id === source);
			if (!meta) throw new Error('Saved model not found.');
			const bytes = await loadSavedWeights(source);
			const side =
				meta.mode === 'lora'
					? { weights: original.weights, lora: parseLora(bytes, def.config) }
					: { weights: parseWeights(bytes, def.config), lora: null };
			this.#chatSaved = { id: source, side };
		}
		return { def, tok, side: this.#chatSaved.side };
	}

	/**
	 * Make sure the chat can run `baseId`: reuse the trainer's model, or load a
	 * bare one. Returns true when that replaced the training session.
	 */
	async #ensureChatBase(baseId: string, loraRank: number, loraAlpha: number, progress: (u: ChatUpdate) => void) {
		if (this.#policyDef?.id === baseId && this.#trainer) return false;
		if (!(await this.initGpu())) throw new Error('WebGPU is not available in this browser.');
		const def = getModel(baseId);
		this.#disposeAll(true);
		this.#policyTok = await loadTokenizer(def);
		const item: LoadItem = { label: `${def.label} (for chat)`, loaded: 0, total: def.downloadBytes, downloaded: 0, done: false };
		const weights = await loadWeights(def, np.float32, (x) => progress({ loads: [Object.assign(item, x)] }));
		progress({ loads: [Object.assign(item, { done: true })] });
		// A trainer with no passes is the simplest holder for "original" weights.
		this.#trainer = new PolicyTrainer(def, weights, null, {
			mode: 'lora',
			learningRate: 0,
			loraRank,
			loraAlpha,
			temperature: 1,
			microbatch: 8
		});
		this.#policyDef = def;
		return true;
	}

	/**
	 * Answer the same conversation with two models at once. Sources are
	 * 'original', 'session', or a saved model id, all on the base `baseId`.
	 * Weights are resolved one side at a time, then both generate in parallel.
	 */
	async chatPair(
		input: {
			sources: [string, string];
			baseId: string;
			histories: [ChatTurn[], ChatTurn[]];
			saved: SavedMeta[];
			loraRank: number;
			loraAlpha: number;
		},
		progress: (u: ChatUpdate) => void
	): Promise<[string, string]> {
		if (await this.#ensureChatBase(input.baseId, input.loraRank, input.loraAlpha, progress)) progress({ replaced: true });
		const sides: { def: ModelDef; tok: Tokenizer; side: ChatSide }[] = [];
		for (const source of input.sources) sides.push(await this.#chatSide(source, input.baseId, input.saved));
		const run = async (k: 0 | 1) => {
			const { def, tok, side } = sides[k];
			const ids = tok.encode(promptFor(def, input.histories[k]));
			const texts = ['', ''];
			const sync = throttled(() => progress({ side: k, text: texts[k] }));
			const gen = await generate(this.#trainer!.model, side.weights, side.lora, ids, {
				rows: 1,
				maxNew: 160,
				temperature: 0.7,
				stopTokens: def.stopTokens,
				padToken: def.padToken,
				onStep: (tokens) => {
					texts[k] = tok.decodeText(tokens[0]);
					sync();
				}
			});
			return tok.decodeText(gen.tokens[0]);
		};
		return Promise.all([run(0), run(1)]) as Promise<[string, string]>;
	}

	#disposeSide({ weights, lora }: ChatSide) {
		if (lora) tree.dispose(lora);
		else tree.dispose(weights);
	}

	#disposeAll(keepJudge = false, keepPolicyWeights = false) {
		// Reference answers come from the policy's original weights: they survive only if those do.
		if (!keepPolicyWeights) this.#references.clear();
		if (!keepJudge) {
			for (const p of this.#prefixes.values()) tree.dispose(p.kv);
			this.#prefixes.clear();
			if (this.#judgeWeights) tree.dispose(this.#judgeWeights);
			this.#judgeWeights = null;
			this.#judgeModel = null;
			this.#judgeDef = null;
		}
		this.#trainer?.dispose(keepPolicyWeights);
		this.#trainer = null;
		this.#policyDef = null;
		if (this.#chatSaved) this.#disposeSide(this.#chatSaved.side);
		this.#chatSaved = null;
	}
}

export type ChatUpdate = { side?: 0 | 1; text?: string; loads?: LoadItem[]; replaced?: boolean };

export type Request = { id: number; method: keyof Engine; input: unknown };
export type Response = { id: number; result?: unknown; error?: string; progress?: unknown };

const engine = new Engine();

self.onmessage = async ({ data: { id, method, input } }: MessageEvent<Request>) => {
	const post = (msg: Omit<Response, 'id'>, transfer: Transferable[] = []) => self.postMessage({ id, ...msg }, { transfer });
	try {
		const result = await (engine[method] as (input: unknown, progress: (p: unknown) => void) => Promise<unknown>)(input, (progress) =>
			post({ progress })
		);
		post({ result }, result instanceof Uint8Array ? [result.buffer] : []);
	} catch (e) {
		console.error(e);
		post({ error: e instanceof Error ? e.message : String(e) });
	}
};
