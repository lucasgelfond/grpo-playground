import { defaultDevice, init, numpy as np, tree } from '@jax-js/jax';

import {
	downloadBlob,
	loraTensors,
	parseLora,
	parseWeights,
	peftConfig,
	weightTensors,
	writeSafetensors,
	zip
} from '$lib/export';
import { compileModel, type CompiledModel, type Lora } from '$lib/models/llama';
import { chatPrompt, getModel, type ChatTurn, type ModelDef } from '$lib/models/registry';
import { loadTokenizer, type Tokenizer } from '$lib/models/tokenizer';
import { loadWeights, persistStorage, type Weights } from '$lib/models/weights';
import { generate } from '$lib/rl/generate';
import { compareAnswers, judgeAnswers, judgePrefix, schedulePairs, winRates, type JudgePrefix, type Match, type Verdict } from '$lib/rl/judge';
import { RULES } from '$lib/rl/rules';
import { PolicyTrainer } from '$lib/rl/trainer';
import { createProject, deleteProjectOfModel, listPolicyModels, recordPass, renameModel } from '$lib/db';
import { deleteSaved, loadSavedWeights, newModelId, saveModel, type SavedMeta } from '$lib/saved';
import { petname } from '$lib/petname';
import { config, learningRate, type Config } from './config.svelte';

export type Phase = 'prefill' | 'sampling' | 'judging' | 'updating' | 'done';

export type Answer = {
	tokens: number[];
	text: string;
	stopped: boolean;
	/** Display pieces, one per token, filled in once sampling finishes. */
	pieces: string[];
	verdict?: Verdict;
	advantage?: number;
	logpBefore?: number[];
	logpAfter?: number[];
	logpRef?: number[];
	/** Mean per-token KL from the original model, before this pass's update. */
	kl?: number;
};

export type Pass = {
	index: number;
	prompt: string;
	phase: Phase;
	answers: Answer[];
	best?: number;
	meanScore?: number;
	loss?: number;
	skipped?: boolean;
	/** Mean per-token KL from the original model across the group, and the β used. */
	kl?: number;
	klBeta?: number;
	/**
	 * Compare mode: one answer from the untouched original model, entered into
	 * the matches (as index answers.length) but never trained on. How often the
	 * trained model's answers beat it is the pass's absolute progress measure.
	 */
	reference?: Answer;
	/** Mean P(a trained answer beats the original model's answer). */
	vsOriginal?: number;
	/** Compare mode: every head-to-head choice the judge made, and how many were scheduled. */
	matches?: Match[];
	matchesTotal?: number;
	/** [layer][target] ||ΔW|| from the original weights, after this pass. */
	layerNorms?: number[][];
	ms: { sample?: number; judge?: number; update?: number };
};

export type LoadItem = { label: string; loaded: number; total: number; downloaded: number; done: boolean };

type ChatSide = { weights: Weights; lora: Lora | null };

/**
 * What makes a project: its models, how they're trained, and what they're
 * trained on. Changing any of these starts a new project (fresh model, new
 * name, new database row) instead of silently changing the task mid-run.
 */
function trainingKey(c: Config): string {
	return JSON.stringify([
		c.policyId,
		c.judgeId,
		c.mode,
		c.loraRank,
		c.loraAlpha,
		learningRate(c),
		c.temperature,
		c.prompts.map((p) => p.trim()).filter(Boolean),
		c.constitution.trim(),
		c.rule
	]);
}

class Runtime {
	gpu = $state<'unknown' | 'ok' | 'missing'>('unknown');
	loads = $state<LoadItem[]>([]);
	status = $state('');
	error = $state<string | null>(null);
	loading = $state(false);
	ready = $state(false);

	passes = $state<Pass[]>([]);
	running = $state(false);
	playing = $state(false);
	/** Which pass the training view shows; null follows the latest. */
	viewing = $state<number | null>(null);

	saved = $state<SavedMeta[]>([]);
	/** The model being trained: its name, and the OPFS id it autosaves under after every pass. */
	sessionName = $state('');
	sessionId = '';
	sessionCreatedAt = '';
	/** This session's rows in the database (null if the database is unavailable). */
	#projectId: number | null = null;
	#policyRowId: number | null = null;

	// GPU-backed objects must not be deep-proxied by Svelte, so they live in
	// plain fields; `version` bumps whenever they change.
	version = $state(0);
	policyDef: ModelDef | null = null;
	judgeDef: ModelDef | null = null;
	policyTok: Tokenizer | null = null;
	judgeTok: Tokenizer | null = null;
	trainer: PolicyTrainer | null = null;
	judgeModel: CompiledModel | null = null;
	judgeWeights: Weights | null = null;
	#loadedKey = '';
	#promptOrder: number[] = [];
	#promptCursor = 0;
	#chatModel: CompiledModel | null = null;
	/** Judge prompt prefixes (guidelines + question), prefilled once and reused every pass. */
	#prefixes = new Map<string, JudgePrefix>();
	/**
	 * The original model's answer to each prompt ("GT"): its greedy, most
	 * likely answer. The original never changes, so this is computed once per
	 * prompt and is a steady yardstick (random samples from a 135M model are
	 * often just wrong, e.g. "a tomato is a vegetable").
	 */
	#references = new Map<string, { answers: Answer[]; next: number }>();
	#chatSaved: { id: string; side: ChatSide } | null = null;

	get currentPass(): Pass | undefined {
		return this.viewing === null ? this.passes.at(-1) : this.passes[this.viewing];
	}

	async initGpu(): Promise<boolean> {
		if (this.gpu !== 'unknown') return this.gpu === 'ok';
		const devices = await init('webgpu');
		this.gpu = devices.includes('webgpu') ? 'ok' : 'missing';
		if (this.gpu === 'ok') defaultDevice('webgpu');
		return this.gpu === 'ok';
	}

	get needsReload(): boolean {
		return this.ready && this.#loadedKey !== trainingKey(config);
	}

	#track(label: string, total: number): LoadItem {
		this.loads.push({ label, loaded: 0, total, downloaded: 0, done: false });
		return this.loads[this.loads.length - 1];
	}

	/** Download (or read from cache) both models and set up a fresh trainer. */
	async load(): Promise<void> {
		if (this.loading) return;
		const key = trainingKey(config);
		if (this.ready && key === this.#loadedKey) return;
		this.loading = true;
		this.error = null;
		try {
			if (!(await this.initGpu())) throw new Error('WebGPU is not available in this browser.');
			await persistStorage();
			// Keep the (3 GB) judge in memory if it hasn't changed; everything else restarts.
			const keepJudge = this.judgeDef?.id === config.judgeId && !!this.judgeWeights;
			this.#disposeAll(keepJudge);
			this.loads = [];
			const policyDef = getModel(config.policyId);
			const judgeDef = getModel(config.judgeId);

			this.status = 'Loading tokenizers…';
			[this.policyTok, this.judgeTok] = await Promise.all([loadTokenizer(policyDef), loadTokenizer(judgeDef)]);

			this.status = `Loading ${policyDef.label}…`;
			const p = this.#track(`${policyDef.label} (trainee, f32)`, policyDef.downloadBytes);
			const weights = await loadWeights(policyDef, np.float32, (x) => Object.assign(p, x));
			p.done = true;
			let base: Weights | null = null;
			if (config.mode === 'full') {
				// Full fine-tuning keeps an untouched copy for comparisons (read from cache).
				const b = this.#track(`${policyDef.label} (frozen copy)`, policyDef.downloadBytes);
				base = await loadWeights(policyDef, np.float32, (x) => Object.assign(b, x));
				b.done = true;
			}

			if (!keepJudge) {
				this.status = `Loading ${judgeDef.label}…`;
				const j = this.#track(`${judgeDef.label} (judge, f16)`, judgeDef.downloadBytes);
				this.judgeWeights = await loadWeights(judgeDef, np.float16, (x) => Object.assign(j, x));
				j.done = true;
				this.judgeModel = compileModel(judgeDef.config);
			}

			this.trainer = new PolicyTrainer(policyDef, weights, base, {
				mode: config.mode,
				learningRate: learningRate(config),
				loraRank: config.loraRank,
				loraAlpha: config.loraAlpha,
				temperature: config.temperature,
				microbatch: policyDef.config.hidden > 600 ? 4 : 8
			});
			this.policyDef = policyDef;
			this.judgeDef = judgeDef;
			this.#chatModel = this.trainer.model;
			this.passes = [];
			this.viewing = null;
			this.#promptOrder = [];
			this.#loadedKey = key;
			this.sessionName = petname();
			this.sessionId = newModelId(this.sessionName);
			this.sessionCreatedAt = new Date().toISOString();
			await this.#recordProject(policyDef.id, judgeDef.id);
			this.ready = true;
			this.status = '';
			this.version++;
		} catch (e) {
			console.error(e);
			this.error = e instanceof Error ? e.message : String(e);
			this.status = 'Failed to load';
		} finally {
			this.loading = false;
		}
	}

	#nextPrompt(): string {
		const prompts = config.prompts.filter((p) => p.trim());
		if (prompts.length === 0) throw new Error('Add at least one prompt first.');
		if (this.#promptCursor >= this.#promptOrder.length || this.#promptOrder.some((i) => i >= prompts.length)) {
			this.#promptOrder = prompts.map((_, i) => i);
			if (config.promptOrder === 'shuffle') {
				for (let i = this.#promptOrder.length - 1; i > 0; i--) {
					const j = Math.floor(Math.random() * (i + 1));
					[this.#promptOrder[i], this.#promptOrder[j]] = [this.#promptOrder[j], this.#promptOrder[i]];
				}
			}
			this.#promptCursor = 0;
		}
		return prompts[this.#promptOrder[this.#promptCursor++]];
	}

	async #recordProject(policyId: string, judgeId: string) {
		this.#projectId = this.#policyRowId = null;
		try {
			const { projectId, policyId: rowId } = await createProject({
				name: this.sessionName,
				mode: config.mode,
				settings: {
					groupSize: config.groupSize,
					matchesPerAnswer: config.matchesPerAnswer,
					bothOrders: config.bothOrders,
					maxNew: config.maxNew,
					temperature: config.temperature,
					learningRate: learningRate(config),
					loraRank: config.loraRank,
					loraAlpha: config.loraAlpha,
					klBeta: config.klBeta,
					rule: config.rule
				},
				initialModelId: policyId,
				judgeModelId: judgeId,
				policy: { petname: this.sessionName, weightsPath: this.sessionId },
				judgePrompt: config.constitution,
				prompts: config.prompts.filter((p) => p.trim())
			});
			this.#projectId = projectId;
			this.#policyRowId = rowId;
		} catch (e) {
			console.warn('Could not record the project in the database', e);
		}
	}

	async #prefixFor(question: string): Promise<JudgePrefix> {
		const key = `${config.constitution}\u0000${question}`;
		let prefix = this.#prefixes.get(key);
		if (!prefix) {
			prefix = await judgePrefix(this.judgeModel!, this.judgeWeights!, this.judgeTok!, this.judgeDef!.padToken, config.constitution, question);
			this.#prefixes.set(key, prefix);
			// Bounded: each prefix is ~10 MB of K/V for the 1.5B judge.
			if (this.#prefixes.size > 24) {
				const [oldest, old] = this.#prefixes.entries().next().value!;
				tree.dispose(old.kv);
				this.#prefixes.delete(oldest);
			}
		}
		return prefix;
	}

	async #referenceFor(prompt: string, promptIds: number[]): Promise<Answer> {
		const key = `${config.maxNew}\u0000${prompt}`;
		let pool = this.#references.get(key);
		if (!pool) {
			const trainer = this.trainer!;
			const def = this.policyDef!;
			const tok = this.policyTok!;
			const original = trainer.original();
			const gen = await generate(trainer.model, original.weights, original.lora, promptIds, {
				rows: 1,
				maxNew: config.maxNew,
				temperature: 0,
				stopTokens: def.stopTokens,
				padToken: def.padToken
			});
			pool = {
				answers: gen.tokens.map((ids, i) => ({
					tokens: ids,
					text: tok.decodeText(ids),
					stopped: gen.stopped[i],
					pieces: tok.pieces(ids)
				})),
				next: 0
			};
			this.#references.set(key, pool);
		}
		const ref = pool.answers[pool.next++ % pool.answers.length];
		// A fresh copy per pass: each pass writes its own verdict onto it.
		return { ...ref, verdict: undefined };
	}

	#promptIds(prompt: string): number[] {
		return this.policyTok!.encode(
			chatPrompt([
				{ role: 'system', content: this.policyDef!.defaultSystemPrompt },
				{ role: 'user', content: prompt }
			])
		);
	}

	/** Sample a group of answers, judge them, and apply one GRPO update. */
	async runPass(): Promise<void> {
		if (this.running) return;
		this.running = true;
		this.error = null;
		try {
			await this.load();
			const trainer = this.trainer;
			const def = this.policyDef;
			if (!trainer || !def || !this.policyTok || !this.judgeModel || !this.judgeWeights || !this.judgeTok) return;
			const tok = this.policyTok;
			const G = config.groupSize;

			const prompt = this.#nextPrompt();
			this.passes.push({
				index: this.passes.length,
				prompt,
				phase: 'prefill',
				answers: Array.from({ length: G }, () => ({ tokens: [], text: '', stopped: false, pieces: [] })),
				ms: {}
			});
			const pass = this.passes[this.passes.length - 1];
			const promptIds = tok.encode(
				chatPrompt([
					{ role: 'system', content: def.defaultSystemPrompt },
					{ role: 'user', content: prompt }
				])
			);

			// Prefill: the prompt goes into the policy model while the judge prefills
			// its own prompt for this question (cached after the first time). Then
			// the answers sample in parallel, alongside the original model's answers.
			this.status = `Pass ${pass.index + 1}: prefilling`;
			let t = performance.now();
			const compareMode = config.judgeMode === 'compare';
			const judgeReady = compareMode ? this.#prefixFor(prompt) : Promise.resolve(null);
			const referenceReady = compareMode ? this.#referenceFor(prompt, promptIds) : Promise.resolve(undefined);
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
						this.status = `Pass ${pass.index + 1}: sampling ${G} answers`;
					}
					tokens.forEach((ids, i) => {
						const a = pass.answers[i];
						if (a.tokens.length === ids.length) return;
						a.tokens = [...ids];
						a.text = tok.decodeText(ids);
						a.stopped = stopped[i];
					});
				}
			});
			const [gen, prefix, reference] = await Promise.all([genReady, judgeReady, referenceReady]);
			if (reference) pass.reference = reference;
			gen.tokens.forEach((ids, i) => {
				pass.answers[i].pieces = tok.pieces(ids);
				pass.answers[i].stopped = gen.stopped[i];
			});
			pass.ms.sample = performance.now() - t;

			pass.phase = 'judging';
			t = performance.now();
			// Presets with a rule check reward judge and rule together (averaged).
			const rule = config.rule ? RULES[config.rule] : null;
			const useJudge = true;
			const ruleScores = pass.answers.map((a) => (rule ? rule.score(a.text, a.stopped) : 0));
			let verdicts: Verdict[];
			const withRule = (i: number, judged: number): Verdict =>
				rule
					? { score: (judged + ruleScores[i]) / 2, mass: 1, judge: judged, rule: ruleScores[i] }
					: { score: judged, mass: 1, judge: judged };
			if (useJudge && config.judgeMode === 'compare') {
				const pairs: [number, number][] = [
					...schedulePairs(G, Math.min(config.matchesPerAnswer, G - 1)),
					...pass.answers.map((_, i) => [i, G] as [number, number])
				];
				pass.matches = [];
				pass.matchesTotal = pairs.length;
				this.status = `Pass ${pass.index + 1}: judge choosing between ${pairs.length} pairs`;
				const texts = [...pass.answers.map((a) => a.text), pass.reference!.text];
				const matches = await compareAnswers(
					this.judgeModel,
					this.judgeWeights,
					this.judgeTok,
					this.judgeDef!.padToken,
					config.constitution,
					prompt,
					texts,
					pairs,
					{
						bothOrders: config.bothOrders,
						prefix: prefix ?? undefined,
						onMatch: (m) => {
						pass.matches!.push(m);
						// Live standings: win rates over the matches played so far.
						const rates = winRates(G + 1, pass.matches!);
						pass.answers.forEach((a, i) => {
							if (pass.matches!.some((x) => x.a === i || x.b === i)) a.verdict = withRule(i, rates[i]);
						});
						pass.reference!.verdict = { score: rates[G], mass: 1 };
						}
					}
				);
				const rates = winRates(G + 1, matches);
				const vsRef = matches.filter((m) => m.b === G);
				pass.vsOriginal = vsRef.reduce((s, m) => s + m.p, 0) / Math.max(1, vsRef.length);
				pass.reference!.verdict = { score: rates[G], mass: 1 };
				verdicts = rates.slice(0, G).map((r, i) => withRule(i, r));
				verdicts.forEach((v, i) => (pass.answers[i].verdict = v));
			} else if (useJudge) {
				this.status = `Pass ${pass.index + 1}: judging`;
				verdicts = await judgeAnswers(
					this.judgeModel,
					this.judgeWeights,
					this.judgeTok,
					this.judgeDef!.padToken,
					config.constitution,
					prompt,
					pass.answers.map((a) => a.text),
					(i, v) => {
						pass.answers[i].verdict = rule
							? { score: (v.score + ruleScores[i]) / 2, mass: v.mass, judge: v.score, rule: ruleScores[i] }
							: v;
					}
				);
				verdicts = pass.answers.map((a) => a.verdict!);
			} else {
				verdicts = ruleScores.map((r) => ({ score: r, mass: 1, rule: r }));
				verdicts.forEach((v, i) => (pass.answers[i].verdict = v));
			}
			pass.ms.judge = performance.now() - t;
			const scores = verdicts.map((v) => v.score);
			pass.meanScore = scores.reduce((s, x) => s + x, 0) / scores.length;
			pass.best = scores.indexOf(Math.max(...scores));

			pass.phase = 'updating';
			this.status = `Pass ${pass.index + 1}: updating weights`;
			t = performance.now();
			const res = await trainer.update(promptIds, gen.tokens, scores, def.padToken, config.klBeta);
			res.advantages.forEach((adv, i) => {
				const a = pass.answers[i];
				a.advantage = adv;
				a.logpBefore = res.logpBefore[i];
				a.logpAfter = res.logpAfter[i];
				a.logpRef = res.logpRef[i];
				a.kl = res.kl[i];
			});
			pass.klBeta = config.klBeta;
			if (!res.skipped) pass.kl = res.kl.reduce((s, x) => s + x, 0) / res.kl.length;
			pass.loss = res.loss;
			pass.skipped = res.skipped;
			pass.layerNorms = await trainer.layerNorms();
			pass.ms.update = performance.now() - t;
			pass.phase = 'done';
			await this.#autosave();
			if (this.#projectId !== null && this.#policyRowId !== null) {
				await recordPass(this.#projectId, this.#policyRowId, pass.index, prompt, $state.snapshot(pass)).catch((e) =>
					console.warn('Could not record the pass', e)
				);
			}
			await this.refreshSaved();
			this.status = res.skipped
				? `Pass ${pass.index + 1}: all answers scored the same, so there was nothing to learn`
				: `Pass ${pass.index + 1} done`;
			this.version++;
		} catch (e) {
			console.error(e);
			this.error = e instanceof Error ? e.message : String(e);
			this.playing = false;
		} finally {
			this.running = false;
		}
	}

	async play() {
		if (this.playing) return;
		this.playing = true;
		while (this.playing) {
			await this.runPass();
			if (this.error) break;
		}
		this.playing = false;
	}

	pause() {
		this.playing = false;
	}

	// --- saving, exporting and chatting -----------------------------------------------

	/** Trained models, from the database; their weights live in OPFS under `id`. */
	async refreshSaved() {
		try {
			const rows = await listPolicyModels();
			this.saved = rows.map((r) => ({
				id: r.weights_path,
				dbId: r.id,
				name: r.petname,
				createdAt: r.created_at,
				updatedAt: r.updated_at,
				baseId: r.base_model_id,
				mode: r.mode,
				loraRank: Number(r.settings.loraRank ?? 16),
				loraAlpha: Number(r.settings.loraAlpha ?? 32),
				passes: r.passes,
				constitution: r.judge_prompt ?? '',
				bytes: 0
			}));
		} catch (e) {
			console.warn('Could not list saved models', e);
		}
	}

	async #sessionBytes(): Promise<Uint8Array<ArrayBuffer>> {
		const t = this.trainer!;
		return writeSafetensors(t.lora ? await loraTensors(t.lora) : await weightTensors(t.weights), {
			base_model: this.policyDef!.repo
		});
	}

	/** Keep the current adapter (or weights) saved in the browser under the session's name. */
	async #autosave() {
		if (!this.trainer || !this.policyDef || !this.sessionId) return;
		try {
			await saveModel(
				{
					name: this.sessionName,
					baseId: this.policyDef.id,
					mode: this.trainer.settings.mode,
					loraRank: this.trainer.settings.loraRank,
					loraAlpha: this.trainer.settings.loraAlpha,
					passes: this.passes.length,
					constitution: config.constitution
				},
				await this.#sessionBytes(),
				this.sessionId,
				this.sessionCreatedAt
			);
		} catch (e) {
			console.warn('Autosave failed', e);
		}
	}

	async renameSession(name: string) {
		this.sessionName = name;
		if (this.#policyRowId !== null) await renameModel(this.#policyRowId, name).catch(() => {});
		await this.refreshSaved();
	}

	async deleteSaved(meta: SavedMeta) {
		await deleteSaved(meta.id).catch(() => {});
		if (meta.dbId !== undefined) await deleteProjectOfModel(meta.dbId);
		await this.refreshSaved();
	}

	async saveSession(name: string): Promise<SavedMeta> {
		if (!this.trainer || !this.policyDef) throw new Error('Nothing to save yet.');
		const meta = await saveModel(
			{
				name,
				baseId: this.policyDef.id,
				mode: this.trainer.settings.mode,
				loraRank: this.trainer.settings.loraRank,
				loraAlpha: this.trainer.settings.loraAlpha,
				passes: this.passes.length,
				constitution: config.constitution
			},
			await this.#sessionBytes()
		);
		await this.refreshSaved();
		return meta;
	}

	async downloadSession() {
		if (!this.trainer || !this.policyDef) return;
		await this.#download(this.policyDef, this.trainer.settings.mode, this.trainer.settings, await this.#sessionBytes(), 'session');
	}

	async downloadSaved(meta: SavedMeta) {
		await this.#download(getModel(meta.baseId), meta.mode, meta, await loadSavedWeights(meta.id), meta.id);
	}

	async #download(
		def: ModelDef,
		mode: 'lora' | 'full',
		lora: { loraRank: number; loraAlpha: number },
		bytes: Uint8Array<ArrayBuffer>,
		name: string
	) {
		if (mode === 'lora') {
			const cfg = new TextEncoder().encode(peftConfig(def, lora.loraRank, lora.loraAlpha));
			downloadBlob(
				zip([
					{ name: 'adapter_model.safetensors', data: bytes },
					{ name: 'adapter_config.json', data: cfg }
				]),
				`${def.id}-${name}-lora.zip`
			);
		} else {
			downloadBlob(new Blob([bytes]), `${def.id}-${name}-model.safetensors`);
		}
	}

	/**
	 * Parameters for one side of the Test chat. `source` is 'session', a saved
	 * model id, or 'original' (the untouched base of the given model).
	 */
	async #chatSide(source: string, baseId: string): Promise<{ def: ModelDef; tok: Tokenizer; side: ChatSide }> {
		if (source === 'session') {
			if (!this.trainer || !this.policyDef || !this.policyTok) throw new Error('No training session loaded.');
			return { def: this.policyDef, tok: this.policyTok, side: this.trainer.policy() };
		}
		const def = getModel(baseId);
		await this.#ensureChatBase(def);
		const tok = this.policyTok!;
		const original = this.trainer!.original();
		if (source === 'original') return { def, tok, side: original };

		if (this.#chatSaved?.id !== source) {
			if (this.#chatSaved) {
				const { weights, lora } = this.#chatSaved.side;
				if (lora) tree.dispose(lora);
				else tree.dispose(weights);
			}
			const meta = this.saved.find((m) => m.id === source);
			if (!meta) throw new Error('Saved model not found.');
			this.status = `Loading ${meta.name}…`;
			const bytes = await loadSavedWeights(source);
			const side =
				meta.mode === 'lora'
					? { weights: original.weights, lora: parseLora(bytes, def.config.layers) }
					: { weights: parseWeights(bytes, def.config.layers), lora: null };
			this.#chatSaved = { id: source, side };
			this.status = '';
		}
		return { def, tok, side: this.#chatSaved.side };
	}

	/** Make sure the chat can run `def`: reuse the trainer's model, or load a bare one. */
	async #ensureChatBase(def: ModelDef) {
		if (this.policyDef?.id === def.id && this.trainer) return;
		if (!(await this.initGpu())) throw new Error('WebGPU is not available in this browser.');
		this.#disposeAll(true);
		this.loads = [];
		this.policyTok = await loadTokenizer(def);
		const item = this.#track(`${def.label} (for chat)`, def.downloadBytes);
		const weights = await loadWeights(def, np.float32, (x) => Object.assign(item, x));
		item.done = true;
		// A trainer with no passes is the simplest holder for "original" weights.
		this.trainer = new PolicyTrainer(def, weights, null, {
			mode: 'lora',
			learningRate: 0,
			loraRank: config.loraRank,
			loraAlpha: config.loraAlpha,
			temperature: config.temperature,
			microbatch: 8
		});
		this.policyDef = def;
		this.#chatModel = this.trainer.model;
		this.#loadedKey = '';
		this.ready = false;
		this.version++;
	}

	/**
	 * Answer the same conversation with two models at once. Sources are
	 * 'original', 'session', or a saved model id, all on the base `baseId`.
	 * Weights are resolved one side at a time, then both generate in parallel.
	 */
	async chatPair(
		sources: [string, string],
		baseId: string,
		histories: [ChatTurn[], ChatTurn[]],
		onText: (side: 0 | 1, text: string) => void
	): Promise<[string, string]> {
		const sides: { def: ModelDef; tok: Tokenizer; side: ChatSide }[] = [];
		for (const source of sources) sides.push(await this.#chatSide(source, baseId));
		const run = async (k: 0 | 1) => {
			const { def, tok, side } = sides[k];
			const ids = tok.encode(chatPrompt([{ role: 'system', content: def.defaultSystemPrompt }, ...histories[k]]));
			const gen = await generate(this.#chatModel!, side.weights, side.lora, ids, {
				rows: 1,
				maxNew: 160,
				temperature: 0.7,
				stopTokens: def.stopTokens,
				padToken: def.padToken,
				onStep: (tokens) => onText(k, tok.decodeText(tokens[0]))
			});
			return tok.decodeText(gen.tokens[0]);
		};
		return Promise.all([run(0), run(1)]) as Promise<[string, string]>;
	}

	#disposeAll(keepJudge = false) {
		// Reference answers come from the old policy base; judge prefixes stay valid with the same judge.
		this.#references.clear();
		if (!keepJudge) {
			for (const p of this.#prefixes.values()) tree.dispose(p.kv);
			this.#prefixes.clear();
		}
		this.trainer?.dispose();
		this.trainer = null;
		if (!keepJudge) {
			if (this.judgeWeights) tree.dispose(this.judgeWeights);
			this.judgeWeights = null;
			this.judgeModel = null;
		}
		if (this.#chatSaved) {
			const { weights, lora } = this.#chatSaved.side;
			if (lora) tree.dispose(lora);
			else tree.dispose(weights);
		}
		this.#chatSaved = null;
		this.ready = false;
	}
}

export const runtime = new Runtime();

// Test hook for driving training from Playwright; stripped from production builds.
if (import.meta.env.DEV && typeof window !== 'undefined') {
	(window as unknown as { __runtime: Runtime }).__runtime = runtime;
}
