import { downloadBlob, peftConfig, zip } from '$lib/export';
import { engine } from '$lib/engine/engine';
import { getModel, type ChatTurn, type ModelDef } from '$lib/models/registry';
import { persistStorage } from '$lib/models/weights';
import type { Match, Verdict } from '$lib/rl/judge';
import { createProject, deleteProjectOfModel, listPolicyModels, recordPass, renameModel } from '$lib/db';
import { deleteSaved, loadSavedWeights, newModelId, type SavedMeta } from '$lib/saved';
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
	 * One answer from the untouched original model, entered into
	 * the matches (as index answers.length) but never trained on. How often the
	 * trained model's answers beat it is the pass's absolute progress measure.
	 */
	reference?: Answer;
	/** Mean P(a trained answer beats the original model's answer). */
	vsOriginal?: number;
	/** Every head-to-head choice the judge made, and how many were scheduled. */
	matches?: Match[];
	matchesTotal?: number;
	/** [layer][target] ||ΔW|| from the original weights, after this pass. */
	layerNorms?: number[][];
};

export type LoadItem = { label: string; loaded: number; total: number; downloaded: number; done: boolean };

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

/** Copy `src` into the reactive `target`, touching only what changed, so the view updates in place. */
function merge<T extends object>(target: T, src: T) {
	for (const [key, value] of Object.entries(src)) {
		const current = (target as Record<string, unknown>)[key];
		if (value && typeof value === 'object' && current && typeof current === 'object' && Array.isArray(value) === Array.isArray(current)) {
			merge(current, value);
			if (Array.isArray(value)) (current as unknown[]).length = value.length;
		} else if (current !== value) {
			(target as Record<string, unknown>)[key] = value;
		}
	}
}

/**
 * The app's training state. The models themselves live in a worker
 * (engine/engine.worker.ts); this drives it and mirrors what it reports.
 */
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
	/** A chat message is waiting for the current training pass to finish. */
	chatWaiting = $state(false);
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

	/** The model loaded in the worker. */
	policyDef = $state<ModelDef | null>(null);
	#loadedKey = '';
	#promptOrder: number[] = [];
	#promptCursor = 0;
	#gpuTurn: Promise<unknown> = Promise.resolve();

	get currentPass(): Pass | undefined {
		return this.viewing === null ? this.passes.at(-1) : this.passes[this.viewing];
	}

	async initGpu(): Promise<boolean> {
		if (this.gpu === 'unknown') this.gpu = (await engine('initGpu', undefined)) ? 'ok' : 'missing';
		return this.gpu === 'ok';
	}

	get needsReload(): boolean {
		return this.ready && this.#loadedKey !== trainingKey(config);
	}

	/** Download (or read from cache) both models and set up a fresh trainer. */
	async load(): Promise<void> {
		if (this.loading) return;
		const key = trainingKey(config);
		if (this.ready && key === this.#loadedKey) return;
		this.loading = true;
		this.ready = false;
		this.error = null;
		this.loads = [];
		try {
			if (!(await this.initGpu())) throw new Error('WebGPU is not available in this browser.');
			await persistStorage();
			await engine(
				'load',
				{
					policyId: config.policyId,
					judgeId: config.judgeId,
					mode: config.mode,
					loraRank: config.loraRank,
					loraAlpha: config.loraAlpha,
					learningRate: learningRate(config),
					temperature: config.temperature
				},
				(u) => {
					if (u.status !== undefined) this.status = u.status;
					if (u.loads) this.loads = u.loads;
				}
			);
			this.policyDef = getModel(config.policyId);
			this.passes = [];
			this.viewing = null;
			this.#promptOrder = [];
			this.#loadedKey = key;
			this.sessionName = petname();
			this.sessionId = newModelId(this.sessionName);
			this.sessionCreatedAt = new Date().toISOString();
			await this.#recordProject(config.policyId, config.judgeId);
			this.ready = true;
			this.status = '';
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

	#saveTarget(id: string, name: string, createdAt: string) {
		return { id, name, createdAt, passes: this.passes.length, constitution: config.constitution };
	}

	/** Sample a group of answers, judge them, and apply one GRPO update. */
	async runPass(): Promise<void> {
		if (this.running) return;
		this.running = true;
		return this.#exclusive(() => this.#runPass());
	}

	async #runPass(): Promise<void> {
		this.error = null;
		try {
			await this.load();
			if (!this.ready) return;
			const prompt = this.#nextPrompt();
			const index = this.passes.length;
			this.passes.push({
				index,
				prompt,
				phase: 'prefill',
				answers: Array.from({ length: config.groupSize }, () => ({ tokens: [], text: '', stopped: false, pieces: [] }))
			});
			const pass = this.passes[index];
			const done = await engine(
				'runPass',
				{
					index,
					prompt,
					config: $state.snapshot(config),
					save: { ...this.#saveTarget(this.sessionId, this.sessionName, this.sessionCreatedAt), passes: index + 1 }
				},
				(u) => {
					if (u.status !== undefined) this.status = u.status;
					if (u.pass) merge(pass, u.pass);
				}
			);
			merge(pass, done);
			if (this.#projectId !== null && this.#policyRowId !== null) {
				await recordPass(this.#projectId, this.#policyRowId, pass.index, prompt, $state.snapshot(pass)).catch((e) =>
					console.warn('Could not record the pass', e)
				);
			}
			await this.refreshSaved();
			this.status = pass.skipped
				? `Pass ${index + 1}: the answers scored about the same, so there was nothing to learn`
				: `Pass ${index + 1} done`;
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

	/**
	 * Get ahead of the train page: load the models (from cache, or downloading
	 * them) and precompute the judge's prompt and the original model's answer
	 * for the first prompts. Safe to call repeatedly; it waits its turn.
	 * `warmOnly` skips (re)loading, for refreshing after prompt edits.
	 */
	async preload(warmOnly = false): Promise<void> {
		if (!warmOnly) await this.load();
		if (!this.ready || this.running) return;
		const prompts = config.prompts.map((p) => p.trim()).filter(Boolean);
		const input = { prompts, constitution: config.constitution, maxNew: config.maxNew };
		await this.#exclusive(() => engine('warm', input)).catch((e) => console.warn('Warm-up failed', e));
	}

	/** Training passes and chat share the GPU and the policy's weights, so they take turns. */
	#exclusive<T>(fn: () => Promise<T>): Promise<T> {
		const next = this.#gpuTurn.then(fn, fn);
		this.#gpuTurn = next.catch(() => {});
		return next;
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
		const meta = await this.#exclusive(() => engine('save', this.#saveTarget('', name, '')));
		await this.refreshSaved();
		return meta;
	}

	async downloadSession() {
		if (!this.ready || !this.policyDef) return;
		const bytes = await this.#exclusive(() => engine('sessionBytes', undefined));
		this.#download(this.policyDef, config.mode, config, bytes, 'session');
	}

	async downloadSaved(meta: SavedMeta) {
		this.#download(getModel(meta.baseId), meta.mode, meta, await loadSavedWeights(meta.id), meta.id);
	}

	#download(
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
	 * Answer the same conversation with two models at once. Sources are
	 * 'original', 'session', or a saved model id, all on the base `baseId`.
	 */
	async chatPair(
		sources: [string, string],
		baseId: string,
		histories: [ChatTurn[], ChatTurn[]],
		onText: (side: 0 | 1, text: string) => void
	): Promise<[string, string]> {
		this.chatWaiting = this.running;
		return this.#exclusive(async () => {
			this.chatWaiting = false;
			if (!(await this.initGpu())) throw new Error('WebGPU is not available in this browser.');
			const replies = await engine(
				'chatPair',
				{
					sources,
					baseId,
					histories,
					saved: $state.snapshot(this.saved),
					loraRank: config.loraRank,
					loraAlpha: config.loraAlpha
				},
				(u) => {
					if (u.side !== undefined) onText(u.side, u.text!);
					if (u.loads) this.loads = u.loads;
					if (u.replaced) {
						// The chat loaded a bare model in place of the training session.
						this.policyDef = getModel(baseId);
						this.#loadedKey = '';
						this.ready = false;
					}
				}
			);
			replies.forEach((text, k) => onText(k as 0 | 1, text));
			return replies;
		});
	}
}

export const runtime = new Runtime();

// Test hook for driving training from Playwright; stripped from production builds.
if (import.meta.env.DEV && typeof window !== 'undefined') {
	(window as unknown as { __runtime: Runtime }).__runtime = runtime;
}
