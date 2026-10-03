import { PGliteWorker } from '@electric-sql/pglite/worker';

/**
 * Everything about training runs, in a local Postgres (PGlite):
 *
 *   projects  one training session: which models, method and settings
 *   models    the models in a project: its initial model, its judge, and the
 *             trained policy (petname, dates, passes, OPFS path of its weights)
 *   prompts   a project's prompts, typed JUDGE (the judge prompt) or
 *             EVALUATION (the prompts it trains on)
 *   passes    every pass: prompt, answers, the judge's choices and metrics
 *
 * Model weights stay in OPFS (see saved.ts); rows only point at them.
 */

const SCHEMA = `
	CREATE TABLE IF NOT EXISTS projects (
		id SERIAL PRIMARY KEY,
		name TEXT NOT NULL,
		method TEXT NOT NULL DEFAULT 'grpo',
		mode TEXT NOT NULL,
		settings JSONB NOT NULL,
		created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
	);

	CREATE TABLE IF NOT EXISTS models (
		id SERIAL PRIMARY KEY,
		project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
		kind TEXT NOT NULL CHECK (kind IN ('initial', 'judge', 'policy')),
		base_model_id TEXT NOT NULL,
		petname TEXT,
		weights_path TEXT,
		passes INTEGER NOT NULL DEFAULT 0,
		created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
		updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
	);

	CREATE TABLE IF NOT EXISTS prompts (
		id SERIAL PRIMARY KEY,
		project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
		prompt TEXT NOT NULL,
		type TEXT NOT NULL CHECK (type IN ('JUDGE', 'EVALUATION'))
	);

	CREATE TABLE IF NOT EXISTS passes (
		id SERIAL PRIMARY KEY,
		project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
		model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
		pass_index INTEGER NOT NULL,
		prompt TEXT NOT NULL,
		data JSONB NOT NULL,
		created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
	);

	CREATE INDEX IF NOT EXISTS idx_models_project ON models(project_id);
	CREATE INDEX IF NOT EXISTS idx_prompts_project ON prompts(project_id);
	CREATE INDEX IF NOT EXISTS idx_passes_project ON passes(project_id);
`;

let db: Promise<PGliteWorker> | null = null;

export function getDb(): Promise<PGliteWorker> {
	db ??= (async () => {
		const pg = new PGliteWorker(new Worker(new URL('./pglite.worker.ts', import.meta.url), { type: 'module' }));
		await pg.waitReady;
		await pg.exec(SCHEMA);
		return pg;
	})();
	return db;
}

export type ProjectInput = {
	name: string;
	mode: 'lora' | 'full';
	settings: Record<string, unknown>;
	initialModelId: string;
	judgeModelId: string;
	policy: { petname: string; weightsPath: string };
	judgePrompt: string;
	prompts: string[];
};

/** Record a new training session. Returns the project and its policy model's ids. */
export async function createProject(p: ProjectInput): Promise<{ projectId: number; policyId: number }> {
	const pg = await getDb();
	return pg.transaction(async (tx) => {
		const { rows } = await tx.query<{ id: number }>(
			`INSERT INTO projects (name, mode, settings) VALUES ($1, $2, $3) RETURNING id`,
			[p.name, p.mode, JSON.stringify(p.settings)]
		);
		const projectId = rows[0].id;
		await tx.query(`INSERT INTO models (project_id, kind, base_model_id) VALUES ($1, 'initial', $2), ($1, 'judge', $3)`, [
			projectId,
			p.initialModelId,
			p.judgeModelId
		]);
		const policy = await tx.query<{ id: number }>(
			`INSERT INTO models (project_id, kind, base_model_id, petname, weights_path) VALUES ($1, 'policy', $2, $3, $4) RETURNING id`,
			[projectId, p.initialModelId, p.policy.petname, p.policy.weightsPath]
		);
		await tx.query(`INSERT INTO prompts (project_id, prompt, type) VALUES ($1, $2, 'JUDGE')`, [projectId, p.judgePrompt]);
		for (const prompt of p.prompts) {
			await tx.query(`INSERT INTO prompts (project_id, prompt, type) VALUES ($1, $2, 'EVALUATION')`, [projectId, prompt]);
		}
		return { projectId, policyId: policy.rows[0].id };
	});
}

export async function recordPass(projectId: number, modelId: number, index: number, prompt: string, data: unknown) {
	const pg = await getDb();
	await pg.query(`INSERT INTO passes (project_id, model_id, pass_index, prompt, data) VALUES ($1, $2, $3, $4, $5)`, [
		projectId,
		modelId,
		index,
		prompt,
		JSON.stringify(data)
	]);
	await pg.query(`UPDATE models SET passes = $1, updated_at = NOW() WHERE id = $2`, [index + 1, modelId]);
}

export async function renameModel(modelId: number, petname: string) {
	const pg = await getDb();
	await pg.query(`UPDATE models SET petname = $1, updated_at = NOW() WHERE id = $2`, [petname, modelId]);
}

export type PolicyModelRow = {
	id: number;
	project_id: number;
	petname: string;
	base_model_id: string;
	weights_path: string;
	passes: number;
	mode: 'lora' | 'full';
	settings: Record<string, unknown>;
	judge_prompt: string | null;
	created_at: string;
	updated_at: string;
};

/** Trained models that have at least one pass, newest first. */
export async function listPolicyModels(): Promise<PolicyModelRow[]> {
	const pg = await getDb();
	const { rows } = await pg.query<PolicyModelRow>(`
		SELECT m.id, m.project_id, m.petname, m.base_model_id, m.weights_path, m.passes,
		       m.created_at, m.updated_at, p.mode, p.settings,
		       (SELECT prompt FROM prompts WHERE project_id = p.id AND type = 'JUDGE' LIMIT 1) AS judge_prompt
		FROM models m JOIN projects p ON p.id = m.project_id
		WHERE m.kind = 'policy' AND m.passes > 0
		ORDER BY m.updated_at DESC`);
	return rows;
}

/** Delete a trained model and the project it belongs to (prompts and passes cascade). */
export async function deleteProjectOfModel(modelId: number) {
	const pg = await getDb();
	await pg.query(`DELETE FROM projects WHERE id = (SELECT project_id FROM models WHERE id = $1)`, [modelId]);
}
