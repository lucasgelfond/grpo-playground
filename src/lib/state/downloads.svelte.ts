import { MODELS, type ModelDef } from '$lib/models/registry';
import { engine } from '$lib/engine/engine';
import { persistStorage } from '$lib/models/weights';

/**
 * Model downloads, kept going across pages. The models page only checks what's
 * already cached; loading (and so downloading) starts once you leave it. The train page's loader shares the same in-flight requests
 * (see weights.ts): both run in the GPU worker, so nothing is fetched twice.
 */
export type DownloadState = { loaded: number; total: number; done: boolean; queued?: boolean; error?: string };

export const downloads: Record<string, DownloadState> = $state({});

const started = new Set<string>();

async function ensure(def: ModelDef) {
	if (started.has(def.id)) return;
	started.add(def.id);
	downloads[def.id] = { loaded: 0, total: def.downloadBytes, done: false };
	try {
		if (await engine('isCached', def.id)) {
			downloads[def.id] = { loaded: def.downloadBytes, total: def.downloadBytes, done: true };
			return;
		}
		await persistStorage();
		await engine('download', def.id, (p) => Object.assign(downloads[def.id], { loaded: p.loaded, total: p.total }));
		downloads[def.id].done = true;
	} catch (e) {
		started.delete(def.id);
		downloads[def.id].error = e instanceof Error ? e.message : String(e);
	}
}

/** Mark which of these models are already in the browser's cache, without downloading anything. */
export async function checkCached(ids: string[]) {
	for (const id of ids) {
		if (downloads[id]) continue;
		const def = MODELS.find((m) => m.id === id)!;
		if (await engine('isCached', id).catch(() => false)) {
			downloads[id] = { loaded: def.downloadBytes, total: def.downloadBytes, done: true };
		}
	}
}

/** Download these models one after another (small first), if not already cached. */
export async function downloadAll(ids: string[]) {
	const defs = ids.map((id) => MODELS.find((m) => m.id === id)!).sort((a, b) => a.downloadBytes - b.downloadBytes);
	for (const def of defs) {
		if (!started.has(def.id) && !downloads[def.id]?.done) {
			downloads[def.id] = { loaded: 0, total: def.downloadBytes, done: false, queued: true };
		}
	}
	for (const def of defs) await ensure(def);
}
