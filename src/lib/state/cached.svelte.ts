import { engine } from '$lib/engine/engine';

/**
 * Which models are already in the browser's cache, for the "downloaded" mark
 * on the models and reward pages. Downloading only starts once you leave the
 * models page, as part of loading.
 */
export const cached: Record<string, boolean> = $state({});

/** Check which of these models are cached, without downloading anything. */
export async function checkCached(ids: string[]) {
	for (const id of ids) {
		if (cached[id]) continue;
		cached[id] = await engine('isCached', id).catch(() => false);
	}
}
