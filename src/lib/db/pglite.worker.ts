import { PGlite } from '@electric-sql/pglite';
import { worker } from '@electric-sql/pglite/worker';

// Postgres in a worker, stored in OPFS (its sync access handles only exist in
// workers). Falls back to IndexedDB where OPFS access handles are unavailable.
worker({
	async init() {
		try {
			return await PGlite.create('opfs-ahp://jax-rl-model');
		} catch (e) {
			console.warn('PGlite OPFS storage unavailable, using IndexedDB', e);
			return await PGlite.create('idb://jax-rl-model');
		}
	}
});
