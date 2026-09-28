import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	optimizeDeps: {
		// PGlite ships its own wasm and data files; pre-bundling breaks their URLs.
		exclude: ['@electric-sql/pglite']
	},
	worker: {
		format: 'es'
	}
});
