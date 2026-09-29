import type { Engine, Request, Response } from './engine.worker';

type Method = { [K in keyof Engine]: Engine[K] extends (...args: never[]) => Promise<unknown> ? K : never }[keyof Engine];
type Input<K extends Method> = Parameters<Engine[K]>[0];
type Progress<K extends Method> = Parameters<Engine[K]>[1];

let worker: Worker | null = null;
let nextId = 0;
const pending = new Map<number, { resolve: (r: unknown) => void; reject: (e: Error) => void; progress?: (p: never) => void }>();

function getWorker(): Worker {
	if (!worker) {
		worker = new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' });
		worker.onmessage = ({ data }: MessageEvent<Response>) => {
			const call = pending.get(data.id);
			if (!call) return;
			if ('progress' in data) return call.progress?.(data.progress as never);
			pending.delete(data.id);
			if (data.error !== undefined) call.reject(new Error(data.error));
			else call.resolve(data.result);
		};
	}
	return worker;
}

/** Run an Engine method in the GPU worker. */
export function engine<K extends Method>(
	method: K,
	input: Input<K>,
	progress?: Progress<K>
): Promise<Awaited<ReturnType<Engine[K]>>> {
	const id = nextId++;
	return new Promise((resolve, reject) => {
		pending.set(id, { resolve: resolve as (r: unknown) => void, reject, progress: progress as (p: never) => void });
		getWorker().postMessage({ id, method, input } satisfies Request);
	});
}
