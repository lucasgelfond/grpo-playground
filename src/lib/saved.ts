/**
 * Fine-tuned models saved under a name in OPFS:
 *   saved-models/<id>/meta.json
 *   saved-models/<id>/weights.safetensors   (LoRA adapter or full weights)
 */

export type SavedMeta = {
	/** OPFS directory of the weights. */
	id: string;
	/** The model's row in the database, when listed from there. */
	dbId?: number;
	name: string;
	createdAt: string;
	updatedAt?: string;
	baseId: string;
	mode: 'lora' | 'full';
	loraRank: number;
	loraAlpha: number;
	passes: number;
	constitution: string;
	bytes: number;
};

async function root(): Promise<FileSystemDirectoryHandle> {
	const dir = await navigator.storage.getDirectory();
	return dir.getDirectoryHandle('saved-models', { create: true });
}

async function writeFile(dir: FileSystemDirectoryHandle, name: string, data: Blob | Uint8Array<ArrayBuffer> | string) {
	const handle = await dir.getFileHandle(name, { create: true });
	const writable = await handle.createWritable();
	await writable.write(data);
	await writable.close();
}

function slug(name: string): string {
	const base = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'model';
	return `${base}-${Date.now().toString(36)}`;
}

export function newModelId(name: string): string {
	return slug(name);
}

/** Save (or overwrite, when `id` is given) a model. */
export async function saveModel(
	meta: Omit<SavedMeta, 'id' | 'createdAt' | 'bytes'>,
	weights: Uint8Array<ArrayBuffer>,
	id = slug(meta.name),
	createdAt = new Date().toISOString()
): Promise<SavedMeta> {
	const full: SavedMeta = {
		...meta,
		id,
		createdAt,
		updatedAt: new Date().toISOString(),
		bytes: weights.byteLength
	};
	const dir = await (await root()).getDirectoryHandle(full.id, { create: true });
	await writeFile(dir, 'weights.safetensors', weights);
	await writeFile(dir, 'meta.json', JSON.stringify(full, null, 2));
	return full;
}

export async function listSaved(): Promise<SavedMeta[]> {
	const out: SavedMeta[] = [];
	try {
		for await (const handle of (await root() as any).values() as AsyncIterable<FileSystemHandle>) {
			if (handle.kind !== 'directory') continue;
			try {
				const file = await (await (handle as FileSystemDirectoryHandle).getFileHandle('meta.json')).getFile();
				out.push(JSON.parse(await file.text()));
			} catch {
				// Half-written entry; skip it.
			}
		}
	} catch {
		return [];
	}
	return out.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export async function loadSavedWeights(id: string): Promise<Uint8Array<ArrayBuffer>> {
	const dir = await (await root()).getDirectoryHandle(id);
	const file = await (await dir.getFileHandle('weights.safetensors')).getFile();
	return new Uint8Array(await file.arrayBuffer());
}

export async function deleteSaved(id: string): Promise<void> {
	await (await root()).removeEntry(id, { recursive: true });
}

export async function renameSaved(id: string, name: string): Promise<void> {
	const dir = await (await root()).getDirectoryHandle(id);
	const file = await (await dir.getFileHandle('meta.json')).getFile();
	const meta = { ...JSON.parse(await file.text()), name };
	await writeFile(dir, 'meta.json', JSON.stringify(meta, null, 2));
}
