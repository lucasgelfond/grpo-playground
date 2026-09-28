<script lang="ts">
	import { goto } from '$app/navigation';
	import { tick } from 'svelte';

	import ClipboardCopyIcon from '@lucide/svelte/icons/clipboard-copy';
	import CheckIcon from '@lucide/svelte/icons/check';
	import FileInputIcon from '@lucide/svelte/icons/file-input';

	import { TASKS, type Task } from '$lib/presets';
	import { config } from '$lib/state/config.svelte';
	import { runtime } from '$lib/state/runtime.svelte';

	const PRESET_PROMPTS = 5;

	let importing = $state(false);
	let importText = $state('');
	let importError = $state('');
	let copied = $state(false);
	let list: HTMLOListElement;

	const same = (a: string[], b: string[]) => a.length === b.length && a.every((x, i) => x === b[i]);
	const activeTask = $derived(
		TASKS.find(
			(t) => config.constitution === t.constitution && same(config.prompts, t.prompts.slice(0, PRESET_PROMPTS))
		)
	);

	function loadTask(t: Task) {
		config.prompts = t.prompts.slice(0, PRESET_PROMPTS);
		config.constitution = t.constitution;
		config.rule = t.rule ?? null;
	}

	async function copyJson() {
		await navigator.clipboard.writeText(JSON.stringify(config.prompts.filter((p) => p.trim()), null, 2));
		copied = true;
		setTimeout(() => (copied = false), 1500);
	}

	/** Accept a JSON array of strings, {prompts: [...]}, or one prompt per line. */
	function applyImport() {
		importError = '';
		const text = importText.trim();
		let prompts: string[];
		try {
			if (text.startsWith('[') || text.startsWith('{')) {
				const parsed = JSON.parse(text);
				const arr = Array.isArray(parsed) ? parsed : parsed.prompts;
				if (!Array.isArray(arr)) throw new Error('Expected an array of strings, or {"prompts": [...]}.');
				prompts = arr.map((p: unknown) => (typeof p === 'string' ? p : ((p as { prompt?: string })?.prompt ?? ''))).filter(Boolean);
			} else {
				prompts = text.split('\n').map((l) => l.trim()).filter(Boolean);
			}
		} catch (e) {
			importError = e instanceof Error ? e.message : String(e);
			return;
		}
		if (prompts.length === 0) {
			importError = 'No prompts found.';
			return;
		}
		config.prompts = prompts;
		importing = false;
		importText = '';
	}

	async function addPrompt(after = config.prompts.length - 1) {
		config.prompts.splice(after + 1, 0, '');
		await tick();
		list.querySelectorAll('input')[after + 1]?.focus();
	}

	async function start() {
		await goto('/train');
		void runtime.load();
	}
</script>

<div class="space-y-8">
	<div class="space-y-4">
		<p class="max-w-2xl text-xs leading-relaxed text-gray-400">
			To fine-tune, we'll ask the base model all of the prompts on the left. Based on how it answers (as evaluated by
			the judge prompt on the right), we'll update its weights.
		</p>
		<label class="flex items-center gap-3 text-xs text-gray-400">
			preset
			<select
				class="field w-64"
				value={activeTask?.id ?? 'custom'}
				onchange={(e) => {
					const id = e.currentTarget.value;
					const task = TASKS.find((t) => t.id === id);
					if (task) loadTask(task);
					else {
						config.prompts = [''];
						config.constitution = '';
						config.rule = null;
					}
				}}
			>
				<option value="custom">custom</option>
				{#each TASKS as task (task.id)}<option value={task.id}>{task.label.toLowerCase()}</option>{/each}
			</select>
		</label>
	</div>

	<div class="grid gap-6 lg:grid-cols-2">
		<section class="space-y-2">
			<div class="flex h-6 items-center justify-between">
				<h2 class="label">prompts</h2>
				<div class="flex gap-1 text-ink-soft">
					<button type="button" class="rounded-token p-0.5 hover:bg-hover hover:text-ink" title="Copy prompts as JSON" aria-label="Copy prompts as JSON" onclick={copyJson}>
						{#if copied}<CheckIcon class="size-4" />{:else}<ClipboardCopyIcon class="size-4" />{/if}
					</button>
					<button type="button" class="rounded-token p-0.5 hover:bg-hover hover:text-ink" title="Import prompts from JSON" aria-label="Import prompts from JSON" onclick={() => (importing = !importing)}>
						<FileInputIcon class="size-4" />
					</button>
				</div>
			</div>

			{#if importing}
				<div class="card space-y-2 p-3">
					<textarea
						class="field h-32 font-terminal text-[0.78rem]"
						placeholder={'["Why is the sky blue?", …]  or one prompt per line'}
						bind:value={importText}
					></textarea>
					{#if importError}<p class="text-[0.8rem] text-danger">{importError}</p>{/if}
					<div class="flex justify-end gap-2">
						<button type="button" class="btn" onclick={() => (importing = false)}>Cancel</button>
						<button type="button" class="btn-primary" onclick={applyImport}>Replace prompts</button>
					</div>
				</div>
			{/if}

			<ol class="space-y-1.5" bind:this={list}>
				{#each config.prompts as _, i (i)}
					<li class="group relative">
						<input
							class="field pr-8"
							bind:value={config.prompts[i]}
							placeholder="Ask something…"
							onkeydown={(e) => {
								if (e.key === 'Enter') {
									e.preventDefault();
									addPrompt(i);
								}
							}}
						/>
						<button
							type="button"
							class="absolute top-1/2 right-1.5 -translate-y-1/2 rounded-token px-1.5 text-ink-soft opacity-0 group-hover:opacity-100 hover:text-ink focus:opacity-100"
							aria-label="Remove prompt {i + 1}"
							onclick={() => config.prompts.splice(i, 1)}
						>
							×
						</button>
					</li>
				{/each}
			</ol>
			<button
				type="button"
				class="w-full rounded-token border border-dashed border-rule-strong px-2 py-1 text-left text-xs text-ink-soft hover:border-accent hover:text-ink"
				onclick={() => addPrompt()}
			>
				+ add a prompt
			</button>
		</section>

		<section class="space-y-2">
			<div class="flex h-6 items-center">
				<h2 class="label">judge prompt</h2>
			</div>
			<textarea
				class="field min-h-[16rem] resize-y p-3 leading-relaxed"
				bind:value={config.constitution}
				spellcheck="true"
				placeholder="What should the judge prefer? e.g. Reward answers that are short and friendly."
			></textarea>
		</section>
	</div>

	<div class="flex items-center justify-between">
		<a href="/" class="text-[0.85rem] text-ink-soft hover:text-ink">← back</a>
		<button type="button" class="btn-primary" onclick={start} disabled={runtime.gpu === 'missing'}>
			train →
		</button>
	</div>
</div>
