<script lang="ts">
	import { goto } from '$app/navigation';
	import { onMount, tick } from 'svelte';

	import ClipboardCopyIcon from '@lucide/svelte/icons/clipboard-copy';
	import CheckIcon from '@lucide/svelte/icons/check';
	import FileInputIcon from '@lucide/svelte/icons/file-input';

	import { config } from '$lib/state/config.svelte';
	import { runtime } from '$lib/state/runtime.svelte';

	let importing = $state(false);
	let importText = $state('');
	let importError = $state('');
	let copied = $state(false);
	let list: HTMLOListElement;

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

	// Load the models and precompute the first passes' work while prompts are edited.
	onMount(() => void runtime.preload());
	$effect(() => {
		// Re-runs when the prompts, the judge prompt or the answer length change.
		JSON.stringify([config.prompts, config.constitution, config.maxNew]);
		if (!runtime.ready) return;
		const t = setTimeout(() => void runtime.preload(true), 800);
		return () => clearTimeout(t);
	});

	async function start() {
		await goto('/train');
		void runtime.load();
	}
</script>

<div class="max-w-3xl space-y-8">
	<p class="max-w-2xl text-xs leading-relaxed text-gray-400">
		Each pass asks the model one of these prompts. How its answers score (with the reward you picked) decides how its
		weights change. Prompts the training never sees are a good test of whether a behavior generalizes.
	</p>

	<div class="max-w-3xl">
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

	</div>

	<details class="group card p-3">
		<summary class="cursor-pointer list-none text-xs text-gray-400 select-none hover:text-gray-200">
			<span class="inline-block transition-transform group-open:rotate-90">▶</span> advanced
		</summary>
		<div class="mt-4 grid gap-x-8 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
			{#snippet num(label: string, value: number, set: (v: number) => void, min: number, max: number, step: number)}
				<label class="space-y-1 text-xs text-gray-400">
					<span class="whitespace-nowrap">{label}</span>
					<input
						class="field tabular-nums"
						type="number"
						{min}
						{max}
						{step}
						{value}
						onchange={(e) => {
							const v = e.currentTarget.valueAsNumber;
							if (Number.isFinite(v)) set(Math.min(max, Math.max(min, v)));
						}}
					/>
				</label>
			{/snippet}
			{@render num('answers per pass', config.groupSize, (v) => (config.groupSize = Math.round(v)), 2, 8, 1)}
			{@render num('max tokens per answer', config.maxNew, (v) => (config.maxNew = Math.round(v)), 16, 256, 16)}
			{@render num('temperature', config.temperature, (v) => (config.temperature = v), 0.1, 1.5, 0.1)}
			{#if config.reward === 'ai'}
				<label class="space-y-1 text-xs text-gray-400">
					<span>judge</span>
					<select class="field" bind:value={config.judgeMode}>
						<option value="pointwise">scores each answer (yes / no)</option>
						<option value="pairwise">compares answers in pairs (A / B)</option>
					</select>
				</label>
			{/if}
			{#if config.reward === 'ai' && config.judgeMode === 'pairwise'}
				{@render num('judge matches per answer', config.matchesPerAnswer, (v) => (config.matchesPerAnswer = Math.round(v)), 1, 7, 1)}
			{/if}
			{#if config.mode === 'lora'}
				{@render num('learning rate', config.loraLearningRate, (v) => (config.loraLearningRate = v), 1e-6, 1e-2, 1e-5)}
				{@render num('LoRA rank', config.loraRank, (v) => (config.loraRank = Math.round(v)), 1, 64, 1)}
			{:else}
				{@render num('learning rate', config.fullLearningRate, (v) => (config.fullLearningRate = v), 1e-7, 1e-3, 1e-6)}
			{/if}
			{@render num('KL penalty β', config.klBeta, (v) => (config.klBeta = v), 0, 1, 0.01)}
			{#if config.reward === 'ai' && config.judgeMode === 'pairwise'}
				<label class="flex items-center gap-2 self-end pb-1.5 text-xs text-gray-400">
					<input type="checkbox" bind:checked={config.bothOrders} />
					judge both orders <span class="text-gray-600">(2× judge time)</span>
				</label>
			{/if}
			<label class="space-y-1 text-xs text-gray-400">
				<span>prompt order</span>
				<select class="field" bind:value={config.promptOrder}>
					<option value="shuffle">shuffled</option>
					<option value="sequential">in order</option>
				</select>
			</label>
		</div>
	</details>

	<div class="flex items-center justify-between">
		<a href="/reward" class="text-[0.85rem] text-ink-soft hover:text-ink">← back</a>
		<button type="button" class="btn-primary" onclick={start} disabled={runtime.gpu === 'missing'}>
			train →
		</button>
	</div>
</div>
