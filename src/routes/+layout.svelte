<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import { onMount } from 'svelte';

	import { config } from '$lib/state/config.svelte';
	import { runtime } from '$lib/state/runtime.svelte';

	let { children } = $props();

	const STEPS = [
		{ href: '/', label: 'models' },
		{ href: '/prompts', label: 'prompts & judge' },
		{ href: '/train', label: 'train' },
		{ href: '/evaluate', label: 'evaluate' }
	];
	const current = $derived(STEPS.findIndex((s) => s.href === page.url.pathname));
	const title = $derived(`${config.algorithm.toUpperCase()}, in the browser`);
	// DPO works but isn't exposed in the UI (mostly because GRPO playground was a better name, LOL). if you do ?mode=dpo or go to dpo.lucasgelfond.online you can try the DPO version
	// ?mode=grpo switches back; dpo.lucasgelfond.online forwards here with ?mode=dpo (see app.html).
	onMount(() => {
		const mode = page.url.searchParams.get('mode') ?? page.url.searchParams.get('algo');
		if (mode === 'dpo' || mode === 'grpo') config.algorithm = mode;
	});
	// Models load in the background from the prompts page on; show where that's at everywhere.
	const loadedPct = $derived.by(() => {
		const total = runtime.loads.reduce((s, l) => s + l.total, 0);
		return total ? Math.round((100 * runtime.loads.reduce((s, l) => s + l.loaded, 0)) / total) : 0;
	});
</script>

<svelte:head>
	<title>{title}</title>
</svelte:head>

<div class="flex min-h-screen flex-col overflow-x-clip bg-gray-950 text-gray-100">
	<main class="mx-auto w-full max-w-5xl flex-1 space-y-8 px-8 pt-12 sm:px-16">
		<header class="space-y-4">
			<h1 class="text-sm font-bold text-white">{title}</h1>
			<!-- The steps read as a progression. -->
			<nav class="flex flex-wrap items-center gap-2 text-xs">
				{#each STEPS as step, i (step.href)}
					{#if i > 0}<span class="text-gray-700">›</span>{/if}
					<a
						href={step.href}
						class={[
							'flex items-center gap-1.5 rounded px-1.5 py-0.5 transition-colors',
							i === current ? 'text-white' : i < current ? 'text-gray-300 hover:text-white' : 'text-gray-500 hover:text-gray-300'
						]}
					>
						<span
							class={[
								'flex size-4 items-center justify-center rounded-full text-[10px] tabular-nums',
								i === current ? 'bg-blue-600 text-white' : i < current ? 'bg-gray-700 text-gray-200' : 'border border-gray-700 text-gray-500'
							]}>{i + 1}</span
						>
						{step.label}
					</a>
				{/each}
				<span class="ml-auto text-gray-500 tabular-nums">
					{#if runtime.loading}loading models · {loadedPct}%
					{:else if runtime.ready}<span class="text-up">●</span> models ready{/if}
				</span>
			</nav>
		</header>
		{@render children()}
	</main>
	<!-- Pinned to the bottom, as far from it as the title is from the top. -->
	<footer class="mx-auto w-full max-w-5xl px-8 pt-16 pb-12 text-xs text-gray-500 sm:px-16">
		Source <a href="https://github.com/lucasgelfond/grpo-playground" class="text-gray-300 underline hover:text-white">here</a>. Built by
		<a href="https://lucasgelfond.online/" class="text-gray-300 underline hover:text-white">Lucas Gelfond</a> in New York City.
	</footer>
</div>
