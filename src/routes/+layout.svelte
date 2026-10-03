<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';

	let { children } = $props();

	const STEPS = [
		{ href: '/', label: 'models' },
		{ href: '/prompts', label: 'prompts & judge' },
		{ href: '/train', label: 'train' },
		{ href: '/evaluate', label: 'evaluate' }
	];
	const current = $derived(STEPS.findIndex((s) => s.href === page.url.pathname));
</script>

<svelte:head>
	<title>GRPO, in the browser</title>
</svelte:head>

<div class="min-h-screen overflow-x-clip bg-gray-950 text-gray-100">
	<main class="mx-auto max-w-5xl space-y-8 px-8 py-12 sm:px-16">
		<header class="space-y-4">
			<h1 class="text-sm font-bold text-white">GRPO, in the browser</h1>
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
			</nav>
		</header>
		{@render children()}
		<footer class="border-t border-gray-800 pt-6 text-xs text-gray-500">
			Source <a href="https://github.com/lucasgelfond/grpo-playground" class="text-gray-300 underline hover:text-white">here</a>. Built by
			<a href="https://lucasgelfond.online/" class="text-gray-300 underline hover:text-white">Lucas Gelfond</a> in New York City.
		</footer>
	</main>
</div>
