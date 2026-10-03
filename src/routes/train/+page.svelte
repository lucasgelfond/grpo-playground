<script lang="ts">
	import { onMount } from 'svelte';

	import LayerGrid from '$lib/components/LayerGrid.svelte';
	import LineChart, { type Series } from '$lib/components/LineChart.svelte';
	import PassGraph from '$lib/components/PassGraph.svelte';
	import { formatBytes, getModel } from '$lib/models/registry';
	import { config } from '$lib/state/config.svelte';
	import { judgeSummary, TASKS } from '$lib/presets';
	import { CHECKS } from '$lib/rl/rules';
	import { runtime, type Pass } from '$lib/state/runtime.svelte';

	let gridView = $state<'total' | 'pass'>('total');

	const pass = $derived(runtime.currentPass);
	const passIdx = $derived(pass?.index ?? -1);
	const prevPass = $derived(passIdx > 0 ? runtime.passes[passIdx - 1] : undefined);
	const done = $derived(runtime.passes.filter((p) => p.phase === 'done'));
	const sumNorms = (n: number[][] | undefined) => n?.flat().reduce((s, v) => s + v, 0);

	// Largest |Δlog p| in the pass, so token colors are comparable across answers.
	const tokenScale = $derived.by(() => {
		let m = 0;
		for (const a of pass?.answers ?? [])
			a.logpAfter?.forEach((v, i) => (m = Math.max(m, Math.abs(v - a.logpBefore![i]))));
		return Math.max(m, 0.05);
	});

	const rewardSeries = $derived([
		...(done.some((p) => p.vsOriginal !== undefined)
			? [{ label: 'win rate vs original model', values: done.map((p) => p.vsOriginal), color: 'var(--color-up)', width: 2.5 }]
			: []),
		{ label: 'mean reward', values: done.map((p) => p.meanScore), color: 'var(--color-accent)', width: 2.25 },
		{
			label: 'best answer',
			values: done.map((p) => Math.max(...p.answers.map((a) => a.verdict?.score ?? 0))),
			color: 'oklch(0.65 0.13 250)',
			width: 1.25
		},
		{
			label: 'worst answer',
			values: done.map((p) => Math.min(...p.answers.map((a) => a.verdict?.score ?? 1))),
			color: 'var(--color-down)',
			width: 1.25
		},
		{
			label: 'mean, 5-pass rolling',
			values: done.map((_, i) => {
				const w = done.slice(Math.max(0, i - 4), i + 1);
				return w.reduce((s, p) => s + (p.meanScore ?? 0), 0) / w.length;
			}),
			color: 'var(--color-ink-soft)',
			dashed: true
		}
	]);
	const rewardDots = $derived(
		done.flatMap((p, i) => p.answers.map((a) => [i, a.verdict?.score ?? 0] as [number, number]))
	);

	onMount(() => {
		void runtime.refreshSaved();
		// A new project if nothing is loaded yet, or if the models, prompts or judge changed.
		if ((!runtime.ready || runtime.needsReload) && !runtime.loading && !runtime.running) void runtime.load();
	});

	// ③ and ④: one chart each, with a toggle for what it shows.
	type ChartSpec = { series: Series[]; scatter?: [number, number][]; yMin?: number; yMax?: number; format: (v: number) => string };
	let changeView = $state('size');
	let answerView = $state('win');
	const pctFmt = (v: number) => `${Math.round(v * 100)}%`;
	const changeCharts = $derived<Record<string, { label: string } & ChartSpec>>({
		size: {
			label: 'weight change',
			series: [{ label: 'total ‖ΔW‖', values: done.map((p) => sumNorms(p.layerNorms)), color: 'var(--color-accent)' }],
			yMin: 0,
			format: (v) => v.toFixed(1)
		},
		loss: {
			label: 'loss',
			series: [{ label: 'GRPO loss', values: done.map((p) => (p.skipped ? null : p.loss)), color: 'oklch(0.7 0.13 250)', dots: true }],
			format: (v) => v.toFixed(2)
		},
		drift: {
			label: 'behavior drift (KL)',
			series: [{ label: `KL from original (β ${config.klBeta})`, values: done.map((p) => p.kl), color: 'var(--color-down)', dots: true }],
			yMin: 0,
			format: (v) => v.toFixed(2)
		}
	});
	/** The answer check matching the selected preset's goal, if any. */
	const presetCheck = $derived(TASKS.find((t) => t.constitution === config.constitution)?.rule);
	/** Share of a pass's answers passing a heuristic check. */
	const share = (p: Pass, test: (t: string, f: boolean) => boolean) =>
		p.answers.filter((a) => test(a.text, a.stopped)).length / p.answers.length;
	const answerCharts = $derived<Record<string, { label: string } & ChartSpec>>({
		win: {
			label: 'Win rate',
			series: [{ label: 'win rate vs original model', values: done.map((p) => p.vsOriginal), color: 'var(--color-up)', dots: true }],
			yMin: 0,
			yMax: 1,
			format: pctFmt
		},
		reward: {
			label: 'Reward',
			series: rewardSeries,
			scatter: rewardDots,
			yMin: 0,
			yMax: 1,
			format: pctFmt
		},
		length: {
			label: 'Length',
			series: [{ label: 'mean tokens per answer', values: done.map((p) => p.answers.reduce((s, a) => s + a.tokens.length, 0) / p.answers.length), color: 'var(--color-accent)' }],
			yMin: 0,
			format: (v) => v.toFixed(0)
		},
		// A check only makes sense against a preset's goal: show just that preset's one.
		...Object.fromEntries(
			CHECKS.filter((h) => h.id === presetCheck).map((h) => [
				h.id,
				{
					label: h.label,
					series: [{ label: `share of answers: ${h.label.toLowerCase()}`, values: done.map((p) => share(p, h.test)), color: 'var(--color-accent)', dots: true }],
					yMin: 0,
					yMax: 1,
					format: pctFmt
				}
			])
		)
	});

	function selectPass(i: number | null) {
		runtime.viewing = i;
	}

</script>

<div class="space-y-5">
	<!-- One fixed place for the models: open with progress while loading, folded once loaded. -->
	{#if runtime.loads.length || runtime.loading}
		<details class="text-xs text-gray-500" open={runtime.loading}>
			<summary class="cursor-pointer select-none hover:text-gray-300">
				{runtime.loading ? runtime.status || 'loading models…' : 'models loaded'}
			</summary>
			<ul class="mt-2 space-y-1.5 pl-4">
				{#each runtime.loads as load (load.label)}
					<li class="max-w-xl space-y-1">
						<div class="flex justify-between gap-3">
							<span class="text-gray-300">{load.label}</span>
							<span class="tabular-nums">
								{load.done
									? load.downloaded
										? `downloaded ${formatBytes(load.downloaded)}`
										: 'from browser cache'
									: `${load.downloaded ? 'downloading' : 'reading from cache'} · ${formatBytes(load.loaded)} / ${formatBytes(load.total)}`}
							</span>
						</div>
						{#if !load.done}
							<div class="h-1 overflow-hidden rounded-full bg-muted">
								<div class="h-full bg-accent transition-[width]" style:width="{(100 * load.loaded) / Math.max(1, load.total)}%"></div>
							</div>
						{/if}
					</li>
				{/each}
			</ul>
		</details>
	{:else if runtime.gpu === 'missing'}
		<p class="text-xs text-danger">This browser doesn't support WebGPU. Try a recent Chrome, Edge, or Safari 26+.</p>
	{:else if !runtime.ready}
		<button type="button" class="btn-primary" onclick={() => runtime.load()}>load models</button>
	{/if}

	<!-- Controls -->
	<div class="card flex flex-wrap items-center gap-2 p-2">
		<button type="button" class="btn-primary" disabled={!runtime.ready || runtime.running} onclick={() => runtime.runPass()}>
			step
		</button>
		{#if runtime.playing}
			<button type="button" class="btn" onclick={() => runtime.pause()}>pause after this pass</button>
		{:else}
			<button type="button" class="btn" disabled={!runtime.ready || runtime.running} onclick={() => runtime.play()}>play</button>
		{/if}

		<div class="mx-2 h-5 w-px bg-rule"></div>
		<button type="button" class="btn px-2" disabled={passIdx <= 0} onclick={() => selectPass(passIdx - 1)} aria-label="Previous pass">‹</button>
		<span class="min-w-24 text-center text-[0.8rem] tabular-nums">
			{#if runtime.passes.length}pass {passIdx + 1} / {runtime.passes.length}{:else}no passes yet{/if}
		</span>
		<button
			type="button"
			class="btn px-2"
			disabled={runtime.viewing === null || passIdx >= runtime.passes.length - 1}
			onclick={() => selectPass(passIdx + 1 >= runtime.passes.length - 1 ? null : passIdx + 1)}
			aria-label="Next pass">›</button
		>
		{#if runtime.viewing !== null}
			<button type="button" class="btn" onclick={() => selectPass(null)}>follow latest</button>
		{/if}

	</div>
	{#if runtime.error}<p class="-mt-3 text-[0.8rem] text-danger">{runtime.error}</p>{/if}
	<!-- The pass, as a graph: full bleed, breaking out of the page column -->
	<section class="relative left-1/2 w-screen -translate-x-1/2 border-y border-gray-800/60 bg-gray-950 px-8 py-3 sm:px-16">
		<PassGraph
			{pass}
			groupSize={config.groupSize}
			maxNew={config.maxNew}
			judgeLabel={getModel(config.judgeId).label}
			totalChange={sumNorms(pass?.layerNorms)}
			{tokenScale}
			modelLabel={getModel(config.policyId).label}
			method={config.mode === 'lora' ? 'LoRA' : 'Whole model'}
			name={runtime.sessionName || '…'}
			createdAt={runtime.sessionCreatedAt}
			judgePromptText={config.constitution}
			judgeSummary={judgeSummary(config.constitution)}
			onrename={(n) => runtime.renameSession(n)}
		/>
	</section>

	<!-- What changed: answers, then the model, then where in the network. -->
	{#if done.length}
		<h2 class="pt-2 text-xs text-gray-400">what changed</h2>
		<div class="space-y-4">
			<div class="grid gap-4 lg:grid-cols-2">
			{#each [['answers', answerCharts, 'answer'], ['model changes', changeCharts, 'change']] as const as [title, charts, which] (which)}
				{@const view = which === 'change' ? changeView : answerView}
				{@const chart = charts[view] ?? Object.values(charts)[0]}
				<section class="card space-y-2 p-4">
					<div class="space-y-1.5">
						<h3 class="text-xs text-gray-400">{title}</h3>
						<div class="flex flex-wrap gap-1 text-[0.72rem]">
							{#each Object.entries(charts) as [key, c] (key)}
								<button
									type="button"
									class={['rounded-token px-2 py-0.5', view === key ? 'bg-hover font-medium' : 'text-ink-soft hover:text-ink']}
									onclick={() => (which === 'change' ? (changeView = key) : (answerView = key))}>{c.label}</button
								>
							{/each}
						</div>
					</div>
					<LineChart series={chart.series} scatter={chart.scatter} height={180} yMin={chart.yMin} yMax={chart.yMax} format={chart.format} />
				</section>
			{/each}
			</div>

			<section class="card space-y-2 p-4">
				<div class="flex items-baseline justify-between gap-2">
					<h3 class="text-xs text-gray-400">where the weights moved</h3>
					<div class="flex gap-1 text-[0.72rem]">
						<button type="button" class={['rounded-token px-2 py-0.5', gridView === 'total' ? 'bg-hover font-medium' : 'text-ink-soft']} onclick={() => (gridView = 'total')}>since start</button>
						<button type="button" class={['rounded-token px-2 py-0.5', gridView === 'pass' ? 'bg-hover font-medium' : 'text-ink-soft']} onclick={() => (gridView = 'pass')}>this pass</button>
					</div>
				</div>
				{#if pass?.layerNorms}
					<LayerGrid norms={pass.layerNorms} prev={prevPass?.layerNorms} view={gridView} />
				{/if}
			</section>
		</div>
	{/if}
</div>
