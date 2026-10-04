<script lang="ts">
	import { onMount } from 'svelte';

	import { formatBytes, getModel, type ModelDef } from '$lib/models/registry';
	import { estimateCost, type Mode } from '$lib/rl/trainer';
	import { config } from '$lib/state/config.svelte';
	import { checkCached, downloads } from '$lib/state/downloads.svelte';
	import { runtime } from '$lib/state/runtime.svelte';

	const pick = (ids: string[]) => ids.map(getModel);
	const trainees = pick(['lfm2-350m', 'smollm2-135m', 'smollm2-360m']);
	// Choosing between answers needs a 1.5B+ judge; the 0.5B nearly always picks the first answer.
	const judges = pick(['qwen3-1.7b', 'qwen2.5-1.5b']);

	onMount(() => {
		deviceMemory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
		void checkCached([...trainees, ...judges].map((m) => m.id));
	});
	// Browsers only expose a rounded RAM figure, capped at 8 GB (Chrome); no GPU memory.
	let deviceMemory = $state<number | undefined>();

	const policy = $derived(getModel(config.policyId));
	const judge = $derived(getModel(config.judgeId));
	const fullBlocked = $derived(policy.fullFinetune === 'no');

	// Update times measured for SmolLM2 135M on an Apple M5; scaled by parameter count.
	const MEASURED_SECONDS: Record<Mode, number> = { lora: 4.5, full: 7 };
	const methods = $derived(
		(['lora', 'full'] as Mode[]).map((mode) => {
			const cost = estimateCost(policy.config, mode, config.loraRank);
			return {
				mode,
				...cost,
				gpuBytes: cost.memoryBytes + judge.downloadBytes,
				seconds: (MEASURED_SECONDS[mode] * cost.totalParams) / 134.5e6,
				sizeBytes: cost.trainable * 4
			};
		})
	);

	const fmtParams = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : `${(n / 1e6).toFixed(1)}M`);
</script>

{#snippet modelRow(m: ModelDef, role: 'policy' | 'judge')}
	{@const d = downloads[m.id]}
	<div class="max-w-sm space-y-1 text-[0.85rem]">
		<div class="flex items-baseline justify-between gap-3">
			<label class="flex cursor-pointer items-center gap-2">
				<!-- bind:group doesn't work inside a snippet, so set the choice by hand. -->
				<input
					type="radio"
					name={role}
					value={m.id}
					checked={(role === 'policy' ? config.policyId : config.judgeId) === m.id}
					onchange={() => (role === 'policy' ? (config.policyId = m.id) : (config.judgeId = m.id))}
				/>
				{m.label}
			</label>
			<span class="text-xs whitespace-nowrap text-gray-500 tabular-nums">
				{#if d?.error}<span class="text-danger">download failed</span>
				{:else if d?.done}<span class="text-up">downloaded</span> · {formatBytes(m.downloadBytes)}
				{:else if d?.queued}queued · {formatBytes(m.downloadBytes)}
				{:else if d}{formatBytes(d.loaded)} / {formatBytes(m.downloadBytes)}
				{:else}{formatBytes(m.downloadBytes)}{/if}
			</span>
		</div>
		{#if d && !d.done && !d.error && !d.queued}
			<div class="h-1 overflow-hidden rounded-full bg-muted">
				<div class="h-full bg-accent transition-[width]" style:width="{(100 * d.loaded) / Math.max(1, d.total)}%"></div>
			</div>
		{/if}
		{#if d?.error}<p class="text-xs text-danger">{d.error}</p>{/if}
	</div>
{/snippet}

<div class="max-w-3xl space-y-7">
	<section class="grid gap-6 sm:grid-cols-2">
		<fieldset class="space-y-2">
			<legend class="label mb-2">fine-tune</legend>
			{#each trainees as m (m.id)}{@render modelRow(m, 'policy')}{/each}
		</fieldset>
		<fieldset class="space-y-2">
			<legend class="label mb-2">judge</legend>
			{#each judges as m (m.id)}{@render modelRow(m, 'judge')}{/each}
		</fieldset>
	</section>

	<section class="space-y-2">
		<h2 class="label">what gets trained</h2>
		<div class="card overflow-hidden">
			<table class="w-full table-fixed text-[0.82rem] tabular-nums">
				<colgroup>
					<col class="w-[34%]" />
					<col class="w-[33%]" />
					<col class="w-[33%]" />
				</colgroup>
				<thead>
					<tr class="hairline-b border-rule">
						<th></th>
						{#each methods as c (c.mode)}
							{@const disabled = c.mode === 'full' && fullBlocked}
							<th class="px-3 py-2 text-left">
								<label class={['flex items-center gap-2 font-medium', disabled && 'opacity-50']}>
									<input type="radio" name="mode" value={c.mode} bind:group={config.mode} {disabled} />
									{c.mode === 'lora' ? 'lora' : 'whole model'}
								</label>
							</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					<tr class="hairline-b border-rule">
						<td class="px-3 py-1.5 text-ink-soft">trainable parameters</td>
						{#each methods as c (c.mode)}
							<td class="px-3 py-1.5">{fmtParams(c.trainable)} <span class="text-ink-soft">({(c.fraction * 100).toFixed(c.fraction < 0.1 ? 1 : 0)}%)</span></td>
						{/each}
					</tr>
					<tr class="hairline-b border-rule">
						<td class="px-3 py-1.5 text-ink-soft">
							gpu memory
							{#if deviceMemory}<span class="text-[0.72rem]">(you have {deviceMemory >= 8 ? '>8' : `~${deviceMemory}`} GB)</span>{/if}
						</td>
						{#each methods as c (c.mode)}
							<td class="px-3 py-1.5">~{formatBytes(c.gpuBytes)}</td>
						{/each}
					</tr>
					<tr class="hairline-b border-rule">
						<td class="px-3 py-1.5 text-ink-soft">update time per pass</td>
						{#each methods as c (c.mode)}
							<td class="px-3 py-1.5">~{c.seconds.toFixed(0)} s</td>
						{/each}
					</tr>
					<tr>
						<td class="px-3 py-1.5 text-ink-soft">size</td>
						{#each methods as c (c.mode)}
							<td class="px-3 py-1.5">{formatBytes(c.sizeBytes)}</td>
						{/each}
					</tr>
				</tbody>
			</table>
		</div>
	</section>

	{#if runtime.needsReload}
		<p class="text-[0.8rem] text-ink-soft">Changed settings reset the loaded session when training starts again.</p>
	{/if}

	<div class="flex justify-end">
		<a href="/prompts" class="btn-primary">next: prompts & judge →</a>
	</div>
</div>
