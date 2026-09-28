<script lang="ts">
	import { onMount } from 'svelte';

	import { formatBytes, getModel, MODELS, type ModelDef } from '$lib/models/registry';
	import { downloadWeights, isCached, persistStorage } from '$lib/models/weights';
	import { estimateCost, type Mode } from '$lib/rl/trainer';
	import { config } from '$lib/state/config.svelte';
	import { runtime } from '$lib/state/runtime.svelte';

	// Just the small model for now: fast passes and room beside the judge.
	const trainees = MODELS.filter((m) => m.id === 'smollm2-135m');
	// Choosing between answers needs the 1.5B judge; the 0.5B nearly always picks the first answer.
	const judges = MODELS.filter((m) => m.judge && m.id !== 'qwen2.5-0.5b');

	let cached = $state<Record<string, boolean | undefined>>({});
	let downloading = $state<Record<string, number | undefined>>({});
	let downloadError = $state('');

	async function download(m: ModelDef) {
		downloadError = '';
		downloading[m.id] = 0;
		await persistStorage();
		try {
			await downloadWeights(m, (p) => (downloading[m.id] = p.loaded / p.total));
			cached[m.id] = true;
		} catch (e) {
			downloadError = `${m.label}: ${e instanceof Error ? e.message : e}`;
		} finally {
			downloading[m.id] = undefined;
		}
	}
	// Browsers only expose a rounded RAM figure, capped at 8 GB (Chrome); no GPU memory.
	let deviceMemory = $state<number | undefined>();
	onMount(async () => {
		deviceMemory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
		for (const m of MODELS) cached[m.id] = await isCached(m);
	});

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

	const label = (m: ModelDef) => `${m.label} (${m.params} / ${formatBytes(m.downloadBytes)})`;
	const fmtParams = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : `${(n / 1e6).toFixed(1)}M`);
</script>

{#snippet modelRadio(m: ModelDef, name: string, group: 'policyId' | 'judgeId')}
	<div class="flex items-center gap-2 text-[0.85rem] whitespace-nowrap">
		<label
			class={['flex items-center gap-2', !cached[m.id] && 'cursor-not-allowed text-ink-soft']}
			title={cached[m.id] ? undefined : 'Download this model first'}
		>
			<input type="radio" {name} value={m.id} bind:group={config[group]} disabled={!cached[m.id]} />
			{label(m)}
		</label>
		{#if downloading[m.id] !== undefined}
			<span class="text-[0.75rem] text-ink-soft tabular-nums">{Math.round(downloading[m.id]! * 100)}%</span>
		{:else if cached[m.id] === false}
			<button type="button" class="text-[0.75rem] text-ink-soft underline hover:text-ink" onclick={() => download(m)}>download</button>
		{/if}
	</div>
{/snippet}

<div class="max-w-3xl space-y-7">
	<section class="grid gap-6 sm:grid-cols-2">
		<fieldset class="space-y-2">
			<legend class="label mb-2">fine-tune</legend>
			{#each trainees as m (m.id)}{@render modelRadio(m, 'policy', 'policyId')}{/each}
		</fieldset>
		<fieldset class="space-y-2">
			<legend class="label mb-2">judge</legend>
			{#each judges as m (m.id)}{@render modelRadio(m, 'judge', 'judgeId')}{/each}
		</fieldset>
		{#if downloadError}<p class="text-[0.78rem] text-danger sm:col-span-2">{downloadError}</p>{/if}
	</section>

	<section class="space-y-2">
		<h2 class="label">algorithm</h2>
		<label class="flex w-fit items-center gap-2 text-[0.85rem]" title="PPO and DPO are planned: see TODO.md">
			<input type="radio" checked />
			grpo
		</label>
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
