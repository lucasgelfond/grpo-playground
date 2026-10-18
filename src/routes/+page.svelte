<script lang="ts">
	import { onMount } from 'svelte';

	import ModelRow from '$lib/components/ModelRow.svelte';
	import { formatBytes, getModel, TRAINEES } from '$lib/models/registry';
	import { estimateCost, type Mode } from '$lib/rl/trainer';
	import { config } from '$lib/state/config.svelte';
	import { checkCached } from '$lib/state/cached.svelte';
	import { runtime } from '$lib/state/runtime.svelte';


	onMount(() => {
		deviceMemory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
		void checkCached(TRAINEES.map((m) => m.id));
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
				gpuBytes: cost.memoryBytes + (config.reward === 'ai' ? judge.downloadBytes : 0),
				seconds: (MEASURED_SECONDS[mode] * cost.totalParams) / 134.5e6,
				sizeBytes: cost.trainable * 4
			};
		})
	);

	const fmtParams = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : `${(n / 1e6).toFixed(1)}M`);
</script>

<div class="max-w-3xl space-y-7">
	<fieldset class="space-y-2">
		<legend class="label mb-2">fine-tune</legend>
		{#each TRAINEES as m (m.id)}<ModelRow {m} role="policy" />{/each}
	</fieldset>

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
		<a href="/reward" class="btn-primary">next: reward →</a>
	</div>
</div>
