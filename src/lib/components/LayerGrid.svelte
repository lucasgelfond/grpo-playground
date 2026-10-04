<script lang="ts">
	import { layerTargets, LORA_TARGETS, type Target } from '$lib/models/llama';
	import type { ModelConfig } from '$lib/models/registry';

	const {
		cfg,
		norms,
		prev,
		view
	}: {
		/** The trained model: which projections each layer has (LFM2 mixes attention and conv layers). */
		cfg: ModelConfig;
		norms: number[][];
		prev: number[][] | undefined;
		/** 'total': distance from the original weights; 'pass': how much this pass moved each matrix. */
		view: 'total' | 'pass';
	} = $props();

	const has = $derived(Array.from({ length: cfg.layers }, (_, l) => new Set(layerTargets(cfg, l))));
	const rows = $derived(LORA_TARGETS.map((target, t) => ({ target, t })).filter(({ target }) => has.some((s) => s.has(target))));
	const values = $derived(
		view === 'total' || !prev ? norms : norms.map((row, l) => row.map((v, t) => Math.abs(v - prev[l][t])))
	);
	const max = $derived(Math.max(1e-12, ...values.flat()));
	const rowTotals = $derived(values.map((r) => r.reduce((s, v) => s + v, 0)));
	const maxRow = $derived(Math.max(1e-12, ...rowTotals));
	const LABEL: Record<Target, string> = { q: 'q', k: 'k', v: 'v', in: 'in', o: 'o', gate: 'gate', up: 'up', down: 'down' };
</script>

<!-- Transposed so it stays short: one row per projection, one column per layer. -->
<div class="overflow-x-auto">
	<table class="border-separate border-spacing-[2px] text-[0.68rem] tabular-nums">
		<tbody>
			{#each rows as { target, t } (target)}
				<tr>
					<td class="pr-1.5 text-right text-ink-soft">{LABEL[target]}</td>
					{#each values as row, l (l)}
						<td class="p-0">
							{#if has[l]?.has(target)}<div
								class="h-3 w-3 rounded-[2px]"
								style:background="color-mix(in oklch, var(--color-accent) {Math.round((row[t] / max) * 100)}%, var(--color-muted))"
								title="layer {l} {target}: {row[t].toExponential(2)}"
							></div>{:else}<div class="h-3 w-3"></div>{/if}
						</td>
					{/each}
				</tr>
			{/each}
			<tr>
				<td class="pr-1.5 text-right text-ink-soft">all</td>
				{#each rowTotals as total, l (l)}
					<td class="p-0 align-bottom">
						<div class="flex h-6 w-3 items-end" title="layer {l}: {total.toExponential(2)}">
							<div class="w-full rounded-[1px] bg-accent/70" style:height="{(total / maxRow) * 100}%"></div>
						</div>
					</td>
				{/each}
			</tr>
			<tr>
				<td class="pr-1.5 text-right text-ink-soft">layer</td>
				{#each values as _, l (l)}
					<td class="p-0 text-center text-ink-soft">{l % 5 === 0 ? l : ''}</td>
				{/each}
			</tr>
		</tbody>
	</table>
</div>
