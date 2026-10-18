<script lang="ts">
	import { formatBytes, type ModelDef } from '$lib/models/registry';
	import { config } from '$lib/state/config.svelte';
	import { downloads } from '$lib/state/downloads.svelte';

	const { m, role }: { m: ModelDef; role: 'policy' | 'judge' } = $props();

	const d = $derived(downloads[m.id]);
</script>

<div class="max-w-sm space-y-1 text-[0.85rem]">
	<div class="flex items-baseline justify-between gap-3">
		<label class="flex cursor-pointer items-center gap-2">
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
