<script lang="ts">
	import { formatBytes, type ModelDef } from '$lib/models/registry';
	import { config } from '$lib/state/config.svelte';
	import { cached } from '$lib/state/cached.svelte';

	const { m, role }: { m: ModelDef; role: 'policy' | 'judge' } = $props();
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
			{#if cached[m.id]}<span class="text-up">downloaded</span> · {/if}{formatBytes(m.downloadBytes)}
		</span>
	</div>
</div>
