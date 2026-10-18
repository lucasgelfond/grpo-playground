<script lang="ts">
	import { onMount } from 'svelte';

	import ModelRow from '$lib/components/ModelRow.svelte';
	import { getModel } from '$lib/models/registry';
	import { PRESETS } from '$lib/presets';
	import { compileReward, type RewardKind } from '$lib/rl/rewards';
	import { applyPreset, config, TASK_MAX } from '$lib/state/config.svelte';
	import { checkCached } from '$lib/state/cached.svelte';

	const KINDS: { kind: RewardKind; title: string; name: string }[] = [
		{ kind: 'human', title: 'Human feedback', name: 'RLHF' },
		{ kind: 'ai', title: 'AI feedback', name: 'RLAIF' },
		{ kind: 'code', title: 'Code', name: 'RLVR' }
	];
	// Choosing between answers needs a 1.5B+ judge; the 0.5B nearly always picks the first answer.
	const judges = ['qwen3-1.7b', 'qwen2.5-1.5b'].map(getModel);
	onMount(() => void checkCached(judges.map((m) => m.id)));

	const ofKind = (kind: RewardKind) => PRESETS.filter((p) => p.reward === kind);
	const codePresets = ofKind('code');
	const codeIndex = $derived(codePresets.findIndex((p) => p.code === config.rewardCode));

	function pickKind(kind: RewardKind) {
		if (config.reward === kind) return;
		applyPreset(config, ofKind(kind)[0]);
	}
	function flip(step: number) {
		const i = codeIndex < 0 ? 0 : (codeIndex + step + codePresets.length) % codePresets.length;
		applyPreset(config, codePresets[i]);
	}
	function custom() {
		config.rewardCode = `// Score an answer: return a number from 0 to 1, or true / false.\nreturn answer.length < 200;`;
		config.check = null;
		config.task = '';
	}

	// Try the reward code on an answer as you type.
	let sample = $state('Sure! Rainbows form when sunlight passes through raindrops.');
	const trial = $derived.by(() => {
		try {
			return { score: compileReward(config.rewardCode)(sample, config.prompts[0] ?? '') };
		} catch (e) {
			return { error: e instanceof Error ? e.message : String(e) };
		}
	});
</script>

<div class="max-w-3xl space-y-7">
	<p class="max-w-2xl text-xs leading-relaxed text-gray-400">
		Each pass, the model answers one prompt {config.groupSize} times. Something scores those answers, and the update makes the
		higher-scoring ones more likely. Where do the scores come from?
	</p>

	<div class="grid gap-3 sm:grid-cols-3">
		{#each KINDS as k (k.kind)}
			<button
				type="button"
				class={[
					'card flex items-baseline justify-between gap-2 p-3 text-left transition-colors',
					config.reward === k.kind ? 'border-accent bg-hover' : 'hover:bg-hover'
				]}
				onclick={() => pickKind(k.kind)}
			>
				<span class="font-medium">{k.title}</span>
				<span class="font-terminal text-[0.7rem] text-ink-soft">{k.name}</span>
			</button>
		{/each}
	</div>

	<label class="block space-y-2">
		<span class="label flex justify-between">
			<span>task</span>
			<span class="text-ink-soft tabular-nums">{config.task.length}/{TASK_MAX}</span>
		</span>
		<input class="field" maxlength={TASK_MAX} bind:value={config.task} placeholder="What do you want to accomplish?" />
	</label>

	{#if config.reward === 'human'}
		<p class="text-[0.85rem] leading-relaxed text-gray-300">
			After each forward pass, click your favorite option. You should see changes after 10-15 passes of consistent picks.
		</p>
	{:else if config.reward === 'ai'}
		<fieldset class="space-y-2">
			<legend class="label mb-2">judge model</legend>
			{#each judges as m (m.id)}<ModelRow {m} role="judge" />{/each}
		</fieldset>
		<section class="space-y-2">
			<h2 class="label">judge prompt</h2>
			<textarea
				class="field min-h-[10rem] resize-y p-3 leading-relaxed"
				bind:value={config.constitution}
				spellcheck="true"
				placeholder="What should the judge reward? e.g. Reward answers that are short and friendly."
			></textarea>
		</section>
	{:else}
		<section class="space-y-2">
			<div class="flex h-7 items-center justify-between gap-3">
				<h2 class="label">reward code</h2>
				<div class="flex items-center gap-1 text-xs">
					<button type="button" class="btn px-2" aria-label="Previous example" onclick={() => flip(-1)}>‹</button>
					<span class="w-52 truncate text-center whitespace-nowrap text-ink-soft">
						{codeIndex >= 0 ? codePresets[codeIndex].label.toLowerCase() : 'custom'}
					</span>
					<button type="button" class="btn px-2" aria-label="Next example" onclick={() => flip(1)}>›</button>
					<button type="button" class={['btn ml-2', codeIndex < 0 && 'bg-hover']} onclick={custom}>custom</button>
				</div>
			</div>
			<div class="font-terminal text-[0.78rem] text-ink-soft">function (answer, prompt) {'{'}</div>
			<textarea
				class="field min-h-[7rem] resize-y p-3 font-terminal text-[0.8rem] leading-relaxed"
				bind:value={config.rewardCode}
				spellcheck="false"
				oninput={() => (config.check = null)}
			></textarea>
			<div class="font-terminal text-[0.78rem] text-ink-soft">{'}'}</div>
			<label class="block space-y-1 text-xs text-gray-400">
				<span>try it on an answer</span>
				<input class="field" bind:value={sample} />
			</label>
			<p class="text-[0.8rem]">
				{#if 'error' in trial}<span class="text-danger">{trial.error}</span>
				{:else}reward: <span class="tabular-nums">{trial.score.toFixed(2)}</span>{/if}
			</p>
		</section>
	{/if}

	<div class="flex items-center justify-between">
		<a href="/" class="text-[0.85rem] text-ink-soft hover:text-ink">← back</a>
		<a href="/prompts" class="btn-primary">next: prompts →</a>
	</div>
</div>
