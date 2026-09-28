<script lang="ts">
	import { onMount, tick } from 'svelte';

	import { getModel, MODELS, type ChatTurn } from '$lib/models/registry';
	import type { SavedMeta } from '$lib/saved';
	import { config } from '$lib/state/config.svelte';
	import { runtime } from '$lib/state/runtime.svelte';

	type Exchange = { user: string; replies: [string, string] };
	type Option = { value: string; label: string; baseId: string };

	let left = $state('');
	let right = $state('');
	let exchanges = $state<Exchange[]>([]);
	let input = $state('');
	let busy = $state(false);
	let error = $state('');
	let scroller: HTMLElement;

	const hasSession = $derived(runtime.ready && runtime.passes.length > 0);
	// Every model you can chat with: the originals, the live session, and saved ones.
	const options = $derived<Option[]>([
		...MODELS.filter((m) => m.id === 'smollm2-135m').map((m) => ({
			value: `original:${m.id}`,
			label: `${m.label} (original)`,
			baseId: m.id
		})),
		...(hasSession
			? [{ value: 'session', label: `${runtime.sessionName} (training now)`, baseId: runtime.policyDef!.id }]
			: []),
		...runtime.saved
			.filter((m) => !(hasSession && m.id === runtime.sessionId))
			.map((m) => ({ value: `saved:${m.id}`, label: `${m.name} · ${m.passes} passes`, baseId: m.baseId }))
	]);
	const pick = (v: string) => options.find((o) => o.value === v);
	const mismatch = $derived(!!pick(left) && !!pick(right) && pick(left)!.baseId !== pick(right)!.baseId);

	onMount(async () => {
		await runtime.refreshSaved();
		const base = runtime.policyDef?.id ?? config.policyId;
		left ||= `original:${base}`;
		right ||= hasSession ? 'session' : runtime.saved[0] ? `saved:${runtime.saved[0].id}` : `original:${base}`;
	});

	/** 'original:<id>' | 'session' | 'saved:<id>' -> runtime chat source. */
	const source = (v: string) => (v.startsWith('original:') ? 'original' : v === 'session' ? 'session' : v.slice(6));

	async function send() {
		const text = input.trim();
		if (!text || busy || !pick(left) || !pick(right) || mismatch) return;
		busy = true;
		error = '';
		input = '';
		exchanges.push({ user: text, replies: ['', ''] });
		const ex = exchanges[exchanges.length - 1];
		const history = (k: 0 | 1): ChatTurn[] =>
			exchanges.flatMap((e) => [
				{ role: 'user' as const, content: e.user },
				...(e === ex ? [] : [{ role: 'assistant' as const, content: e.replies[k] }])
			]);
		try {
			await runtime.chatPair([source(left), source(right)], pick(left)!.baseId, [history(0), history(1)], (k, t) => (ex.replies[k] = t));
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
			await tick();
			scroller?.scrollTo({ top: scroller.scrollHeight, behavior: 'smooth' });
		}
	}

	async function remove(m: SavedMeta) {
		await runtime.deleteSaved(m);
		if (right === `saved:${m.id}`) right = '';
		if (left === `saved:${m.id}`) left = '';
	}
</script>

<div class="grid gap-5 lg:grid-cols-[1fr_280px]">
	<div class="space-y-3">
		<div class="grid grid-cols-2 gap-3">
			{#each [0, 1] as k (k)}
				<select
					class="field py-1.5 font-medium"
					value={k === 0 ? left : right}
					onchange={(e) => {
						if (k === 0) left = e.currentTarget.value;
						else right = e.currentTarget.value;
						exchanges = [];
					}}
				>
					{#each options as o (o.value)}<option value={o.value}>{o.label}</option>{/each}
				</select>
			{/each}
		</div>
		{#if mismatch}
			<p class="text-[0.8rem] text-danger">Pick two models with the same base to compare them.</p>
		{/if}

		<div bind:this={scroller} class="max-h-[62vh] space-y-4 overflow-y-auto">
			{#if exchanges.length === 0}
				<p class="py-10 text-center text-ink-soft">Ask something to see both models answer it.</p>
			{/if}
			{#each exchanges as ex, i (i)}
				<div class="space-y-2">
					<div class="ml-auto w-fit max-w-[80%] rounded-token bg-hover px-3 py-2">{ex.user}</div>
					<div class="grid grid-cols-2 gap-3">
						{#each ex.replies as reply, k (k)}
							<div class="card min-h-12 p-3 text-[0.85rem] leading-relaxed whitespace-pre-wrap">
								{reply || (busy && i === exchanges.length - 1 ? '…' : '')}
							</div>
						{/each}
					</div>
				</div>
			{/each}
		</div>

		{#if error}<p class="text-[0.8rem] text-danger">{error}</p>{/if}
		<form
			class="flex gap-2"
			onsubmit={(e) => {
				e.preventDefault();
				send();
			}}
		>
			<input class="field" placeholder="ask both models…" bind:value={input} />
			<button type="submit" class="btn-primary" disabled={busy || !input.trim() || mismatch}>{busy ? 'answering…' : 'send'}</button>
		</form>
	</div>

	<aside class="space-y-2">
		<h2 class="label">saved in this browser</h2>
		{#if runtime.saved.length === 0}
			<p class="text-[0.8rem] text-ink-soft">Models save here automatically as they train.</p>
		{/if}
		{#each runtime.saved as m (m.id)}
			<div class="card flex items-center justify-between gap-2 px-3 py-2">
				<div class="min-w-0">
					<div class="truncate font-terminal text-[0.8rem]">{m.name}</div>
					<div class="text-[0.7rem] text-ink-soft tabular-nums">
						{getModel(m.baseId).params} · {m.passes} passes ·
						{new Date(m.createdAt).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}
					</div>
				</div>
				<button type="button" class="text-[0.75rem] text-ink-soft hover:text-danger" onclick={() => remove(m)}>delete</button>
			</div>
		{/each}
	</aside>
</div>
