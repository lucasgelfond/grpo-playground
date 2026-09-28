<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { Attachment } from 'svelte/attachments';

	const {
		width,
		fraction = 0,
		live = false,
		highlight = false,
		dim = false,
		onclick,
		children
	}: {
		width: number;
		/** 0-1 progress, drawn as an accent ring that closes when done. */
		fraction?: number;
		live?: boolean;
		/** Extra-strong ring, e.g. the answer the judge liked best. */
		highlight?: boolean;
		dim?: boolean;
		onclick?: () => void;
		children: Snippet;
	} = $props();

	const ANGLE = '145deg';
	const SOFTNESS = 14;
	const done = $derived(fraction >= 1);
	const fillMask = $derived(
		done ? 'none' : `linear-gradient(${ANGLE}, #000 ${fraction * 100 - SOFTNESS}%, transparent ${fraction * 100 + SOFTNESS}%)`
	);
	const reduced = typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;

	// While a node is working its border breathes: fades up and back down with
	// no restart, so the loop is seamless.
	const pulse: Attachment<HTMLElement> = (element) => {
		if (reduced) return;
		const animation = element.animate([{ opacity: 0.15 }, { opacity: 0.85 }], {
			duration: 900,
			iterations: Infinity,
			direction: 'alternate',
			easing: 'ease-in-out'
		});
		return () => animation.cancel();
	};
</script>

<!--
	After Lacuna's processing-graph StageNode: always a panel over a rule, never a
	fill, with progress drawn as an accent ring that runs around the border.
-->
<svelte:element
	this={onclick ? 'button' : 'div'}
	type={onclick ? 'button' : undefined}
	role={onclick ? undefined : 'group'}
	{onclick}
	class={[
		'relative block rounded-token hairline bg-panel px-3 py-2 text-left transition-opacity',
		'border-rule-strong',
		dim && 'opacity-55',
		onclick && 'hover:bg-hover'
	]}
	style:width="{width}px"
>
	{#if fraction > 0 || live}
		<div class="pointer-events-none absolute -inset-px" style:mask-image={fillMask} aria-hidden="true">
			<div class="absolute inset-0 rounded-token border-2 border-accent/35"></div>
			<div class={['absolute inset-0 rounded-token border-accent', highlight ? 'border-[3px]' : 'border-2']}></div>
		</div>
	{/if}
	{#if live}
		<div class="pointer-events-none absolute -inset-px rounded-token border-2 border-accent" aria-hidden="true" {@attach pulse}></div>
	{/if}
	{@render children()}
</svelte:element>
