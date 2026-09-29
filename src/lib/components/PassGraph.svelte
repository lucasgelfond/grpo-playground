<script lang="ts">
	import type { Answer, Pass } from '$lib/state/runtime.svelte';
	import GraphNode from './GraphNode.svelte';

	const {
		pass,
		groupSize,
		maxNew,
		judgeLabel,
		totalChange,
		tokenScale,
		modelLabel,
		method,
		name,
		createdAt,
		judgePromptText,
		judgeSummary,
		showReference,
		onrename
	}: {
		pass: Pass | undefined;
		groupSize: number;
		maxNew: number;
		judgeLabel: string;
		/** Sum of ||ΔW|| over all adapted matrices after this pass. */
		totalChange: number | undefined;
		/** Largest |Δlog p| in the pass, so token tints are comparable across answers. */
		tokenScale: number;
		/** e.g. "SmolLM2 135M Instruct", and how it's trained ("LoRA"). */
		modelLabel: string;
		method: string;
		/** The policy model's name (it autosaves under this name after every pass). */
		name: string;
		/** When this model was created (ISO), shown under its name. */
		createdAt: string;
		/** The judge's guidelines, shown when hovering the judge. */
		judgePromptText: string;
		/** The judge prompt in one line, e.g. "Starts with yes or no". */
		judgeSummary: string;
		/** Reserve a row for the original model's answer (GT) from the start. */
		showReference: boolean;
		onrename: (name: string) => void;
	} = $props();

	/** Answer the pointer is over (raised above its neighbors so its popovers show). */
	let hovered = $state<number | null>(null);
	/** Answer clicked open to show its full text over the ones below. */
	let expanded = $state<number | null>(null);

	/** Per-token Δlog p from this pass's update, and the most-changed token each way. */
	function shifts(a: Answer | undefined) {
		if (!a?.logpAfter || !a.logpBefore) return null;
		const d = a.logpAfter.map((v, i) => v - a.logpBefore![i]);
		let up = -1;
		let down = -1;
		d.forEach((x, i) => {
			if (x > 0 && (up < 0 || x > d[up])) up = i;
			if (x < 0 && (down < 0 || x < d[down])) down = i;
		});
		return { d, up, down };
	}

	function tint(x: number) {
		const t = Math.min(1, Math.abs(x) / tokenScale);
		return `color-mix(in oklch, ${x > 0 ? 'var(--color-up)' : 'var(--color-down)'} ${Math.round(t * 60)}%, transparent)`;
	}

	// Fixed geometry, so edges are arithmetic rather than measurement (as in Lacuna's graph.ts).
	const PAD = 16;
	const PROMPT_W = 190;
	const MODEL_W = 180;
	const ANSWER_W = 340;
	const JUDGE_W = 170;
	const RANK_W = 210;
	const UPDATE_W = 180;
	const GAP = 48;
	const ROW = 84;
	// The original model's answer (compare mode) gets an extra row at the bottom.
	const rows = $derived((pass?.answers.length ?? groupSize) + (showReference || pass?.reference ? 1 : 0));
	const G = $derived(pass?.answers.length ?? groupSize);
	const height = $derived(rows * ROW);
	const xPrompt = PAD;
	const xModel = PAD + PROMPT_W + GAP;
	const xAnswers = xModel + MODEL_W + GAP;
	const xJudge = xAnswers + ANSWER_W + GAP;
	const xRank = xJudge + JUDGE_W + GAP;
	const xUpdate = xRank + RANK_W + GAP;
	const width = xUpdate + UPDATE_W + PAD;

	// Shrink the whole graph to fit narrower windows instead of scrolling.
	let available = $state(width);
	const scale = $derived(Math.min(1, available / width));

	/** Answers ordered by the judge's score, best first; unscored ones last. */
	const ranking = $derived(
		(pass?.answers ?? [])
			.map((a, i) => ({ i, score: a.verdict?.score, adv: a.advantage }))
			.concat(pass?.reference ? [{ i: pass.answers.length, score: pass.reference.verdict?.score, adv: undefined }] : [])
			.filter((r) => r.score !== undefined)
			.sort((a, b) => b.score! - a.score! || a.i - b.i)
	);
	const midY = $derived(PAD + height / 2);
	const rowY = (i: number) => PAD + (i + 0.5) * ROW;

	type EdgeState = 'idle' | 'flowing' | 'done';
	const phaseIndex = $derived(
		pass ? { prefill: 0, sampling: 0, judging: 1, updating: 2, done: 3 }[pass.phase] : -1
	);
	const stateFor = (stage: number): EdgeState =>
		phaseIndex === stage ? 'flowing' : phaseIndex > stage ? 'done' : 'idle';

	function curve(x1: number, y1: number, x2: number, y2: number) {
		const mid = (x1 + x2) / 2;
		return `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`;
	}

	const edges = $derived.by(() => {
		const out: { key: string; path: string; x1: number; x2: number; y: number; state: EdgeState; strong?: boolean }[] = [];
		for (let i = 0; i < rows; i++) {
			const y = rowY(i);
			out.push({ key: `p${i}`, path: curve(xModel + MODEL_W, midY, xAnswers, y), x1: xModel + MODEL_W, x2: xAnswers, y: midY, state: pass?.phase === 'prefill' ? 'idle' : stateFor(0) });
			out.push({
				key: `j${i}`,
				path: curve(xAnswers + ANSWER_W, y, xJudge, midY),
				x1: xAnswers + ANSWER_W,
				x2: xJudge,
				y,
				state: stateFor(1),
				strong: pass?.best === i && phaseIndex > 1
			});
		}
		const prefilling = pass?.phase === 'prefill';
		out.push({ key: 'm', path: curve(xPrompt + PROMPT_W, midY, xModel, midY), x1: xPrompt + PROMPT_W, x2: xModel, y: midY, state: !pass ? 'idle' : prefilling ? 'flowing' : 'done' });
		out.push({ key: 'r', path: curve(xJudge + JUDGE_W, midY, xRank, midY), x1: xJudge + JUDGE_W, x2: xRank, y: midY, state: stateFor(1) });
		out.push({ key: 'u', path: curve(xRank + RANK_W, midY, xUpdate, midY), x1: xRank + RANK_W, x2: xUpdate, y: midY, state: stateFor(2) });
		const order = { idle: 0, done: 1, flowing: 2 };
		return out.sort((a, b) => order[a.state] - order[b.state] + (a.strong ? 1 : 0) - (b.strong ? 1 : 0));
	});

	const uid = $props.id();
	const compare = $derived(pass?.matchesTotal !== undefined);
	const judgeFraction = $derived(
		!pass
			? 0
			: compare
				? (pass.matches?.length ?? 0) / Math.max(1, pass.matchesTotal!)
				: pass.answers.filter((a) => a.verdict).length / pass.answers.length
	);
	const sampleFraction = $derived(
		pass ? pass.answers.reduce((s, a) => s + (a.stopped ? 1 : a.tokens.length / maxNew), 0) / pass.answers.length : 0
	);

	/** Answers are lettered A, B, C…; the original model's answer is "GT". */
	const letter = (i: number) => String.fromCharCode(65 + i);
	const nameOf = (i: number) => (i === G ? 'GT' : letter(i));
	/** An answer's matches, from its point of view. */
	function matchesOf(i: number) {
		return (pass?.matches ?? [])
			.filter((m) => m.a === i || m.b === i)
			.map((m) => ({ other: m.a === i ? m.b : m.a, p: m.a === i ? m.p : 1 - m.p }));
	}

	const pushed = $derived(
		pass?.phase === 'done' && !pass.skipped
			? {
					up: pass.answers.flatMap((a, i) => ((a.advantage ?? 0) > 0 ? [i] : [])),
					down: pass.answers.flatMap((a, i) => ((a.advantage ?? 0) < 0 ? [i] : []))
				}
			: null
	);

	function pct(x: number | undefined) {
		return x === undefined ? '–' : `${Math.round(x * 100)}%`;
	}
</script>

<div bind:clientWidth={available} style:height="{(height + PAD * 2) * scale}px">
	<div
		class="relative origin-top-left"
		style:width="{width}px"
		style:margin-left="{Math.max(0, (available - width * scale) / 2)}px"
		style:height="{height + PAD * 2}px"
		style:transform="scale({scale})"
	>
		<svg class="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
			{#each edges as e (e.key)}
				<path
					d={e.path}
					fill="none"
					stroke-width={e.strong ? 2.5 : 1.5}
					stroke-linecap="round"
					class={[
						e.state === 'idle' && 'stroke-rule-strong',
						e.state === 'flowing' && 'stroke-accent/25',
						e.state === 'done' && (e.strong ? 'stroke-accent' : 'stroke-accent/45')
					]}
				/>
				{#if e.state === 'flowing'}
					{@const length = e.x2 - e.x1}
					<linearGradient id="{uid}-{e.key}" gradientUnits="userSpaceOnUse" x1={e.x1} y1={e.y} x2={e.x2} y2={e.y}>
						<stop offset="0.25" style="stop-color: var(--color-accent); stop-opacity: 0" />
						<stop offset="0.5" style="stop-color: var(--color-accent); stop-opacity: 1" />
						<stop offset="0.6" style="stop-color: var(--color-accent); stop-opacity: 0" />
						<animateTransform
							attributeName="gradientTransform"
							type="translate"
							from="{-length * 0.6} 0"
							to="{length * 0.8} 0"
							dur="1.2s"
							repeatCount="indefinite"
						/>
					</linearGradient>
					<path d={e.path} fill="none" stroke="url(#{uid}-{e.key})" stroke-width="2" stroke-linecap="round" />
				{/if}
			{/each}
		</svg>

		<!-- Prompt -->
		<div class="absolute -translate-y-1/2" style:left="{xPrompt}px" style:top="{midY}px">
			<GraphNode width={PROMPT_W} fraction={pass ? 1 : 0}>
				<div class="label">prompt {pass ? `· pass ${pass.index + 1}` : ''}</div>
				<div class="mt-1 text-[0.85rem] leading-snug font-medium">{pass?.prompt ?? 'Press Step to run the first pass'}</div>
			</GraphNode>
		</div>

		<!-- Policy model -->
		<div class="group absolute -translate-y-1/2" style:left="{xModel}px" style:top="{midY}px">
			<GraphNode width={MODEL_W} fraction={pass ? 1 : 0} live={pass?.phase === 'prefill' || pass?.phase === 'updating'}>
				<div class="label">policy model{pass?.phase === 'prefill' ? ' · prefilling' : pass?.phase === 'sampling' ? ' · sampling' : ''}</div>
				<input
					class="mt-1 w-full rounded-[0.25rem] bg-transparent font-terminal text-[0.82rem] font-medium outline-none focus:bg-hover"
					value={name}
					aria-label="Model name"
					onchange={(e) => onrename(e.currentTarget.value.trim() || name)}
				/>
				<div class="mt-0.5 text-[0.72rem] text-ink-soft">{modelLabel}</div>
				<div class="text-[0.72rem] text-ink-soft">{method}</div>
				{#if createdAt}
					<div class="text-[0.68rem] text-ink-soft tabular-nums">
						{new Date(createdAt).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}
					</div>
				{/if}
			</GraphNode>
			{#if pass?.layerNorms}
				<div
					class="pointer-events-none absolute top-full left-0 z-10 mt-2 hidden w-56 card p-2.5 text-[0.75rem] tabular-nums shadow-lg group-hover:block"
				>
					<div class="label mb-1">After pass {pass.index + 1}</div>
					<div class="flex justify-between"><span class="text-ink-soft">size of change ‖ΔW‖</span>{totalChange?.toFixed(2)}</div>
					<div class="flex justify-between"><span class="text-ink-soft">loss</span>{pass.loss?.toFixed(3) ?? '–'}</div>
					<div class="flex justify-between"><span class="text-ink-soft">drift (KL from original)</span>{pass.kl?.toFixed(3) ?? '–'}</div>
				</div>
			{/if}
		</div>

		<!-- Answers -->
		{#each Array.from({ length: rows }) as _, i (i)}
			{@const isRef = i === G}
			{@const a = isRef ? pass?.reference : pass?.answers[i]}
			{@const isBest = pass?.best === i && phaseIndex > 1}
			<div
				class={['absolute -translate-y-1/2', (expanded === i || hovered === i) && 'z-30']}
				style:left="{xAnswers}px"
				style:top="{rowY(i)}px"
				role="button"
				tabindex="0"
				class:cursor-pointer={!!a?.text}
				onmouseenter={() => (hovered = i)}
				onmouseleave={() => (hovered = null)}
				onclick={() => (expanded = expanded === i ? null : i)}
				onkeydown={(e) => e.key === 'Escape' && (expanded = null)}
			>
				{#if expanded === i && a?.text}
					<!-- The full answer, opened over the answers below this one. -->
					<div
						class="absolute top-0 left-0 z-20 max-h-[28rem] overflow-y-auto rounded border border-blue-500 bg-gray-900 p-3 text-[0.8rem] leading-relaxed whitespace-pre-wrap shadow-xl"
						style:width="{ANSWER_W}px"
					>
						<div class="mb-1.5 flex justify-between text-[0.72rem] text-gray-500">
							<span>{nameOf(i)} · {a.tokens.length} tokens</span><span>click to close</span>
						</div>
						{a.text}
					</div>
				{/if}
				<GraphNode
					width={ANSWER_W}
					fraction={a ? (a.verdict ? 1 : a.stopped ? 1 : Math.min(1, a.tokens.length / maxNew)) : 0}
					live={pass?.phase === 'sampling' && !a?.stopped}
					highlight={isBest}
					dim={isRef}
				>
					<div class="flex items-center gap-2 text-[0.72rem] text-ink-soft">
						<span class="font-medium tabular-nums">{isRef ? 'GT · original model, not trained' : letter(i)}</span>
						{#if isBest}<span class="rounded bg-blue-900/50 px-1 text-[10px] text-blue-300">chosen</span>{/if}
						<span class="ml-auto tabular-nums">
							{#if a?.advantage !== undefined && !pass?.skipped}
								<span class={a.advantage > 0 ? 'text-up' : a.advantage < 0 ? 'text-down' : ''}>
									{a.advantage > 0 ? '▲' : a.advantage < 0 ? '▼' : '•'} {Math.abs(a.advantage).toFixed(2)}
								</span>
							{/if}
							{a?.tokens.length ?? 0} tok
						</span>
					</div>
					{@const sh = isRef ? null : shifts(a)}
					<div class="mt-0.5 line-clamp-2 h-[2.5em] text-[0.8rem] leading-[1.25em]">
						{#if sh && a}
							{#each a.pieces as piece, t (t)}<span
									class={['rounded-[2px]', (t === sh.up || t === sh.down) && 'font-semibold ring-1 ring-current']}
									style:background={tint(sh.d[t])}
									title="p {Math.exp(a.logpBefore![t]).toFixed(3)} → {Math.exp(a.logpAfter![t]).toFixed(3)}">{piece}</span
								>{/each}
						{:else}
							{a?.text || (pass?.phase === 'sampling' ? '…' : '')}
						{/if}
					</div>
					<div class="mt-1 flex items-center gap-2">
						<div class="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
							<div
								class="h-full rounded-full bg-gray-200 transition-[width] duration-500"
								style:width="{(a?.verdict?.score ?? 0) * 100}%"
							></div>
						</div>
						<span class="group/pct relative w-9 text-right text-[0.72rem] tabular-nums">
							{pct(a?.verdict?.score)}
							{#if matchesOf(i).length}
								<!-- How this answer did in each of its matches. -->
								<span class="absolute top-full right-0 z-40 mt-1 hidden w-52 rounded border border-gray-700 bg-gray-900 p-2 text-left shadow-xl group-hover/pct:block">
									{#each matchesOf(i) as m (m.other)}
										<span class="grid grid-cols-[3.4rem_1.9rem_1fr_2.2rem] items-center gap-1.5">
											<span class={m.p >= 0.5 ? 'text-up' : 'text-down'}>{m.p >= 0.5 ? 'beat' : 'lost to'}</span>
											<span>{nameOf(m.other)}</span>
											<span class="h-1.5 overflow-hidden rounded-full bg-muted">
												<span class={['block h-full rounded-full', m.p >= 0.5 ? 'bg-up' : 'bg-down']} style:width="{Math.max(m.p, 1 - m.p) * 100}%"></span>
											</span>
											<span class="text-right">{pct(Math.max(m.p, 1 - m.p))}</span>
										</span>
									{/each}
								</span>
							{/if}
						</span>
					</div>
				</GraphNode>
			</div>
		{/each}

		<!-- Judge -->
		<div class="group absolute -translate-y-1/2" style:left="{xJudge}px" style:top="{midY}px">
			<div
				class="pointer-events-none absolute top-full left-0 z-20 mt-2 hidden max-h-[22rem] w-80 overflow-hidden rounded border border-gray-700 bg-gray-900 p-3 text-[11px] leading-relaxed whitespace-pre-wrap text-gray-300 shadow-xl group-hover:block"
			>{judgePromptText}</div>
			<GraphNode width={JUDGE_W} fraction={judgeFraction} live={pass?.phase === 'judging' || pass?.phase === 'prefill'}>
				<div class="label">judge{pass?.phase === 'prefill' ? ' · prefilling' : ''}</div>
				<div class="mt-1 text-[0.82rem] font-medium">{judgeLabel}</div>
				{#if judgeSummary}<div class="mt-0.5 line-clamp-2 text-[0.72rem] text-ink-soft">“{judgeSummary}”</div>{/if}
				{#if pass?.phase === 'judging' && compare}
					<div class="mt-1.5 h-1.5 overflow-hidden rounded-full bg-muted">
						<div class="h-full rounded-full bg-gray-200 transition-[width] duration-500" style:width="{judgeFraction * 100}%"></div>
					</div>
				{/if}
			</GraphNode>
		</div>

		<!-- Ranking -->
		<div class="absolute -translate-y-1/2" style:left="{xRank}px" style:top="{midY}px">
			<GraphNode width={RANK_W} fraction={pass ? ranking.length / pass.answers.length : 0} live={pass?.phase === 'judging'}>
				<div class="label">ranking</div>
				<ol class="mt-1.5 space-y-0.5 text-[0.75rem] tabular-nums">
					{#each ranking as r, rank (r.i)}
						<li class="grid grid-cols-[1.1rem_1.9rem_1fr_2.2rem] items-center gap-1.5">
							<span class="text-ink-soft">{rank + 1}</span>
							<span class={r.i === G ? 'text-ink-soft' : ''}>{nameOf(r.i)}</span>
							<div class="h-1.5 overflow-hidden rounded-full bg-muted">
								<div
									class={['h-full rounded-full', r.adv === undefined ? 'bg-gray-200' : r.adv > 0 ? 'bg-up' : r.adv < 0 ? 'bg-down' : 'bg-ink-soft']}
									style:width="{r.score! * 100}%"
								></div>
							</div>
							<span class="text-right">{pct(r.score)}</span>
						</li>
					{/each}
				</ol>
			</GraphNode>
		</div>

		<!-- Update -->
		<div class="absolute -translate-y-1/2" style:left="{xUpdate}px" style:top="{midY}px">
			<GraphNode width={UPDATE_W} fraction={phaseIndex >= 3 ? 1 : 0} live={pass?.phase === 'updating'}>
				<div class="label">grpo update</div>
				<div class="mt-1 text-[0.8rem] tabular-nums">
					{#if pushed}
						<div><span class="text-up">↑</span> {pushed.up.map(letter).join(' ')}</div>
						<div><span class="text-down">↓</span> {pushed.down.map(letter).join(' ')}</div>
					{:else if pass?.phase === 'done'}
						<span class="text-ink-soft">skipped: all scores tied</span>
					{:else if pass?.phase === 'updating'}
						<span class="text-ink-soft">updating…</span>
					{/if}
				</div>
			</GraphNode>
		</div>
	</div>
</div>
