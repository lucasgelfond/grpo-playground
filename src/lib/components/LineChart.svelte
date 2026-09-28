<script lang="ts">
	export type Series = {
		label: string;
		values: (number | null | undefined)[];
		/** CSS color, e.g. var(--color-accent). */
		color: string;
		width?: number;
		dots?: boolean;
		dashed?: boolean;
	};

	const {
		series,
		height = 160,
		yMin,
		yMax,
		format = (v: number) => v.toFixed(2),
		scatter = [],
		showX = true
	}: {
		series: Series[];
		height?: number;
		yMin?: number;
		yMax?: number;
		format?: (v: number) => string;
		/** Extra points drawn as faint dots: [x index, y]. */
		scatter?: [number, number][];
		/** Pass labels on the x axis; off for all but the bottom of a stack sharing one axis. */
		showX?: boolean;
	} = $props();

	let width = $state(400);
	const L = 40;
	const R = 8;
	const T = 8;
	const B = $derived(showX ? 20 : 6);

	const n = $derived(Math.max(1, ...series.map((s) => s.values.length)));
	const finite = $derived(
		[...series.flatMap((s) => s.values), ...scatter.map((p) => p[1])].filter(
			(v): v is number => typeof v === 'number' && Number.isFinite(v)
		)
	);
	const lo = $derived(yMin ?? (finite.length ? Math.min(...finite) : 0));
	const hi = $derived(yMax ?? (finite.length ? Math.max(...finite) : 1));
	const span = $derived(hi - lo || 1);
	const x = (i: number) => L + (n <= 1 ? (width - L - R) / 2 : (i / (n - 1)) * (width - L - R));
	const y = (v: number) => T + (1 - (v - lo) / span) * (height - T - B);

	function path(values: Series['values']) {
		let d = '';
		let pen = false;
		values.forEach((v, i) => {
			if (typeof v !== 'number' || !Number.isFinite(v)) {
				pen = false;
				return;
			}
			d += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
			pen = true;
		});
		return d;
	}
	const ticks = $derived([lo, lo + span / 2, hi]);
</script>

<div bind:clientWidth={width} class="w-full">
	<svg {width} {height} class="block overflow-visible">
		{#each ticks as t, i (i)}
			<line x1={L} x2={width - R} y1={y(t)} y2={y(t)} class="stroke-rule" stroke-width="1" />
			<text x={L - 6} y={y(t)} dy="0.32em" text-anchor="end" class="fill-ink-soft text-[10px] tabular-nums">{format(t)}</text>
		{/each}
		{#if n > 1 && showX}
			<text x={L} y={height - 4} class="fill-ink-soft text-[10px]">pass 1</text>
			<text x={width - R} y={height - 4} text-anchor="end" class="fill-ink-soft text-[10px] tabular-nums">pass {n}</text>
		{/if}
		{#each scatter as [i, v], k (k)}
			<circle cx={x(i)} cy={y(v)} r="2" style:fill="var(--color-ink-soft)" opacity="0.35" />
		{/each}
		{#each series as s (s.label)}
			<path
				d={path(s.values)}
				fill="none"
				style:stroke={s.color}
				stroke-width={s.width ?? 1.75}
				stroke-dasharray={s.dashed ? '4 3' : undefined}
				stroke-linejoin="round"
				stroke-linecap="round"
			/>
			{#if s.dots || n === 1}
				{#each s.values as v, i (i)}
					{#if typeof v === 'number' && Number.isFinite(v)}
						<circle cx={x(i)} cy={y(v)} r="2.5" style:fill={s.color} />
					{/if}
				{/each}
			{/if}
		{/each}
	</svg>
	<div class="mt-1 flex flex-wrap gap-3 pl-10 text-[0.72rem] text-ink-soft">
		{#each series as s (s.label)}
			<span class="flex items-center gap-1.5">
				<span class="inline-block h-0.5 w-3" style:background={s.color}></span>{s.label}
			</span>
		{/each}
	</div>
</div>
