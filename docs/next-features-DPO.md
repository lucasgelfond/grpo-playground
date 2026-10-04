# Next feature: DPO (and choosing the algorithm)

How DPO would be added next to GRPO, with the algorithm picked on the first page. Estimate: **about 2.5 days**, of which 1 is the shared refactor and 1–1.5 is DPO. PPO builds on the same refactor; see [next-features-PPO.md](next-features-PPO.md).

## 1. Shared refactor (~1 day)

**Pick the algorithm on the models page.**
- `src/lib/state/config.svelte.ts`: add `algorithm: 'grpo' | 'dpo' | 'ppo'` and DPO's `dpoBeta` (default 0.1). `load()` already merges new defaults into saved settings.
- `src/routes/+page.svelte`: an "algorithm" radio group above "what gets trained", styled like the LoRA / whole-model choice, with a one-line description of each.

**The header follows the algorithm.**
- `src/routes/+layout.svelte`: the title and `<h1>` become "GRPO, in the browser", "DPO, in the browser", etc.
- `STEPS` becomes derived from the algorithm, so PPO can insert its "reward model" step later.

**A pass per algorithm.**
- `src/lib/state/runtime.svelte.ts`: `Pass` becomes a tagged union (`kind: 'grpo' | 'dpo'`). `recordPass` stores it unchanged.
- `src/lib/engine/engine.worker.ts`: split `runPass` into a shared sampling step (prefill, the original model's answer, sampling 5 answers) and an algorithm-specific step (`runGrpoPass`, `runDpoPass`).

**A graph per algorithm.**
- `src/lib/components/PassGraph.svelte`: pull out the column layout, edges and phase logic into a `PassGraphFrame`. `GrpoGraph` keeps today's columns, and `DpoGraph` swaps the last two.
- The train page's "loss" chart label becomes per-algorithm.

## 2. DPO: online, from the judge's preferences (~1–1.5 days)

Online DPO keeps the whole current pipeline: sample answers, have the judge compare them in pairs, update. The difference is the update. DPO trains directly on (chosen, rejected) pairs instead of turning win rates into group-normalized advantages.

**Pairs per pass.**
- Use the judge matches each pass already makes. Each `Match.p` is a soft preference.
- Keep the most confident pairs, ranked by |p − 0.5|, and drop near-ties (like GRPO's near-tie skip).
- The original model's answer stays out of training, as now.

**Loss** (a new `dpoLoss` jit entry next to `grpoLoss` in `src/lib/models/llama.ts`):
- Stack chosen and rejected answers on the batch axis.
- Sequence log-probs: `s = Σ_t log π(token_t)` over each answer, using the existing completion log-probs and masks.
- Margin: `m = β·((s_c − ref_c) − (s_r − ref_r))`, where `ref` is the original model's log-prob. That's the existing LoRA-off pass, or the frozen copy in whole-model mode.
- Soft labels (conservative DPO): `loss = −[p·logσ(m) + (1−p)·logσ(−m)]`, with `logσ(x) = min(x, 0) − log1p(exp(−|x|))` for stability.
- Same `valueAndGrad` + AdamW + microbatching as GRPO. `PolicyTrainer.updateDpo` reuses the batching and optimizer code factored out of `update`.

**Graph:** prompt → policy → answers → judge → **pairs** (chosen ▸ rejected chips with the judge's confidence) → **dpo update** (each pair's margin before → after).

**Charts:**
- Chosen and rejected log-probs over passes. This catches DPO's known failure where both go down.
- Mean margin, and preference accuracy (share of pairs with margin > 0).
- Win rate vs. the original model, kept from GRPO.
- Per-token Δlog p tint, and the per-layer weight-change grid, unchanged.

**Risks:**
- DPO tends to drift toward longer answers. Watch the length chart, and keep the 128-token cap.
- Few pairs per pass make updates noisy. Take 2–4 pairs per pass.
- Add a tiny-model check of `dpoLoss` and its gradient against PyTorch, like the LFM2 and Qwen3 checks.
