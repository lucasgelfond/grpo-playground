# TODO: PPO and DPO

GRPO is built. This note covers how to add the other two algorithms to this codebase, plus a few more that would fit.

Everything below reuses what GRPO already has:

- `compileModel` in `src/lib/models/llama.ts`: batched sampling, completion log-probs, and `extend` for prefix-cached judging.
- The prefix-cached pairwise judge in `src/lib/rl/judge.ts` (`compareAnswers`).
- The trainer in `src/lib/rl/trainer.ts`: LoRA or full fine-tuning, AdamW, the KL term, and per-token log-probs before and after each update.
- The reference model for free: the same weights with LoRA off (`trainer.original()`).

The Models page shows GRPO as the only method (`config.method`).

**Effort and fit:**

- **DPO:** about half a day. Online DPO keeps the current screen: sample answers, have the judge choose between pairs, then update on winner/loser pairs. It's the natural second method.
- **PPO:** 1–2 days and two stages. First train a reward model on the judge's choices, then run PPO, with the value model trained alongside the policy. It shows one answer at a time, so it's less of the "model choosing among outputs" view. Keep it for later.

## Shared building block: a scalar head

PPO needs two new model outputs, and both are "LM backbone + scalar head":

- **Reward model:** one score per answer, read at the last token.
- **Value model (critic):** one score per token.

Build them the same way:

1. Share the frozen base weights of the model being trained. Each head gets its own LoRA adapter plus `head: { w: [hidden], b: [] }`.
2. Add a `hiddenStates(W, lora, ids, pad)` jit entry point that returns the final-norm hidden states `[B, T, D]`. `runSequence` already computes them; `lastLogits` and `completionLogprobs` just don't return them.
3. Reward: `h[:, -1] · w + b` → `[B]`. Value: `h[:, P-1 : T-1] · w + b` → `[B, L]`, aligned with the completion tokens like `completionLogprobs`.

Memory stays small: base 135M in f32 plus three adapters (policy, reward, value) and two heads.

## PPO (classic RLHF / RLAIF)

Steps in the UI: Models → Prompts → Judge prompt → **AI feedback → Reward model → Train**.

### 1. AI feedback (new page)

- For each prompt, sample N answers from the original model (N = 4 gives 6 pairs).
- Have the judge choose between pairs with `compareAnswers`, both orders.
- Store `{ prompt, chosen, rejected, p }` records. Keep the soft label `p` rather than rounding it to 0/1.
- Show a table of pairs with the judge's pick and confidence.
- About 20 prompts × 6 pairs ≈ 120 pairs is plenty for a demo.
- Save the dataset to OPFS so it survives reloads.

### 2. Reward model (new page)

- Bradley–Terry loss on the pairs: `-[p·log σ(r_c − r_r) + (1−p)·log σ(r_r − r_c)]`.
- Hold out ~20% of pairs.
- Train with `jit(valueAndGrad(...))` over `{ lora, head }` with AdamW, lr ≈ 1e-4, a few epochs of minibatches of 8 pairs.
- Show a loss curve, held-out accuracy against the judge, and a scatter of reward-model score margin vs. judge p.
- When done, the judge can be unloaded (`tree.dispose(judgeWeights)`) to free ~3 GB.

### 3. PPO training (the Train page, PPO mode)

Per pass:

1. Sample K answers per prompt (K can be 1–4; PPO doesn't need a group).
2. Score each with the reward model → `R`.
3. Per-token rewards: `r_t = −β·KL_t` for every token, plus `R` added at the last token. `KL_t` uses the reference log-probs that `trainer.update` already computes (`logpRef`).
4. Values `V_t` from the value head.
5. GAE: `δ_t = r_t + γV_{t+1} − V_t`, `A_t = Σ (γλ)^k δ_{t+k}`, with γ = 1 and λ = 0.95. Returns `G_t = A_t + V_t`. Whiten A over the batch.
6. Run `ppoEpochs` (2–4) minibatch updates on the same samples:
   - Policy loss: `−min(ρ_t A_t, clip(ρ_t, 1±ε) A_t)` with `ρ_t = exp(logp − logp_old)` and ε = 0.2. `logp_old` is the log-prob at sampling time, which `generate` already returns.
   - Value loss: `(V_t − G_t)²`, optionally clipped.
   - These are the first multi-epoch updates. GRPO does one step, so its ratio is exactly 1.
7. Initialize the value head from the reward model's head and adapter (standard practice), then train it jointly.

Visualization (the per-token pieces are the point):

- The value curve `V_t` drawn under each answer's tokens, like a sparkline.
- Per-token advantage `A_t` as token tint (replacing today's Δlog p tint, or as a toggle).
- Per-token KL.
- Reward-model score next to each answer.
- Value-model loss and explained variance over passes.

Code:

- `src/lib/rl/ppo.ts`: GAE and the PPO step.
- `llama.ts`: add a `ppoStep` jit entry next to `trainStep`.
- `runtime.svelte.ts`: branch `runPass` on `config.method`.

## DPO (Direct Preference Optimization)

Steps: Models → Prompts → Judge prompt → **Train**. No reward model, no value model.

Loss per pair (chosen c, rejected r), with a fixed reference π_ref (LoRA off):

```
m = β · [(log π(c) − log π_ref(c)) − (log π(r) − log π_ref(r))]
loss = −log σ(m)
```

`log π(·)` is the sum over completion tokens of `completionLogprobs`. β ≈ 0.1.

Two ways to run it:

- **Offline:** train on the AI feedback dataset from PPO step 1. It's the same data, with DPO instead of a reward model plus RL.
- **Online:** closer to what the GRPO view shows. Each pass, sample G answers, have the judge choose between pairs (reuse `compareAnswers`), then take the most confident pairs, or best vs. worst by win rate, as (c, r) and do one DPO step.

Code:

- `llama.ts`: a `dpoStep` jit entry. Two completions per pair, stacked on the batch axis; ids and targets are built the same way `trainer.update` builds them.
- The reference log-probs come from the same pass `trainer.update` already runs for KL.

Visualization:

- Pairs with the judge's pick.
- The implicit reward margin `m` per pair, before and after the update.
- Chosen and rejected log-prob curves over passes. A known DPO failure mode is both going down.

## Other algorithms that would fit

- **RLOO (REINFORCE leave-one-out):** GRPO without dividing by the group std. The baseline for each answer is the mean of the *other* G−1 rewards. It's a two-line change in `trainer.update` and often more stable with small groups.
- **Dr. GRPO / DAPO tweaks:** drop the std normalization and per-sequence length normalization, and use asymmetric clipping ("clip-higher"). They matter when answers vary a lot in length, which they do here.
- **Best-of-N / rejection-sampling fine-tuning (expert iteration):** sample G answers, keep the judge's favorite, and do plain SFT on it. The simplest baseline, which trains very stably. With a choosing judge it's "pick the winner, imitate it".
- **KTO:** learns from unpaired good/bad labels (yes/no scoring mode) instead of pairs.
- **SimPO / ORPO:** DPO variants without a reference model (SimPO uses length-normalized log-probs). They're cheaper, but they lose the KL anchor that keeps a 135M model coherent.

## Known issues worth fixing alongside

- **0.5B judge:** Qwen2.5 0.5B picks "A" about 85% of the time regardless of order. Choosing needs the 1.5B judge.
- **Formal rules:** the 1.5B judge is near chance on purely formal rules ("ends with ?"). Semantic criteria work (92% pairwise agreement).
- **Slow passes on memory-starved machines:** the 1.5B judge in f16 (3.1 GB) swaps on a busy 16 GB Mac. An int8 judge with dequantization fused into the matmul (possible now that casts fuse) would halve that.
