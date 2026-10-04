# Next feature: PPO

How PPO would be added next to GRPO. It builds on the shared refactor in [next-features-DPO.md](next-features-DPO.md): `config.algorithm`, steps and title that follow the algorithm, and a pass type and graph per algorithm. If DPO isn't built first, add about a day for that refactor.

Estimate: **4–6 days**. That's 2 days for the reward-model step, 2–3 for PPO training, and half a day for a numerics harness to build first.

## What already helps

- `generate` returns each sampled token's log-prob (`src/lib/rl/generate.ts`). These are PPO's `logp_old`.
- `runSequence` in `src/lib/models/llama.ts` already computes the final hidden states. Nothing returns them yet.
- The judge's `Match.p` is a soft preference (`src/lib/rl/judge.ts`). That's what a Bradley–Terry reward model trains on.

## 1. A new step: "reward model" (`/reward`, ~2 days)

PPO inserts this step between "prompts & judge" and "train".

1. **Collect AI feedback.** For each prompt, sample 4 answers from the original model, which gives 6 pairs. Judge every pair in both orders with `compareAnswers`. Store `{prompt, chosen, rejected, p}` in PGlite so collection is resumable.
2. **Train a reward model.**
   - **Model:** the policy's frozen base, a reward LoRA (rank 8), and a scalar head `{w: [D], b: []}` read at the last real token.
   - **Loss:** soft Bradley–Terry, `−[p·logσ(r_c−r_r) + (1−p)·logσ(r_r−r_c)]`.
   - **Training:** hold out 20% of pairs. AdamW at 1e-4, minibatches of 4–8 pairs, a few epochs, early stopping on held-out accuracy.
3. **Visualize**, in the train page's graph and card style:
   - **Collection:** prompt → original model → 4 answers → judge → pairs, showing each pick and its confidence.
   - **Training curves:** train and held-out loss, and held-out agreement with the judge.
   - **Agreement:** a scatter of reward-model margin vs. judge preference.
   - **Ranking:** answers ranked by reward-model score.
   - **Weights:** the per-layer weight-change grid for the reward adapter.

New code:
- `hiddenStates(W, lora, ids, pad)` and a `scalarHead` in `llama.ts`.
- `RewardTrainer` in `src/lib/rl/reward.ts`.
- Collection and training methods in the worker.
- A `/reward` route.

## 2. PPO training (~2–3 days)

**Graph:** prompt → policy → answers → **reward model** (score per answer) → **critic** (value estimate under each token) → **advantages** (tokens tinted by advantage) → **ppo update** (epochs, clip fraction, approximate KL).

**Each pass:**
1. Sample the answers with their log-probs, which become `logp_old`.
2. Score each answer with the reward model. Per-token rewards are `r_t = −β·KL_t`, plus the reward-model score on the last token.
3. Compute GAE on the CPU in a new `src/lib/rl/ppo.ts`: γ = 1, λ = 0.95. Whiten the advantages, and returns are `G_t = A_t + V_t`.
4. Run `ppoLoss` (jit) for each epoch:
   - Ratio: `ρ = exp(logp − logp_old)`.
   - Policy loss: `−min(ρA, clip(ρ, 1±ε)·A)`.
   - Value loss: `vfCoef·(V − G)²`.
   - Aux outputs: clip fraction and approximate KL.
5. Default to 2 epochs on one minibatch.

**Critic:** a value head on the policy's own hidden states (shared trunk, initialized to zero). It's the cheapest option. A separate value LoRA initialized from the reward model is better, but costs another forward and backward pass.

**Charts:**
- Reward-model score of the policy vs. the original model.
- Value loss and explained variance.
- Clip fraction.
- KL from the original model.

## Memory

- **Unload the judge once the reward model is trained.** That frees 3–4 GB. From then on, measure "vs. original" with the reward model instead of judge matches.
- **PPO is LoRA-only.** The policy's 360M weights in f32 are about 1.4 GB. Three adapters (policy, reward, value) plus Adam state come to tens of MB.
- **Update the setup screen's estimate.** Extend `estimateCost` per algorithm.

## Risks, and how to reduce them

- **Numerics.** This is the first multi-epoch update, so the probability ratio ρ ≠ 1 and the clip branches actually run. Build `scripts/parity/` first: a tiny random model and a PyTorch reference for the Bradley–Terry loss, `ppoLoss`, GAE and their gradients. Test both clip branches. The same approach was used for LFM2 and Qwen3.
- **Noise.** 4–8 samples per pass make whitened advantages noisy, so accumulate 2 prompts per update. The reward model gets about 120 pairs and will overfit, so use a low rank and early stopping.
- **Time.** Each update is a reward-model forward, a critic forward, a reference forward, and epochs × backward passes: roughly 3–4× a GRPO update. Show an ETA, and default answers to 64 tokens. Collection takes minutes, so it has to be resumable.

## Cuts that would ship it faster

- A shared-trunk critic initialized to zero.
- Train the reward model on the matches the judge already makes during passes, instead of a separate collection screen.
- One minibatch and no value clipping.
- No reward-model-vs-judge scatter.
