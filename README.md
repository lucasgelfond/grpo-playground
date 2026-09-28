# RLH(AI)F, in the browser

RL fine-tuning of a small language model, entirely in the browser, with [jax-js](https://github.com/ekzhang/jax-js) on WebGPU. Live at [rlplayground.lucasgelfond.online](https://rlplayground.lucasgelfond.online).

A small model (SmolLM2 135M) answers each prompt several times. A bigger local model (Qwen2.5 1.5B) acts as the judge: it chooses between pairs of answers according to a judge prompt you write. GRPO then nudges the small model toward the answers that won. Every stage of every pass is drawn as a graph (prompt → policy model → answers → judge → ranking → update), with charts of what changed.

```bash
pnpm install
pnpm dev          # http://localhost:5173
pnpm run deploy   # build and deploy to Cloudflare
```

Needs a browser with WebGPU and `shader-f16` (recent Chrome/Edge, Safari 26+).

## Steps

1. **Models:** the model to fine-tune, the judge, and LoRA vs. whole-model training.
2. **Prompts & judge:** the prompts the model practices on, and the judge prompt that decides which answers are better. Presets pair the two, e.g. "yes or no first".
3. **Train:** step or play GRPO passes.
4. **Evaluate:** chat with any two models side by side: the original, the one training now, or saved ones.

## How it works

- `src/lib/models/llama.ts`: a Llama-style decoder (RMSNorm, RoPE, SwiGLU, GQA, tied embeddings; Qwen2's qkv bias) in jax-js. It includes batched KV-cache sampling, a prefix-cached forward pass for the judge, and the GRPO loss with a KL penalty under `jit(valueAndGrad(...))`. Logits match Hugging Face transformers.
- `src/lib/rl/judge.ts`: the judge compares answers in pairs, asking "which follows the guidelines better, A or B?", and reads only the A/B logits. The shared part of its prompt is prefilled once per question and cached across passes. Each answer's reward is how often it's picked. Each pass also includes the original model's greedy answer ("GT"), so "win rate vs. original" measures real progress.
- `src/lib/rl/trainer.ts`: GRPO with group-normalized advantages, AdamW, grad clipping, and a KL penalty toward the original model. Trains LoRA adapters (rank 16) or the whole model.
- `src/lib/models/weights.ts`: streams safetensors one tensor at a time with HTTP Range requests, converts BF16 on the fly, and caches it in OPFS.
- `src/lib/db/`: PGlite (Postgres in a worker, stored in OPFS) records projects, their models, prompts (`JUDGE` / `EVALUATION`) and every pass. Trained adapters autosave to OPFS after each pass.

## Hosting

- **Site:** static SvelteKit build served by a Cloudflare Worker (`wrangler.toml`).
- **Weights:** mirrored in the R2 bucket `rl-playground-models`, served from `models.lucasgelfond.online`. Paths follow Hugging Face repo names. Each `model.safetensors` is split into 256 MB parts (`model.safetensors.part00`, …) with a `model.manifest.json`, because R2 uploads through wrangler are size-limited. The loader maps byte ranges onto parts.

## jax-js patch

`patches/@jax-js__jax.patch` (applied by pnpm) fixes two jit compiler issues in `@jax-js/jax@0.1.25`:

1. A view feeding a routine (the scatter in `takeAlongAxis`'s gradient) wasn't materialized, so `jit(grad(...))` threw.
2. Matmul inputs with reuse were materialized through casts, copying f16 weights to f32 on every call. That made the f16 judge 10–40× slower.

See `TODO.md` for how PPO and DPO would fit in.
