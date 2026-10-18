# Vendored jax-js

`jax-js-jax-0.1.26-29a2d19.tgz` is `@jax-js/jax` built from
[`lucasgelfond/jax-js#grpo-playground-prs`](https://github.com/lucasgelfond/jax-js/tree/grpo-playground-prs)
(commit `29a2d19`), which is upstream `main` plus two open PRs:

- [ekzhang/jax-js#236](https://github.com/ekzhang/jax-js/pull/236): routines (sort, scatter, ...) get their view inputs materialized when jit-compiled
- [ekzhang/jax-js#237](https://github.com/ekzhang/jax-js/pull/237): partial-eval tracers are freed iteratively, not recursively. Computing SmolLM2 360M's gradient overflowed the Web Worker's call stack ("Maximum call stack size exceeded").

`package.json` points `@jax-js/jax` here, and a pnpm override makes `@jax-js/optax` and `@jax-js/loaders` use the same copy.

**TODO:** once both PRs are merged and released, switch back to the npm `@jax-js/jax` and delete this folder.

To rebuild: in a jax-js checkout of that branch, `pnpm install && pnpm build && pnpm pack`, copy the `.tgz` here, update the file name in `package.json` (dependency and override), then `pnpm install`.
