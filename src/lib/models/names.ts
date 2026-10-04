import { layerKind, layerTargets, type Target } from './llama';
import type { ModelConfig } from './registry';

/**
 * Hugging Face tensor names for each architecture, mapped to paths in our
 * `Weights` tree. Loading, exporting and saving all go through here.
 */

const LLAMA_MODULES: Record<Target, string> = {
	q: 'self_attn.q_proj',
	k: 'self_attn.k_proj',
	v: 'self_attn.v_proj',
	in: '',
	o: 'self_attn.o_proj',
	gate: 'mlp.gate_proj',
	up: 'mlp.up_proj',
	down: 'mlp.down_proj'
};

const LFM2_MODULES: Record<Target, string> = {
	q: 'self_attn.q_proj',
	k: 'self_attn.k_proj',
	v: 'self_attn.v_proj',
	in: 'conv.in_proj',
	o: 'self_attn.out_proj',
	gate: 'feed_forward.w1',
	up: 'feed_forward.w3',
	down: 'feed_forward.w2'
};

/** HF module of projection `t` in layer `i`, e.g. 'self_attn.q_proj'. */
export function hfModule(cfg: ModelConfig, i: number, t: Target): string {
	if (cfg.arch !== 'lfm2') return LLAMA_MODULES[t];
	return t === 'o' && layerKind(cfg, i) === 'conv' ? 'conv.out_proj' : LFM2_MODULES[t];
}

/** Every tensor of layer `i`: [HF name after `model.layers.{i}.`, path within our Layer]. */
export function layerTensorNames(cfg: ModelConfig, i: number): [string, string[]][] {
	const lfm2 = cfg.arch === 'lfm2';
	const attention = layerKind(cfg, i) === 'attention';
	const out: [string, string[]][] = [
		[lfm2 ? 'operator_norm.weight' : 'input_layernorm.weight', ['inNorm']],
		[lfm2 ? 'ffn_norm.weight' : 'post_attention_layernorm.weight', ['postNorm']]
	];
	for (const t of layerTargets(cfg, i)) {
		const mod = hfModule(cfg, i, t);
		out.push([`${mod}.weight`, [t, 'w']]);
		if (cfg.qkvBias && (t === 'q' || t === 'k' || t === 'v')) out.push([`${mod}.bias`, [t, 'b']]);
	}
	if (attention && cfg.qkNorm) {
		out.push([lfm2 ? 'self_attn.q_layernorm.weight' : 'self_attn.q_norm.weight', ['qNorm']]);
		out.push([lfm2 ? 'self_attn.k_layernorm.weight' : 'self_attn.k_norm.weight', ['kNorm']]);
	}
	if (!attention) out.push(['conv.conv.weight', ['conv']]);
	return out;
}

/** Tensors outside the layers. The output head is tied to the embeddings. */
export function globalTensorNames(cfg: ModelConfig): [string, string[]][] {
	return [
		['model.embed_tokens.weight', ['embed']],
		[cfg.arch === 'lfm2' ? 'model.embedding_norm.weight' : 'model.norm.weight', ['norm']]
	];
}

/** Every tensor in the checkpoint: full HF name -> path in `Weights`. */
export function tensorPaths(cfg: ModelConfig): Map<string, string[]> {
	const map = new Map(globalTensorNames(cfg));
	for (let i = 0; i < cfg.layers; i++) {
		for (const [name, path] of layerTensorNames(cfg, i)) map.set(`model.layers.${i}.${name}`, ['layers', String(i), ...path]);
	}
	return map;
}
