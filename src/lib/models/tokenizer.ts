import { cachedFetch, tokenizers } from '@jax-js/loaders';

import { modelUrl, type ModelDef, type TokenizerKind } from './registry';

/**
 * Byte-level BPE tokenizers loaded from Hugging Face `tokenizer.json` files.
 *
 * jax-js's `BpeEncoding` ranks merges by token id, which matches merge order
 * for both SmolLM2 and Qwen2 vocabularies. What differs is pre-tokenization:
 *
 * - SmolLM2: split every digit into its own piece, then the GPT-2 regex.
 * - Qwen2: NFC-normalize, then a single cl100k-style regex.
 */

// GPT-2 pre-tokenizer pattern (HF ByteLevel with use_regex=true).
const GPT2_PATTERN = String.raw`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`;

/**
 * A regex whose `matchAll` first isolates each digit (HF `Digits` with
 * individual_digits) and then applies the GPT-2 pattern to the remaining text.
 * BpeEncoding only ever calls `string.matchAll(regex)`, so overriding
 * `Symbol.matchAll` is enough to compose the two pre-tokenizers.
 */
class DigitsThenGpt2 extends RegExp {
	constructor() {
		super(GPT2_PATTERN, 'gu');
	}

	[Symbol.matchAll](str: string): RegExpStringIterator<RegExpMatchArray> {
		return digitsThenGpt2(str) as unknown as RegExpStringIterator<RegExpMatchArray>;
	}
}

function* digitsThenGpt2(str: string): Generator<RegExpMatchArray> {
	for (const part of str.split(/(\p{N})/u)) {
		if (part === '') continue;
		if (/^\p{N}$/u.test(part)) {
			yield [part] as unknown as RegExpMatchArray;
			continue;
		}
		yield* part.matchAll(new RegExp(GPT2_PATTERN, 'gu'));
	}
}

class NfcBpeEncoding extends tokenizers.BpeEncoding {
	_beforeEncode(text: string): string {
		return text.normalize('NFC');
	}
}

type TokenizerJson = {
	added_tokens: { id: number; content: string; special: boolean }[];
	pre_tokenizer: { pretokenizers: { type: string; pattern?: { Regex: string } }[] };
	model: { type: string; vocab: Record<string, number> };
};

export class Tokenizer {
	readonly specialIds: Set<number>;
	readonly #enc: tokenizers.BpeEncoding;

	constructor(data: TokenizerJson, kind: TokenizerKind) {
		if (data.model.type !== 'BPE') throw new Error(`Expected BPE, got ${data.model.type}`);

		const special: Record<string, number> = {};
		this.specialIds = new Set();
		for (const t of data.added_tokens) {
			special[t.content] = t.id;
			if (t.special) this.specialIds.add(t.id);
		}

		const byteDecoder = byteLevelDecoder();
		const encoder = new Map<string, number>();
		for (const [piece, id] of Object.entries(data.model.vocab)) {
			if (this.specialIds.has(id)) continue;
			encoder.set(byteLevelToHex(piece, byteDecoder), id);
		}

		if (kind === 'smollm2') {
			this.#enc = new tokenizers.BpeEncoding(encoder, special, new DigitsThenGpt2());
		} else {
			const split = data.pre_tokenizer.pretokenizers.find((p) => p.type === 'Split');
			if (!split?.pattern) throw new Error('Expected a Split pre-tokenizer for Qwen2');
			// JS has no inline `(?i:...)`; it only wraps contractions, so a global
			// `i` flag is equivalent for this pattern.
			const pattern = split.pattern.Regex.replace(/^\(\?i:([^)]*)\)/, '(?:$1)');
			this.#enc = new NfcBpeEncoding(encoder, special, new RegExp(pattern, 'giu'));
		}
	}

	/** Encode text, treating special-token strings like `<|im_start|>` as specials. */
	encode(text: string): number[] {
		return this.#enc.encodeWithSpecialTokens(text);
	}

	decode(ids: number[]): string {
		return this.#enc.decode(ids);
	}

	/** Decode generated ids, dropping special tokens. */
	decodeText(ids: number[]): string {
		return this.decode(ids.filter((id) => !this.specialIds.has(id)));
	}

	/**
	 * Per-token display strings whose concatenation equals `decodeText(ids)`.
	 * Decoding cumulatively keeps multi-byte characters split across tokens intact.
	 */
	pieces(ids: number[]): string[] {
		const out: string[] = [];
		let prev = '';
		for (let i = 0; i < ids.length; i++) {
			const text = this.decodeText(ids.slice(0, i + 1)).replace(/�+$/, '');
			out.push(text.startsWith(prev) ? text.slice(prev.length) : '');
			if (text.startsWith(prev)) prev = text;
		}
		return out;
	}
}

export async function loadTokenizer(def: ModelDef): Promise<Tokenizer> {
	const data = await cachedFetch(modelUrl(def, 'tokenizer.json'));
	return new Tokenizer(JSON.parse(new TextDecoder().decode(data)), def.tokenizer);
}

/** GPT-2's reversible byte <-> unicode mapping used by byte-level vocabularies. */
function byteLevelDecoder(): Map<string, number> {
	const bytes: number[] = [];
	for (let i = 33; i <= 126; i++) bytes.push(i);
	for (let i = 161; i <= 172; i++) bytes.push(i);
	for (let i = 174; i <= 255; i++) bytes.push(i);
	const chars = [...bytes];
	let extra = 0;
	for (let b = 0; b < 256; b++) {
		if (bytes.includes(b)) continue;
		bytes.push(b);
		chars.push(256 + extra++);
	}
	return new Map(chars.map((c, i) => [String.fromCodePoint(c), bytes[i]]));
}

function byteLevelToHex(piece: string, decoder: Map<string, number>): string {
	let hex = '';
	for (const ch of piece) {
		const b = decoder.get(ch);
		if (b === undefined) throw new Error(`Invalid byte-level character: ${ch}`);
		hex += b.toString(16).padStart(2, '0');
	}
	return hex;
}
