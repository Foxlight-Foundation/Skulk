/**
 * "Burst" design language: a model that the local fleet cannot serve as-is
 * is not disabled or an error — it needs capacity the fabric can summon
 * (a burst pod, added nodes). One chip covers both ways a model can exceed
 * the fleet: too large for total fleet memory, or no local node runs its
 * artifact format. Burst rows stay fully interactive; downloading ahead of
 * capacity is legitimate.
 */
export type BurstReason = 'size' | 'engine';

export interface BurstInfo {
  reason: BurstReason;
  /** Weight bytes backing a size verdict; estimated when `estimated`. */
  neededBytes?: number;
  /** True when neededBytes was estimated from the repo name, not a card. */
  estimated?: boolean;
  /** Artifact format behind an engine verdict (GGUF, MLX). */
  format?: string;
}

/** Snapshot of what the local fleet can serve, computed from live state. */
export interface FleetServingSummary {
  /** Sum of every node's total memory in bytes. */
  totalMemoryBytes: number;
  /** True when some node advertises the in-process MLX engine. */
  servesMlx: boolean;
  /** True when some node advertises a GGUF-serving engine (llama.cpp family, vLLM). */
  servesGguf: boolean;
}

/**
 * Rough parameter count in billions parsed from a repo or model name
 * ("Llama-3.3-70B", "8x7B", "Qwen3.5-0.6B"). Active-parameter markers
 * ("A3B" in MoE names) are ignored in favor of the total. Returns null
 * when no trustworthy count is present.
 */
export function parseParamCountBillions(name: string): number | null {
  const tail = name.includes('/') ? name.slice(name.indexOf('/') + 1) : name;
  let best: number | null = null;
  const moe = /(\d+)x(\d+(?:\.\d+)?)[bB](?![a-zA-Z0-9])/g;
  for (const m of tail.matchAll(moe)) {
    const total = Number(m[1]) * Number(m[2]);
    if (best === null || total > best) best = total;
  }
  const plain = /(?:^|[-_. ])(\d+(?:\.\d+)?)[bB](?![a-zA-Z0-9])/g;
  for (const m of tail.matchAll(plain)) {
    const preceding = tail[(m.index ?? 0) - 1];
    // "A3B"-style active-parameter markers describe MoE active params, not
    // the artifact's total weight count.
    if (preceding === 'A' || preceding === 'a') continue;
    const total = Number(m[1]);
    if (best === null || total > best) best = total;
  }
  return best;
}

/** Effective bits per weight for a derived quant label, GGUF codes included. */
export function bitsForQuantLabel(quant: string | null): number {
  if (!quant) return 16; // unquantized repos ship bf16/fp16 weights
  const q = quant.toLowerCase();
  // Multi-digit widths first so "f32"/"bf16" don't match their single digits.
  if (q.includes('32')) return 32;
  if (q.includes('16')) return 16;
  if (q.includes('2')) return 2.8;
  if (q.includes('3')) return 3.6;
  if (q.includes('4')) return 4.6;
  if (q.includes('5')) return 5.6;
  if (q.includes('6')) return 6.6;
  if (q.includes('8')) return 8.5;
  return 16;
}

/**
 * Estimated artifact weight bytes from a repo name and derived quant label.
 * Null when the name carries no parameter count.
 */
export function estimateArtifactBytes(name: string, quant: string | null): number | null {
  const billions = parseParamCountBillions(name);
  if (billions === null) return null;
  return Math.round(billions * 1e9 * (bitsForQuantLabel(quant) / 8));
}

/**
 * Best available weight bytes for a Hugging Face result, most-trusted first:
 * exact GGUF artifact size (measured), then metadata parameter count times
 * the derived quant width (semi-estimated), then a name-derived estimate.
 */
export function hfWeightBytes(
  model: { id: string; param_count?: number | null; total_file_size?: number | null },
  quant: string | null,
): { bytes: number; estimated: boolean } | null {
  if (model.total_file_size) return { bytes: model.total_file_size, estimated: false };
  if (model.param_count) {
    return {
      bytes: Math.round(model.param_count * (bitsForQuantLabel(quant) / 8)),
      estimated: true,
    };
  }
  const named = estimateArtifactBytes(model.id, quant);
  return named === null ? null : { bytes: named, estimated: true };
}

/**
 * Burst verdict for one concrete artifact: known weight bytes (card truth)
 * or a name-derived estimate, plus its format, against the fleet summary.
 * Size wins over engine when both apply so the tooltip leads with the
 * harder constraint.
 */
export function burstVerdict(
  fleet: FleetServingSummary,
  weightBytes: number | null,
  estimated: boolean,
  format: string | null,
): BurstInfo | null {
  if (weightBytes !== null && weightBytes > fleet.totalMemoryBytes) {
    return { reason: 'size', neededBytes: weightBytes, estimated };
  }
  if (format === 'MLX' && !fleet.servesMlx) return { reason: 'engine', format };
  if (format === 'GGUF' && !fleet.servesGguf) return { reason: 'engine', format };
  return null;
}
