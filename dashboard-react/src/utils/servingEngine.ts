/** The engine that serves an instance. `served` is llama.cpp's llama-server. */
export type ServingEngine =
  | 'mlx'
  | 'mlx_audio'
  | 'llama_cpp'
  | 'served'
  | 'vllm'
  | 'comfy'
  | 'audio_cpp'
  | 'test_video';

/** An engine and, when the tag names one, the accelerator it runs on. */
export interface ServingBackend {
  engine: ServingEngine;
  /** Display name of the compute backend, such as `ROCm`; null for bare tags. */
  accelerator: string | null;
}

const ENGINES: Record<string, ServingEngine> = {
  mlx: 'mlx',
  mlx_audio: 'mlx_audio',
  llama_cpp: 'llama_cpp',
  llama_server: 'served',
  vllm: 'vllm',
  comfy: 'comfy',
  audio_cpp: 'audio_cpp',
  test_video: 'test_video',
};

const ACCELERATORS: Record<string, string> = {
  metal: 'Metal',
  cuda: 'CUDA',
  rocm: 'ROCm',
  vulkan: 'Vulkan',
  cpu: 'CPU',
};

/**
 * Read a Skulk backend tag such as `comfy-rocm`, `llama_server-vulkan` or `mlx`.
 *
 * Engine names never contain `-`, so the tag splits at its first hyphen. An
 * unknown or missing engine reads as MLX, the historical default, so older
 * state without a tag keeps its existing label.
 */
export function parseBackendTag(tag: string | null | undefined): ServingBackend {
  if (!tag) return { engine: 'mlx', accelerator: null };
  const separator = tag.indexOf('-');
  const engineName = separator === -1 ? tag : tag.slice(0, separator);
  const computeName = separator === -1 ? null : tag.slice(separator + 1);
  return {
    engine: ENGINES[engineName] ?? 'mlx',
    accelerator: computeName ? ACCELERATORS[computeName] ?? null : null,
  };
}
