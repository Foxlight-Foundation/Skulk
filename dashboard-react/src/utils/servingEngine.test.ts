import { describe, expect, it } from 'vitest';
import { parseBackendTag } from './servingEngine';

describe('parseBackendTag', () => {
  it('names every engine and its accelerator', () => {
    expect(parseBackendTag('comfy-rocm')).toEqual({ engine: 'comfy', accelerator: 'ROCm' });
    expect(parseBackendTag('vllm-cuda')).toEqual({ engine: 'vllm', accelerator: 'CUDA' });
    expect(parseBackendTag('llama_server-vulkan')).toEqual({ engine: 'served', accelerator: 'Vulkan' });
    expect(parseBackendTag('llama_cpp-cpu')).toEqual({ engine: 'llama_cpp', accelerator: 'CPU' });
    expect(parseBackendTag('audio_cpp-metal')).toEqual({ engine: 'audio_cpp', accelerator: 'Metal' });
    expect(parseBackendTag('mlx_audio-metal')).toEqual({ engine: 'mlx_audio', accelerator: 'Metal' });
    expect(parseBackendTag('test_video')).toEqual({ engine: 'test_video', accelerator: null });
    expect(parseBackendTag('mlx')).toEqual({ engine: 'mlx', accelerator: null });
  });

  it('keeps the historical MLX reading for missing or unknown tags', () => {
    expect(parseBackendTag(undefined)).toEqual({ engine: 'mlx', accelerator: null });
    expect(parseBackendTag('')).toEqual({ engine: 'mlx', accelerator: null });
    expect(parseBackendTag('future_engine-npu')).toEqual({ engine: 'mlx', accelerator: null });
  });
});
