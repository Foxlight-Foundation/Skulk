import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { ModelBrowser } from './ModelBrowser';
import type { ModelInfo, ModelFitStatus, HuggingFaceModel } from '../../types/models';

const GB = 1024;

const SAMPLE_MODELS: ModelInfo[] = [
  { id: 'qwen3-30b-4bit', name: 'Qwen3 30B A3B', storage_size_megabytes: 18 * GB, quantization: '4-bit', base_model: 'qwen3-30b', capabilities: ['text', 'thinking', 'code'], family: 'qwen' },
  { id: 'qwen3-30b-8bit', name: 'Qwen3 30B A3B', storage_size_megabytes: 32 * GB, quantization: '8-bit', base_model: 'qwen3-30b', capabilities: ['text', 'thinking', 'code'], family: 'qwen' },
  { id: 'qwen3-8b-4bit', name: 'Qwen3 8B', storage_size_megabytes: 5 * GB, quantization: '4-bit', base_model: 'qwen3-8b', capabilities: ['text', 'thinking', 'code'], family: 'qwen' },
  { id: 'llama-70b-4bit', name: 'Llama 3.1 70B', storage_size_megabytes: 40 * GB, quantization: '4-bit', base_model: 'llama-70b', capabilities: ['text', 'code'], family: 'llama' },
  { id: 'llama-70b-8bit', name: 'Llama 3.1 70B', storage_size_megabytes: 72 * GB, quantization: '8-bit', base_model: 'llama-70b', capabilities: ['text', 'code'], family: 'llama' },
  { id: 'llama-8b-4bit', name: 'Llama 3.1 8B', storage_size_megabytes: 5 * GB, quantization: '4-bit', base_model: 'llama-8b', capabilities: ['text', 'code'], family: 'llama' },
  { id: 'deepseek-v3-4bit', name: 'DeepSeek V3 671B', storage_size_megabytes: 360 * GB, quantization: '4-bit', base_model: 'deepseek-v3', capabilities: ['text', 'thinking', 'code', 'vision'], family: 'deepseek' },
  { id: 'deepseek-r1-8b-4bit', name: 'DeepSeek R1 8B', storage_size_megabytes: 5 * GB, quantization: '4-bit', base_model: 'deepseek-r1-8b', capabilities: ['text', 'thinking'], family: 'deepseek' },
  { id: 'flux-schnell', name: 'FLUX.1 Schnell', storage_size_megabytes: 12 * GB, base_model: 'flux-schnell', capabilities: ['image_gen'], family: 'flux' },
  { id: 'glm4-9b-4bit', name: 'GLM-4 9B', storage_size_megabytes: 6 * GB, quantization: '4-bit', base_model: 'glm4-9b', capabilities: ['text', 'vision'], family: 'glm' },
  { id: 'minimax-text-01', name: 'MiniMax-Text-01', storage_size_megabytes: 250 * GB, base_model: 'minimax-text-01', capabilities: ['text'], family: 'minimax' },
];

const SAMPLE_HF: HuggingFaceModel[] = [
  { id: 'mlx-community/Qwen3-30B-A3B-4bit', author: 'mlx-community', downloads: 125000, likes: 340, last_modified: '2026-03-20', tags: [] },
  { id: 'mlx-community/Llama-3.1-8B-Instruct-4bit', author: 'mlx-community', downloads: 2500000, likes: 1200, last_modified: '2026-03-15', tags: [] },
  { id: 'deepseek-ai/DeepSeek-V3-0324', author: 'deepseek-ai', downloads: 50000, likes: 890, last_modified: '2026-03-24', tags: [] },
];

const canFit = (id: string) => {
  const big = ['deepseek-v3-4bit', 'llama-70b-8bit', 'minimax-text-01'];
  return !big.includes(id);
};

const fitStatus = (id: string): ModelFitStatus => {
  if (id === 'deepseek-v3-4bit' || id === 'minimax-text-01') return 'too_large';
  if (id === 'llama-70b-4bit' || id === 'llama-70b-8bit') return 'fits_cluster_capacity';
  return 'fits_now';
};

const meta: Meta<typeof ModelBrowser> = {
  title: 'Models/ModelBrowser',
  component: ModelBrowser,
  parameters: { layout: 'fullscreen' },
};

export default meta;
type Story = StoryObj<typeof ModelBrowser>;

export const FullPage: Story = {
  render: () => {
    const [selected, setSelected] = useState<string | null>(null);
    const [favs, setFavs] = useState(new Set(['qwen3-30b']));
    return (
      <div style={{ height: '100vh' }}>
        <ModelBrowser
          models={SAMPLE_MODELS}
          selectedModelId={selected}
          favorites={favs}
          recentModelIds={['qwen3-30b-4bit', 'llama-8b-4bit']}
          canModelFit={canFit}
          getModelFitStatus={fitStatus}
          onSelect={setSelected}
          onToggleFavorite={(id) => {
            setFavs((prev) => {
              const next = new Set(prev);
              if (next.has(id)) next.delete(id);
              else next.add(id);
              return next;
            });
          }}
          instanceStatuses={{
            'qwen3-30b-4bit': { status: 'Ready', statusClass: 'ready' },
            'llama-8b-4bit': { status: 'Loading', statusClass: 'loading' },
          }}
          downloadStatusMap={new Map([
            ['qwen3-30b-4bit', { available: true, nodeNames: ['kite3'], nodeIds: ['abc'] }],
            ['llama-8b-4bit', { available: true, nodeNames: ['kite1'], nodeIds: ['def'] }],
          ])}
          hfTrendingModels={SAMPLE_HF}
        />
      </div>
    );
  },
};

export const ConstrainedPanel: Story = {
  render: () => {
    const [selected, setSelected] = useState<string | null>(null);
    return (
      <div style={{ width: 'min(500px, calc(100vw - 48px))', height: 450, margin: '40px auto', borderRadius: 12, overflow: 'hidden' }}>
        <ModelBrowser
          models={SAMPLE_MODELS}
          selectedModelId={selected}
          favorites={new Set()}
          canModelFit={canFit}
          getModelFitStatus={fitStatus}
          onSelect={setSelected}
          onToggleFavorite={() => {}}
        />
      </div>
    );
  },
};

export const EmptyState: Story = {
  render: () => (
    <div style={{ height: '100vh' }}>
      <ModelBrowser
        models={[]}
        selectedModelId={null}
        favorites={new Set()}
        canModelFit={() => false}
        getModelFitStatus={() => 'too_large'}
        onSelect={() => {}}
        onToggleFavorite={() => {}}
      />
    </div>
  ),
};

export const StoreDownloadMode: Story = {
  render: () => {
    const [selected, setSelected] = useState<string | null>(null);
    return (
      <div style={{ height: '100vh' }}>
        <ModelBrowser
          models={SAMPLE_MODELS}
          selectedModelId={selected}
          favorites={new Set()}
          canModelFit={canFit}
          getModelFitStatus={fitStatus}
          onSelect={setSelected}
          onToggleFavorite={() => {}}
          mode="store-download"
        />
      </div>
    );
  },
};
