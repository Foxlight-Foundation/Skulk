import type { Meta, StoryObj } from '@storybook/react-vite';
import { StoreRegistryTable, type StoreRegistryEntry, type StoreDownloadProgress } from './StoreRegistryTable';

const GB = 1024 * 1024 * 1024;

const sampleEntries: StoreRegistryEntry[] = [
  { model_id: 'mlx-community/Qwen3-30B-A3B-4bit', total_bytes: 18 * GB, files: ['model.safetensors', 'config.json', 'tokenizer.json'], downloaded_at: new Date(Date.now() - 3600000).toISOString() },
  { model_id: 'mlx-community/Llama-3.1-8B-Instruct-4bit', total_bytes: 5 * GB, files: ['model.safetensors', 'config.json'], downloaded_at: new Date(Date.now() - 86400000).toISOString() },
  { model_id: 'mlx-community/DeepSeek-R1-8B-4bit', total_bytes: 5.2 * GB, files: ['model-00001.safetensors', 'model-00002.safetensors', 'config.json'], downloaded_at: new Date(Date.now() - 604800000).toISOString() },
];

const activeDownloads: StoreDownloadProgress[] = [
  { modelId: 'mlx-community/Qwen3-30B-A3B-4bit', progress: 0.85, status: 'downloading' },
  { modelId: 'mlx-community/FLUX.1-schnell-4bit', progress: 0.32, status: 'downloading' },
];

const meta: Meta<typeof StoreRegistryTable> = {
  title: 'Layout/StoreRegistryTable',
  component: StoreRegistryTable,
  parameters: { layout: 'centered' },
  decorators: [(Story) => <div style={{ width: 700, padding: 24, background: '#111' }}><Story /></div>],
};

export default meta;
type Story = StoryObj<typeof StoreRegistryTable>;

export const Default: Story = {
  args: {
    entries: sampleEntries,
    activeModelIds: ['mlx-community/Qwen3-30B-A3B-4bit'],
    onRefresh: () => {},
    onDelete: (e) => alert(`Delete: ${e.model_id}`),
  },
};

export const WithDownloads: Story = {
  args: {
    entries: sampleEntries,
    activeDownloads,
    activeModelIds: ['mlx-community/Qwen3-30B-A3B-4bit'],
    onRefresh: () => {},
    onDelete: () => {},
  },
};

export const Loading: Story = {
  args: {
    entries: [],
    loading: true,
    onRefresh: () => {},
    onDelete: () => {},
  },
};

export const Empty: Story = {
  args: {
    entries: [],
    onRefresh: () => {},
    onDelete: () => {},
  },
};
