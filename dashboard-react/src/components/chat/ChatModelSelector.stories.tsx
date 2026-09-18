import type { Meta, StoryObj } from '@storybook/react-vite';
import { ChatModelSelector } from './ChatModelSelector';
import type { ChatModelInfo } from '../../types/chat';

const GB = 1024;

const sampleModels: ChatModelInfo[] = [
  { id: 'qwen3-30b-4bit', name: 'Qwen3 30B A3B', base_model: 'qwen3-30b', storage_size_megabytes: 18 * GB, capabilities: ['text', 'thinking', 'code'], family: 'qwen', quantization: '4-bit' },
  { id: 'qwen3-8b-4bit', name: 'Qwen3 8B', base_model: 'qwen3-8b', storage_size_megabytes: 5 * GB, capabilities: ['text', 'code'], family: 'qwen', quantization: '4-bit' },
  { id: 'llama-70b-4bit', name: 'Llama 3.1 70B', base_model: 'llama-3.1-70b', storage_size_megabytes: 40 * GB, capabilities: ['text', 'code'], family: 'llama', quantization: '4-bit' },
  { id: 'llama-8b-4bit', name: 'Llama 3.1 8B', base_model: 'llama-3.1-8b', storage_size_megabytes: 5 * GB, capabilities: ['text', 'code'], family: 'llama', quantization: '4-bit' },
  { id: 'deepseek-v3-4bit', name: 'DeepSeek V3', base_model: 'deepseek-v3.1', storage_size_megabytes: 360 * GB, capabilities: ['text', 'thinking', 'code', 'vision'], family: 'deepseek', quantization: '4-bit' },
  { id: 'kimi-k2-4bit', name: 'Kimi K2', base_model: 'kimi-k2', storage_size_megabytes: 50 * GB, capabilities: ['text'], family: 'kimi', quantization: '4-bit' },
];

const meta: Meta<typeof ChatModelSelector> = {
  title: 'Chat/ChatModelSelector',
  component: ChatModelSelector,
  parameters: { layout: 'centered' },
  decorators: [(Story) => <div style={{ padding: 40, minHeight: 400 }}><Story /></div>],
};

export default meta;
type Story = StoryObj<typeof ChatModelSelector>;

export const Default: Story = {
  args: {
    models: sampleModels,
    clusterLabel: '3-node Apple Silicon cluster (104GB)',
    totalMemoryGB: 104,
    onSelect: (id, cat) => alert(`Selected ${id} for ${cat}`),
    onAddModel: () => alert('Add model'),
  },
};

export const SmallCluster: Story = {
  args: {
    models: sampleModels,
    clusterLabel: 'Mac Mini (16GB)',
    totalMemoryGB: 16,
    onSelect: () => {},
    onAddModel: () => {},
  },
  name: 'Small cluster (16GB)',
};

export const NoModels: Story = {
  args: {
    models: [],
    clusterLabel: 'Empty cluster',
    totalMemoryGB: 64,
    onSelect: () => {},
    onAddModel: () => {},
  },
};
