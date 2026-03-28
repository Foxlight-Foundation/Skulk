import type { Meta, StoryObj } from '@storybook/react';
import { RunningInstanceCard } from './RunningInstanceCard';

const meta: Meta<typeof RunningInstanceCard> = {
  title: 'Cluster/RunningInstanceCard',
  component: RunningInstanceCard,
  decorators: [(Story) => <div style={{ padding: 32, background: '#000' }}><Story /></div>],
};

export default meta;
type Story = StoryObj<typeof RunningInstanceCard>;

export const Ready: Story = {
  args: {
    instanceId: '4ea190d5-abcd-1234-ef56-789012345678',
    modelId: 'mlx-community/Qwen3.5-9B-4bit',
    sharding: 'Pipeline',
    instanceType: 'MlxRing',
    nodeName: 'kite2',
    status: 'ready',
    onDelete: () => {},
  },
};

export const Running: Story = {
  args: {
    instanceId: '7fb301c2-1111-2222-3333-444455556666',
    modelId: 'mlx-community/Llama-3.1-8B-Instruct-4bit',
    sharding: 'Tensor',
    instanceType: 'MlxJaccl',
    nodeName: 'kite1',
    status: 'running',
    onDelete: () => {},
  },
};

export const Loading: Story = {
  args: {
    instanceId: 'b2c4d6e8-aaaa-bbbb-cccc-ddddeeee0000',
    modelId: 'mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits',
    sharding: 'Pipeline',
    instanceType: 'MlxRing',
    nodeName: 'kite3',
    status: 'loading',
    loadProgress: 45,
    statusMessage: 'Downloading layers 14/32...',
    onDelete: () => {},
  },
};

export const WarmingUp: Story = {
  args: {
    instanceId: 'c3d5e7f9-1234-5678-9abc-def012345678',
    modelId: 'mlx-community/Qwen3-30B-A3B-4bit',
    sharding: 'Pipeline',
    instanceType: 'MlxRing',
    nodeName: 'kite2',
    status: 'warming_up',
    loadProgress: 90,
    statusMessage: 'Compiling model graph...',
    onDelete: () => {},
  },
};

export const Failed: Story = {
  args: {
    instanceId: 'deadbeef-dead-beef-dead-beefdeadbeef',
    modelId: 'mlx-community/DeepSeek-V3-0324',
    sharding: 'Tensor',
    instanceType: 'MlxJaccl',
    nodeName: 'kite1',
    status: 'failed',
    statusMessage: 'Out of memory: requires 48GB, only 32GB available',
    onDelete: () => {},
  },
};

export const ShuttingDown: Story = {
  args: {
    instanceId: 'aabb1122-3344-5566-7788-99aabbccddee',
    modelId: 'mlx-community/Qwen3.5-9B-4bit',
    sharding: 'Pipeline',
    instanceType: 'MlxRing',
    nodeName: 'kite3',
    status: 'shutting_down',
  },
};

export const NoDeleteButton: Story = {
  args: {
    instanceId: '4ea190d5-abcd-1234-ef56-789012345678',
    modelId: 'mlx-community/Qwen3.5-9B-4bit',
    sharding: 'Pipeline',
    instanceType: 'MlxRing',
    nodeName: 'kite2',
    status: 'ready',
  },
};

export const LongModelId: Story = {
  args: {
    instanceId: '12345678-abcd-efgh-ijkl-mnopqrstuvwx',
    modelId: 'mlx-community/some-very-long-model-name-that-might-wrap-to-multiple-lines-4bit',
    sharding: 'Pipeline',
    instanceType: 'MlxRing',
    nodeName: 'kite2',
    status: 'ready',
    onDelete: () => {},
  },
};
