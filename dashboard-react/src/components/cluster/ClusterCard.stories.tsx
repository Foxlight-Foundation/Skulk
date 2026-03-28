import type { Meta, StoryObj } from '@storybook/react-vite';
import { ClusterCard } from './ClusterCard';

const meta: Meta<typeof ClusterCard> = {
  title: 'Cluster/ClusterCard',
  component: ClusterCard,
  parameters: { layout: 'centered' },
  decorators: [(Story) => <div style={{ padding: 32, background: '#000' }}><Story /></div>],
};

export default meta;
type Story = StoryObj<typeof ClusterCard>;

const threeNodes = [
  { nodeId: 'node-1', name: 'kite1', memoryUsedPercent: 52 },
  { nodeId: 'node-2', name: 'kite2', memoryUsedPercent: 64 },
  { nodeId: 'node-3', name: 'kite3', memoryUsedPercent: 87 },
];

const twoNodes = [
  { nodeId: 'node-1', name: 'kite1', memoryUsedPercent: 45 },
  { nodeId: 'node-2', name: 'kite2', memoryUsedPercent: 78 },
];

const oneNode = [
  { nodeId: 'node-1', name: 'kite3', memoryUsedPercent: 79 },
];

export const Running3Nodes: Story = {
  args: {
    modelId: 'mlx-community/Qwen3.5-9B-4bit',
    modelName: 'Qwen3.5-9B-4bit',
    sizeBytes: 5_950_062_560,
    sharding: 'Pipeline',
    instanceType: 'MlxRing',
    nodes: threeNodes,
    isRunning: true,
  },
};

export const Running2Nodes: Story = {
  args: {
    modelId: 'mlx-community/GLM-4.7-Flash-4bit',
    modelName: 'GLM-4.7-Flash-4bit',
    sizeBytes: 16_872_850_407,
    sharding: 'Pipeline',
    instanceType: 'MlxRing',
    nodes: twoNodes,
    isRunning: true,
  },
};

export const Running1Node: Story = {
  args: {
    modelId: 'mlx-community/Llama-3.2-3B-Instruct-4bit',
    modelName: 'Llama-3.2-3B-4bit',
    sizeBytes: 1_863_319_552,
    sharding: 'Pipeline',
    instanceType: 'MlxRing',
    nodes: oneNode,
    isRunning: true,
  },
};

export const Downloading: Story = {
  args: {
    modelId: 'mlx-community/DeepSeek-V3.1-4bit',
    modelName: 'DeepSeek-V3.1-4bit',
    sizeBytes: 405_874_409_472,
    sharding: 'Pipeline',
    instanceType: 'MlxRing',
    nodes: threeNodes,
    isRunning: false,
    downloads: [
      { nodeId: 'node-1', nodeName: 'kite1', percent: 45 },
      { nodeId: 'node-2', nodeName: 'kite2', percent: 100 },
      { nodeId: 'node-3', nodeName: 'kite3', percent: 78 },
    ],
  },
};

export const ReadyToLaunch: Story = {
  args: {
    modelId: 'mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits',
    modelName: 'Nemotron-Nano-9B-4bit',
    sizeBytes: 5_002_791_936,
    sharding: 'Pipeline',
    instanceType: 'MlxRing',
    nodes: twoNodes,
    isRunning: false,
    onLaunch: () => alert('Launch!'),
  },
};

export const TensorJaccl: Story = {
  args: {
    modelId: 'mlx-community/Qwen3.5-122B-A10B-4bit',
    modelName: 'Qwen3.5-122B-A10B-4bit',
    sizeBytes: 69_593_314_272,
    sharding: 'Tensor',
    instanceType: 'MlxJaccl',
    nodes: threeNodes,
    isRunning: true,
  },
};
