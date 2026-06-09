import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { ModelCard } from './ModelCard';
import type { NodeInfo } from '../../types/topology';

const GB = 1024 * 1024 * 1024;

const nodes: Record<string, NodeInfo> = {
  'node-a': {
    system_info: { model_id: 'Mac Studio', chip: 'M4 Max', memory: 64 * GB },
    mactop_info: { memory: { ram_usage: 22 * GB, ram_total: 64 * GB }, temp: { gpu_temp_avg: 40 }, gpu_usage: [0, 0.1], sys_power: 15 },
    last_mactop_update: Date.now(),
    friendly_name: 'kite3',
  },
  'node-b': {
    system_info: { model_id: 'Mac Mini', chip: 'M4', memory: 16 * GB },
    mactop_info: { memory: { ram_usage: 9 * GB, ram_total: 16 * GB }, temp: { gpu_temp_avg: 37 }, gpu_usage: [0, 0.05], sys_power: 9 },
    last_mactop_update: Date.now(),
    friendly_name: 'kite1',
  },
  'node-c': {
    system_info: { model_id: 'MacBook Pro', chip: 'M4 Pro', memory: 36 * GB },
    mactop_info: { memory: { ram_usage: 15 * GB, ram_total: 36 * GB }, temp: { gpu_temp_avg: 45 }, gpu_usage: [0, 0.2], sys_power: 20 },
    last_mactop_update: Date.now(),
    friendly_name: 'macbook-dev',
  },
};

const meta: Meta<typeof ModelCard> = {
  title: 'Models/ModelCard',
  component: ModelCard,
  parameters: { layout: 'centered' },
  decorators: [
    (Story) => (
      <div style={{ width: 320, background: '#000', padding: 16 }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof ModelCard>;

export const FitsWithPlacement: Story = {
  args: {
    model: { id: 'mlx-community/Qwen3-30B-A3B-4bit', name: 'Qwen3 30B A3B', storage_size_megabytes: 18 * 1024 },
    nodes,
    sharding: 'Pipeline',
    runtime: 'MlxRing',
    tags: ['FASTEST'],
    apiPreview: {
      model_id: 'mlx-community/Qwen3-30B-A3B-4bit',
      sharding: 'Pipeline',
      instance_meta: 'MlxRing',
      instance: null,
      memory_delta_by_node: { 'node-a': 12 * GB, 'node-b': 0, 'node-c': 6 * GB },
      error: null,
    },
  },
};

export const TensorSharding: Story = {
  args: {
    model: { id: 'mlx-community/Llama-3.1-70B-4bit', name: 'Llama 3.1 70B', storage_size_megabytes: 40 * 1024 },
    nodes,
    sharding: 'Tensor',
    runtime: 'MlxJaccl',
    apiPreview: {
      model_id: 'mlx-community/Llama-3.1-70B-4bit',
      sharding: 'Tensor',
      instance_meta: 'MlxJaccl',
      instance: null,
      memory_delta_by_node: { 'node-a': 20 * GB, 'node-b': 8 * GB, 'node-c': 12 * GB },
      error: null,
    },
  },
};

export const InsufficientMemory: Story = {
  args: {
    model: { id: 'deepseek-v3-4bit', name: 'DeepSeek V3 671B', storage_size_megabytes: 360 * 1024 },
    nodes,
    sharding: 'Pipeline',
    runtime: 'MlxRing',
    apiPreview: {
      model_id: 'deepseek-v3-4bit',
      sharding: 'Pipeline',
      instance_meta: 'MlxRing',
      instance: null,
      memory_delta_by_node: null,
      error: 'Insufficient memory across cluster',
    },
  },
};

export const Launching: Story = {
  args: {
    model: { id: 'qwen3-8b-4bit', name: 'Qwen3 8B', storage_size_megabytes: 5 * 1024 },
    isLaunching: true,
    nodes: { 'node-a': nodes['node-a'] },
    apiPreview: {
      model_id: 'qwen3-8b-4bit',
      sharding: 'Pipeline',
      instance_meta: 'MlxRing',
      instance: null,
      memory_delta_by_node: { 'node-a': 5 * GB },
      error: null,
    },
  },
};

export const WithDownloadProgress: Story = {
  args: {
    model: { id: 'mlx-community/Qwen3-30B-A3B-4bit', name: 'Qwen3 30B A3B', storage_size_megabytes: 18 * 1024 },
    nodes,
    downloadStatus: {
      isDownloading: true,
      progress: null,
      perNode: [
        { nodeId: 'node-a', nodeName: 'kite3', status: 'completed', percentage: 100, progress: null },
        { nodeId: 'node-b', nodeName: 'kite1', status: 'downloading', percentage: 67, progress: { totalBytes: 18 * GB, downloadedBytes: 12 * GB, speed: 150 * 1024 * 1024, etaMs: 40000, percentage: 67, completedFiles: 3, totalFiles: 5, files: [] } },
        { nodeId: 'node-c', nodeName: 'macbook-dev', status: 'pending', percentage: 0, progress: null },
      ],
    },
    apiPreview: {
      model_id: 'mlx-community/Qwen3-30B-A3B-4bit',
      sharding: 'Pipeline',
      instance_meta: 'MlxRing',
      instance: null,
      memory_delta_by_node: { 'node-a': 12 * GB, 'node-b': 0, 'node-c': 6 * GB },
      error: null,
    },
  },
};

export const SingleNode: Story = {
  args: {
    model: { id: 'qwen3-8b-4bit', name: 'Qwen3 8B', storage_size_megabytes: 5 * 1024 },
    nodes: { 'node-a': nodes['node-a'] },
    apiPreview: {
      model_id: 'qwen3-8b-4bit',
      sharding: 'Pipeline',
      instance_meta: 'MlxRing',
      instance: null,
      memory_delta_by_node: { 'node-a': 5 * GB },
      error: null,
    },
  },
};

export const Interactive: Story = {
  render: () => {
    const [launching, setLaunching] = useState(false);
    return (
      <ModelCard
        model={{ id: 'mlx-community/Qwen3-30B-A3B-4bit', name: 'Qwen3 30B A3B', storage_size_megabytes: 18 * 1024 }}
        nodes={nodes}
        isLaunching={launching}
        onLaunch={() => {
          setLaunching(true);
          setTimeout(() => setLaunching(false), 3000);
        }}
        apiPreview={{
          model_id: 'mlx-community/Qwen3-30B-A3B-4bit',
          sharding: 'Pipeline',
          instance_meta: 'MlxRing',
          instance: null,
          memory_delta_by_node: { 'node-a': 12 * GB, 'node-b': 0, 'node-c': 6 * GB },
          error: null,
        }}
      />
    );
  },
};
