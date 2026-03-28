import type { Meta, StoryObj } from '@storybook/react';
import { InstancePanel } from './InstancePanel';
import type { InstanceCardData } from './InstancePanel';

const meta: Meta<typeof InstancePanel> = {
  title: 'Layout/InstancePanel',
  component: InstancePanel,
  decorators: [(Story) => (
    <div style={{ height: '100vh', display: 'flex', justifyContent: 'flex-end', background: '#000' }}>
      <Story />
    </div>
  )],
};

export default meta;
type Story = StoryObj<typeof InstancePanel>;

const readyInstance: InstanceCardData = {
  instanceId: '4ea190d5-abcd-1234-ef56-789012345678',
  modelId: 'mlx-community/Qwen3.5-9B-4bit',
  sharding: 'Pipeline',
  instanceType: 'MlxRing',
  nodeName: 'kite2',
  status: 'ready',
};

const loadingInstance: InstanceCardData = {
  instanceId: 'b2c4d6e8-aaaa-bbbb-cccc-ddddeeee0000',
  modelId: 'mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits',
  sharding: 'Pipeline',
  instanceType: 'MlxRing',
  nodeName: 'kite3',
  status: 'loading',
  loadProgress: 45,
  statusMessage: 'Downloading layers 14/32...',
};

const failedInstance: InstanceCardData = {
  instanceId: 'deadbeef-dead-beef-dead-beefdeadbeef',
  modelId: 'mlx-community/DeepSeek-V3-0324',
  sharding: 'Tensor',
  instanceType: 'MlxJaccl',
  nodeName: 'kite1',
  status: 'failed',
  statusMessage: 'Out of memory: requires 48GB, only 32GB available',
};

const runningInstance: InstanceCardData = {
  instanceId: '7fb301c2-1111-2222-3333-444455556666',
  modelId: 'mlx-community/Llama-3.1-8B-Instruct-4bit',
  sharding: 'Tensor',
  instanceType: 'MlxJaccl',
  nodeName: 'kite1',
  status: 'running',
};

export const SingleReady: Story = {
  args: {
    instances: [readyInstance],
    onDelete: () => {},
  },
};

export const MultipleInstances: Story = {
  args: {
    instances: [readyInstance, loadingInstance, failedInstance, runningInstance],
    onDelete: () => {},
  },
};

export const AllLoading: Story = {
  args: {
    instances: [
      loadingInstance,
      { ...loadingInstance, instanceId: 'aabb1122-0000-0000-0000-000000000000', modelId: 'mlx-community/Qwen3-30B-A3B-4bit', loadProgress: 78, statusMessage: 'Downloading layers 25/32...' },
    ],
    onDelete: () => {},
  },
};
