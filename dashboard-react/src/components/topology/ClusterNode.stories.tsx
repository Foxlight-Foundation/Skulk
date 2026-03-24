import type { Meta, StoryObj } from '@storybook/react-vite';
import type { NodeInfo } from '../../types/topology';
import { ClusterNode } from './ClusterNode';

const GB = 1024 * 1024 * 1024;

const SvgCanvas = ({ children }: { children: React.ReactNode }) => (
  <svg width={400} height={300} style={{ background: '#111' }}>
    {children}
  </svg>
);

function makeNode(overrides: Partial<NodeInfo> & { model_id?: string } = {}): NodeInfo {
  const { model_id = 'Mac Studio', ...rest } = overrides;
  return {
    system_info: { model_id, chip: 'M4 Max', memory: 24 * GB },
    macmon_info: {
      memory: { ram_usage: 15.4 * GB, ram_total: 24 * GB },
      temp: { gpu_temp_avg: 39 },
      gpu_usage: [0, 0.15],
      sys_power: 11,
    },
    last_macmon_update: Date.now(),
    friendly_name: 'kite3',
    ...rest,
  };
}

const meta: Meta<typeof ClusterNode> = {
  title: 'Topology/ClusterNode',
  component: ClusterNode,
  parameters: { layout: 'centered' },
  argTypes: {
    scale: { control: { type: 'range', min: 0.3, max: 2, step: 0.1 } },
  },
  decorators: [(Story) => <SvgCanvas><Story /></SvgCanvas>],
};

export default meta;
type Story = StoryObj<typeof ClusterNode>;

export const MacStudio: Story = {
  args: {
    nodeId: 'abc123',
    nodeInfo: makeNode(),
    x: 200,
    y: 150,
    scale: 1,
  },
};

export const MacBookPro: Story = {
  args: {
    nodeId: 'def456',
    nodeInfo: makeNode({
      model_id: 'MacBook Pro',
      friendly_name: 'macbook-dev',
      macmon_info: {
        memory: { ram_usage: 28 * GB, ram_total: 36 * GB },
        temp: { gpu_temp_avg: 55 },
        gpu_usage: [0, 0.55],
        sys_power: 35,
      },
    }),
    x: 200,
    y: 150,
    scale: 1,
  },
};

export const MacMini: Story = {
  args: {
    nodeId: 'ghi789',
    nodeInfo: makeNode({
      model_id: 'Mac Mini',
      friendly_name: 'mini-server',
      macmon_info: {
        memory: { ram_usage: 12 * GB, ram_total: 16 * GB },
        temp: { gpu_temp_avg: 45 },
        gpu_usage: [0, 0.30],
        sys_power: 18,
      },
    }),
    x: 200,
    y: 150,
    scale: 1,
  },
};

export const UnknownDevice: Story = {
  args: {
    nodeId: 'unk000',
    nodeInfo: makeNode({
      model_id: undefined,
      friendly_name: 'linux-box',
      macmon_info: {
        memory: { ram_usage: 50 * GB, ram_total: 64 * GB },
        temp: { gpu_temp_avg: 70 },
        gpu_usage: [0, 0.80],
        sys_power: 55,
      },
    }),
    x: 200,
    y: 150,
    scale: 1,
  },
  name: 'Unknown (hexagon with RAM fill)',
};

export const HotNode: Story = {
  args: {
    nodeId: 'hot001',
    nodeInfo: makeNode({
      friendly_name: 'overloaded',
      macmon_info: {
        memory: { ram_usage: 22 * GB, ram_total: 24 * GB },
        temp: { gpu_temp_avg: 82 },
        gpu_usage: [0, 0.95],
        sys_power: 68,
      },
    }),
    x: 200,
    y: 150,
    scale: 1,
  },
  name: 'Hot (92% RAM, 95% GPU, 82°C)',
};

export const NoTelemetry: Story = {
  args: {
    nodeId: 'nodata',
    nodeInfo: {
      system_info: { model_id: 'Mac Studio' },
      last_macmon_update: 0,
      friendly_name: 'offline',
    },
    x: 200,
    y: 150,
    scale: 1,
  },
  name: 'No telemetry data',
};

export const ScaledDown: Story = {
  args: {
    nodeId: 'small01',
    nodeInfo: makeNode({ friendly_name: 'tiny' }),
    x: 200,
    y: 150,
    scale: 0.6,
  },
  name: 'Scaled down (0.6x)',
};

export const MultipleNodes: Story = {
  render: () => (
    <SvgCanvas>
      <ClusterNode
        nodeId="node-a"
        nodeInfo={makeNode({ friendly_name: 'node-a' })}
        x={130} y={150} scale={0.8}
      />
      <ClusterNode
        nodeId="node-b"
        nodeInfo={makeNode({
          model_id: 'MacBook Pro',
          friendly_name: 'node-b',
          macmon_info: {
            memory: { ram_usage: 28 * GB, ram_total: 36 * GB },
            temp: { gpu_temp_avg: 60 },
            gpu_usage: [0, 0.65],
            sys_power: 40,
          },
        })}
        x={300} y={150} scale={0.8}
      />
    </SvgCanvas>
  ),
  decorators: [], // override default decorator since we supply our own canvas
  name: 'Multiple nodes side by side',
};
