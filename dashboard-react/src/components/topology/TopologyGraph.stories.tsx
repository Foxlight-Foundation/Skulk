import type { Meta, StoryObj } from '@storybook/react-vite';
import type { TopologyData, NodeInfo } from '../../types/topology';
import { TopologyGraph } from './TopologyGraph';

const GB = 1024 * 1024 * 1024;

function node(
  name: string,
  model: string,
  ramUsed: number,
  ramTotal: number,
  gpuPct: number,
  temp: number,
  power: number,
): NodeInfo {
  return {
    system_info: { model_id: model, chip: 'M4 Max', memory: ramTotal },
    macmon_info: {
      memory: { ram_usage: ramUsed, ram_total: ramTotal },
      temp: { gpu_temp_avg: temp },
      gpu_usage: [0, gpuPct / 100],
      sys_power: power,
    },
    last_macmon_update: Date.now(),
    friendly_name: name,
  };
}

const Wrap = ({ children }: { children: React.ReactNode }) => (
  <div style={{ width: '100%', height: '100vh', background: '#111' }}>
    {children}
  </div>
);

const meta: Meta<typeof TopologyGraph> = {
  title: 'Topology/TopologyGraph',
  component: TopologyGraph,
  parameters: { layout: 'fullscreen' },
  decorators: [(Story) => <Wrap><Story /></Wrap>],
};

export default meta;
type Story = StoryObj<typeof TopologyGraph>;

const singleNode: TopologyData = {
  nodes: {
    'node-a': node('kite3', 'Mac Studio', 15.4 * GB, 24 * GB, 15, 39, 11),
  },
  edges: [],
};

export const SingleNode: Story = {
  args: { data: singleNode },
};

const twoNodes: TopologyData = {
  nodes: {
    'node-a': node('kite1', 'Mac Mini', 9 * GB, 16 * GB, 5, 37, 9),
    'node-b': node('kite3', 'Mac Studio', 15.3 * GB, 24 * GB, 16, 40, 11),
  },
  edges: [
    { source: 'node-a', target: 'node-b' },
    { source: 'node-b', target: 'node-a' },
  ],
};

export const TwoNodes: Story = {
  args: { data: twoNodes },
};

const threeNodes: TopologyData = {
  nodes: {
    'node-a': node('kite1', 'Mac Mini', 9 * GB, 16 * GB, 5, 37, 9),
    'node-b': node('kite2', 'Mac Mini', 7.5 * GB, 16 * GB, 0, 30, 9),
    'node-c': node('kite3', 'Mac Studio', 15.3 * GB, 24 * GB, 16, 40, 11),
  },
  edges: [
    { source: 'node-a', target: 'node-b' },
    { source: 'node-b', target: 'node-a' },
    { source: 'node-b', target: 'node-c' },
    { source: 'node-c', target: 'node-b' },
    { source: 'node-a', target: 'node-c' },
    { source: 'node-c', target: 'node-a' },
  ],
};

export const ThreeNodes: Story = {
  args: { data: threeNodes },
  name: 'Three nodes (triangle)',
};

const fourNodes: TopologyData = {
  nodes: {
    'node-a': node('kite1', 'Mac Mini', 9 * GB, 16 * GB, 5, 37, 9),
    'node-b': node('kite2', 'Mac Mini', 7.5 * GB, 16 * GB, 0, 30, 9),
    'node-c': node('kite3', 'Mac Studio', 15.3 * GB, 24 * GB, 16, 40, 11),
    'node-d': node('macbook-dev', 'MacBook Pro', 28 * GB, 36 * GB, 55, 55, 35),
  },
  edges: [
    { source: 'node-a', target: 'node-b' },
    { source: 'node-b', target: 'node-a' },
    { source: 'node-b', target: 'node-c' },
    { source: 'node-c', target: 'node-b' },
    { source: 'node-c', target: 'node-d' },
    { source: 'node-d', target: 'node-c' },
    { source: 'node-d', target: 'node-a' },
    { source: 'node-a', target: 'node-d' },
  ],
};

export const FourNodes: Story = {
  args: { data: fourNodes },
  name: 'Four nodes (square)',
};

const sixNodes: TopologyData = {
  nodes: {
    'n1': node('studio-1', 'Mac Studio', 50 * GB, 64 * GB, 80, 70, 55),
    'n2': node('studio-2', 'Mac Studio', 45 * GB, 64 * GB, 65, 60, 45),
    'n3': node('mini-1', 'Mac Mini', 12 * GB, 16 * GB, 30, 45, 18),
    'n4': node('mini-2', 'Mac Mini', 10 * GB, 16 * GB, 20, 42, 15),
    'n5': node('macbook', 'MacBook Pro', 28 * GB, 36 * GB, 55, 55, 35),
    'n6': node('linux-box', '', 100 * GB, 128 * GB, 90, 82, 68),
  },
  edges: [
    { source: 'n1', target: 'n2' }, { source: 'n2', target: 'n1' },
    { source: 'n2', target: 'n3' }, { source: 'n3', target: 'n2' },
    { source: 'n3', target: 'n4' }, { source: 'n4', target: 'n3' },
    { source: 'n4', target: 'n5' }, { source: 'n5', target: 'n4' },
    { source: 'n5', target: 'n6' }, { source: 'n6', target: 'n5' },
    { source: 'n6', target: 'n1' }, { source: 'n1', target: 'n6' },
  ],
};

export const SixNodes: Story = {
  args: { data: sixNodes },
  name: 'Six nodes (hexagon)',
};
