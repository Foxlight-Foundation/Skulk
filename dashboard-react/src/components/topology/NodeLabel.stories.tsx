import type { Meta, StoryObj } from '@storybook/react-vite';
import { NodeLabel } from './NodeLabel';

const SvgWrap = ({ children, width }: { children: React.ReactNode; width: number }) => (
  <svg width={width + 20} height={60} style={{ background: '#000' }}>
    <g transform={`translate(${(width + 20) / 2}, 15)`}>{children}</g>
  </svg>
);

const meta: Meta<typeof NodeLabel> = {
  title: 'Topology/NodeLabel',
  component: NodeLabel,
  parameters: { layout: 'centered' },
  argTypes: {
    fontSize: { control: { type: 'range', min: 8, max: 20, step: 1 } },
  },
  decorators: [
    (Story) => {
      return <SvgWrap width={200}><Story /></SvgWrap>;
    },
  ],
};

export default meta;
type Story = StoryObj<typeof NodeLabel>;

const GB = 1024 * 1024 * 1024;

export const Typical: Story = {
  args: {
    name: 'kite3',
    ramUsed: 15.4 * GB,
    ramTotal: 24 * GB,
    cx: 0,
    fontSize: 13,
  },
};

export const HighUsage: Story = {
  args: {
    name: 'studio-rack',
    ramUsed: 110 * GB,
    ramTotal: 128 * GB,
    cx: 0,
    fontSize: 13,
  },
};

export const LongName: Story = {
  args: {
    name: 'my-very-long-hostname-here',
    ramUsed: 8 * GB,
    ramTotal: 16 * GB,
    cx: 0,
    fontSize: 13,
  },
};

export const ZeroRam: Story = {
  args: {
    name: 'offline-node',
    ramUsed: 0,
    ramTotal: 0,
    cx: 0,
    fontSize: 13,
  },
  name: 'No RAM data',
};
