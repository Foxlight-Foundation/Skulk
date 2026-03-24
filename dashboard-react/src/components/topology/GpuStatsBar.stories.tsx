import type { Meta, StoryObj } from '@storybook/react-vite';
import { GpuStatsBar } from './GpuStatsBar';

const SvgWrap = ({ children, width, height }: { children: React.ReactNode; width: number; height: number }) => (
  <svg width={width + 20} height={height + 20} style={{ background: '#000' }}>
    <g transform="translate(10, 10)">{children}</g>
  </svg>
);

const meta: Meta<typeof GpuStatsBar> = {
  title: 'Topology/GpuStatsBar',
  component: GpuStatsBar,
  parameters: { layout: 'centered' },
  argTypes: {
    gpuPercent: { control: { type: 'range', min: 0, max: 100, step: 1 } },
    gpuTemp: { control: { type: 'range', min: 30, max: 100, step: 1 } },
    sysPower: { control: { type: 'range', min: 0, max: 100, step: 1 } },
    width: { control: { type: 'range', min: 16, max: 60, step: 2 } },
    height: { control: { type: 'range', min: 40, max: 150, step: 5 } },
  },
  decorators: [
    (Story, ctx) => {
      const w = ctx.args.width || 28;
      const h = ctx.args.height || 100;
      return <SvgWrap width={w} height={h}><Story /></SvgWrap>;
    },
  ],
};

export default meta;
type Story = StoryObj<typeof GpuStatsBar>;

export const Cool: Story = {
  args: { gpuPercent: 15, gpuTemp: 39, sysPower: 11, width: 28, height: 95 },
  name: 'Cool (39°C)',
};

export const Warm: Story = {
  args: { gpuPercent: 55, gpuTemp: 55, sysPower: 35, width: 28, height: 95 },
  name: 'Warm (55°C)',
};

export const Hot: Story = {
  args: { gpuPercent: 90, gpuTemp: 78, sysPower: 65, width: 28, height: 95 },
  name: 'Hot (78°C)',
};

export const Idle: Story = {
  args: { gpuPercent: 0, gpuTemp: 35, sysPower: 5, width: 28, height: 95 },
};

export const NoData: Story = {
  args: { gpuPercent: 0, gpuTemp: NaN, sysPower: null, width: 28, height: 95 },
  name: 'No data',
};
