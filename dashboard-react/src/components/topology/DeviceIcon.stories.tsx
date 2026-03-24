import type { Meta, StoryObj } from '@storybook/react-vite';
import { DeviceIcon } from './DeviceIcon';

const SvgWrap = ({ children, width, height }: { children: React.ReactNode; width: number; height: number }) => (
  <svg width={width + 20} height={height + 20} style={{ background: '#000' }}>
    <g transform="translate(10, 10)">{children}</g>
  </svg>
);

const meta: Meta<typeof DeviceIcon> = {
  title: 'Topology/DeviceIcon',
  component: DeviceIcon,
  parameters: { layout: 'centered' },
  argTypes: {
    model: { control: 'select', options: ['macbook-pro', 'mac-studio', 'mac-mini', 'unknown'] },
    ramPercent: { control: { type: 'range', min: 0, max: 100, step: 1 } },
    wireColor: { control: 'color' },
    strokeWidth: { control: { type: 'range', min: 0.5, max: 4, step: 0.5 } },
  },
  decorators: [
    (Story, ctx) => {
      const w = ctx.args.width || 160;
      const h = ctx.args.height || 120;
      return <SvgWrap width={w} height={h}><Story /></SvgWrap>;
    },
  ],
};

export default meta;
type Story = StoryObj<typeof DeviceIcon>;

export const MacBookPro: Story = {
  args: { model: 'macbook-pro', ramPercent: 64, width: 180, height: 140 },
};

export const MacStudio: Story = {
  args: { model: 'mac-studio', ramPercent: 64, width: 150, height: 120 },
};

export const MacMini: Story = {
  args: { model: 'mac-mini', ramPercent: 45, width: 150, height: 100 },
};

export const UnknownDevice: Story = {
  args: { model: 'unknown', ramPercent: 0, width: 120, height: 120 },
};

export const MacBookProEmpty: Story = {
  args: { model: 'macbook-pro', ramPercent: 0, width: 180, height: 140 },
  name: 'MacBook Pro (0% RAM)',
};

export const MacBookProFull: Story = {
  args: { model: 'macbook-pro', ramPercent: 100, width: 180, height: 140 },
  name: 'MacBook Pro (100% RAM)',
};

export const MacStudioHighlighted: Story = {
  args: { model: 'mac-studio', ramPercent: 80, width: 150, height: 120, wireColor: 'rgba(255,215,0,1)', strokeWidth: 3 },
  name: 'Mac Studio (highlighted)',
};
