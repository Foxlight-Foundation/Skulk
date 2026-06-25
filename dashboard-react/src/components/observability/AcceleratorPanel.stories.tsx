import type { Meta, StoryObj } from '@storybook/react-vite';
import { AcceleratorPanel } from './AcceleratorPanel';

const Frame = ({ children }: { children: React.ReactNode }) => (
  <div
    style={{
      width: 260,
      padding: 16,
      background: '#0b0b0d',
      border: '1px solid #2a2a2e',
      borderRadius: 8,
    }}
  >
    {children}
  </div>
);

const meta: Meta<typeof AcceleratorPanel> = {
  title: 'Observability/AcceleratorPanel',
  component: AcceleratorPanel,
  parameters: { layout: 'centered' },
  decorators: [(Story) => <Frame><Story /></Frame>],
};

export default meta;
type Story = StoryObj<typeof AcceleratorPanel>;

/** Apple Silicon: unified memory, so VRAM is "not reported" (no discrete pool). */
export const Apple: Story = {
  args: {
    accelerator: {
      vendor: 'apple',
      name: 'Apple GPU',
      utilizationRatio: 0.07,
      vramTotalBytes: null,
      vramUsedBytes: null,
      powerWatts: 0.07,
      temperatureCelsius: 39,
      clockMhz: null,
    },
  },
};

/** AMD Strix Halo: a discrete 64 GiB GPU VRAM pool carved from unified memory. */
export const AmdStrixHalo: Story = {
  name: 'AMD (Strix Halo)',
  args: {
    accelerator: {
      vendor: 'amd',
      name: 'AMD GPU',
      utilizationRatio: 0.0,
      vramTotalBytes: 68719476736,
      vramUsedBytes: 163233792,
      powerWatts: 7.05,
      temperatureCelsius: 35,
      clockMhz: 600,
    },
  },
};

/** AMD under load, to show the VRAM bar filled. */
export const AmdBusy: Story = {
  name: 'AMD (busy)',
  args: {
    accelerator: {
      vendor: 'amd',
      name: 'AMD GPU',
      utilizationRatio: 0.93,
      vramTotalBytes: 68719476736,
      vramUsedBytes: 41231686041,
      powerWatts: 96.4,
      temperatureCelsius: 71,
      clockMhz: 2900,
    },
  },
};
