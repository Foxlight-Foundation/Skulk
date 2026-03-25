import type { Meta, StoryObj } from '@storybook/react-vite';
import { InfoTooltip } from './InfoTooltip';

const meta: Meta<typeof InfoTooltip> = {
  title: 'Common/InfoTooltip',
  component: InfoTooltip,
  parameters: { layout: 'centered' },
  decorators: [
    (Story) => (
      <div style={{ padding: 100, background: '#111', display: 'flex', gap: 40, alignItems: 'center' }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof InfoTooltip>;

export const Default: Story = {
  args: {
    content: 'Pipeline splits the model into sequential stages across devices. Lower network overhead.',
  },
};

export const Placements: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: 40, alignItems: 'center' }}>
      <InfoTooltip content="Top placement (default)" placement="top" />
      <InfoTooltip content="Right placement" placement="right" />
      <InfoTooltip content="Bottom placement" placement="bottom" />
      <InfoTooltip content="Left placement" placement="left" />
    </div>
  ),
};

export const RichContent: Story = {
  args: {
    content: (
      <div>
        <strong style={{ color: '#FFD700' }}>Tensor Parallelism</strong>
        <br />
        Splits each layer across devices. Best with high-bandwidth connections like Thunderbolt.
        <br /><br />
        <span style={{ color: '#666' }}>Requires RDMA-capable interfaces.</span>
      </div>
    ),
  },
};

export const CustomTrigger: Story = {
  render: () => (
    <InfoTooltip content="This is a custom trigger element">
      <span style={{ color: '#FFD700', fontSize: 13, fontFamily: 'monospace', cursor: 'help', textDecoration: 'underline dotted' }}>
        What is this?
      </span>
    </InfoTooltip>
  ),
};

export const InContext: Story = {
  render: () => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontFamily: 'monospace', fontSize: 12, color: '#999' }}>
      <span style={{ color: '#FFD700' }}>Pipeline</span>
      <InfoTooltip content="Pipeline splits the model into sequential stages across devices. Lower network overhead." />
      <span style={{ margin: '0 8px', color: '#333' }}>|</span>
      <span style={{ color: '#FFD700' }}>MLX Ring</span>
      <InfoTooltip content="Ring: standard networking. Works over any connection (Wi-Fi, Ethernet, Thunderbolt)." />
    </div>
  ),
  name: 'In context (next to badges)',
};
