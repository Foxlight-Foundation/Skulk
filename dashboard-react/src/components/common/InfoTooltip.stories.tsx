import type { Meta, StoryObj } from '@storybook/react-vite';
import { InfoTooltip } from './InfoTooltip';

const meta: Meta<typeof InfoTooltip> = {
  title: 'Common/InfoTooltip',
  component: InfoTooltip,
  parameters: { layout: 'centered' },
  decorators: [
    (Story) => (
      <div style={{ padding: 24, maxWidth: '100%', display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap' }}>
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
    <div style={{ display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap' }}>
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
        <strong style={{ color: 'inherit' }}>Tensor Parallelism</strong>
        <br />
        Splits each layer across devices. Best with high-bandwidth connections like Thunderbolt.
        <br /><br />
        <span style={{ color: 'inherit' }}>Requires RDMA-capable interfaces.</span>
      </div>
    ),
  },
};

export const CustomTrigger: Story = {
  render: () => (
    <InfoTooltip content="This is a custom trigger element">
      <span style={{ color: 'inherit', fontSize: 13, fontFamily: 'monospace', cursor: 'help', textDecoration: 'underline dotted' }}>
        What is this?
      </span>
    </InfoTooltip>
  ),
};

export const InContext: Story = {
  render: () => (
    <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, fontFamily: 'monospace', fontSize: 12, color: 'inherit' }}>
      <span style={{ color: 'inherit' }}>Pipeline</span>
      <InfoTooltip content="Pipeline splits the model into sequential stages across devices. Lower network overhead." />
      <span style={{ margin: '0 8px', color: 'inherit' }}>|</span>
      <span style={{ color: 'inherit' }}>MLX Ring</span>
      <InfoTooltip content="Ring: standard networking. Works over any connection (Wi-Fi, Ethernet, Thunderbolt)." />
    </div>
  ),
  name: 'In context (next to badges)',
};
