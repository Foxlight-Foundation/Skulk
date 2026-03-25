import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { SegmentedControl } from './SegmentedControl';

const meta: Meta<typeof SegmentedControl> = {
  title: 'Common/SegmentedControl',
  component: SegmentedControl,
  parameters: { layout: 'centered' },
  decorators: [
    (Story) => (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 24, padding: 24, background: '#111' }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof SegmentedControl>;

export const StringOptions: Story = {
  render: () => {
    const [value, setValue] = useState('png');
    return <SegmentedControl options={['png', 'jpeg', 'webp']} value={value} onChange={setValue} />;
  },
};

export const ObjectOptions: Story = {
  render: () => {
    const [value, setValue] = useState('medium');
    return (
      <SegmentedControl
        options={[
          { value: 'low', label: 'Low' },
          { value: 'medium', label: 'Medium' },
          { value: 'high', label: 'High' },
        ]}
        value={value}
        onChange={setValue}
      />
    );
  },
};

export const WithDisabled: Story = {
  render: () => {
    const [value, setValue] = useState('pipeline');
    return (
      <SegmentedControl
        options={[
          { value: 'pipeline', label: 'Pipeline' },
          { value: 'tensor', label: 'Tensor' },
          { value: 'hybrid', label: 'Hybrid', disabled: true },
        ]}
        value={value}
        onChange={setValue}
      />
    );
  },
};

export const Sizes: Story = {
  render: () => {
    const [v1, setV1] = useState('a');
    const [v2, setV2] = useState('a');
    const [v3, setV3] = useState('a');
    const opts = ['a', 'b', 'c'];
    return (
      <>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ color: '#666', fontSize: 11, fontFamily: 'monospace', width: 24 }}>sm</span>
          <SegmentedControl options={opts} value={v1} onChange={setV1} size="sm" />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ color: '#666', fontSize: 11, fontFamily: 'monospace', width: 24 }}>md</span>
          <SegmentedControl options={opts} value={v2} onChange={setV2} size="md" />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ color: '#666', fontSize: 11, fontFamily: 'monospace', width: 24 }}>lg</span>
          <SegmentedControl options={opts} value={v3} onChange={setV3} size="lg" />
        </div>
      </>
    );
  },
};

export const TwoOptions: Story = {
  render: () => {
    const [value, setValue] = useState('low');
    return <SegmentedControl options={['low', 'high']} value={value} onChange={setValue} />;
  },
  name: 'Two options (toggle)',
};

export const ManyOptions: Story = {
  render: () => {
    const [value, setValue] = useState('1024x1024');
    return (
      <SegmentedControl
        options={['512x512', '768x768', '1024x1024', '1024x768', '1536x1024']}
        value={value}
        onChange={setValue}
        size="sm"
      />
    );
  },
};
