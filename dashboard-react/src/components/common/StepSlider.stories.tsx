import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { StepSlider } from './StepSlider';

const meta: Meta<typeof StepSlider> = {
  title: 'Common/StepSlider',
  component: StepSlider,
  parameters: { layout: 'centered' },
  decorators: [
    (Story) => (
      <div style={{ width: 'min(300px, calc(100vw - 48px))', padding: 24, }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof StepSlider>;

export const MinimumDevices: Story = {
  render: () => {
    const [value, setValue] = useState(1);
    return (
      <div>
        <div style={{ fontSize: 12, fontFamily: 'monospace', color: 'inherit', marginBottom: 8 }}>
          Minimum Devices:
        </div>
        <StepSlider options={[1, 2, 3]} value={value} onChange={setValue} />
      </div>
    );
  },
};

export const FiveSteps: Story = {
  render: () => {
    const [value, setValue] = useState(3);
    return <StepSlider options={[1, 2, 3, 4, 5]} value={value} onChange={setValue} />;
  },
};

export const StringLabels: Story = {
  render: () => {
    const [value, setValue] = useState('medium');
    return (
      <StepSlider
        options={[
          { value: 'low', label: 'Low' },
          { value: 'medium', label: 'Med' },
          { value: 'high', label: 'High' },
        ]}
        value={value}
        onChange={setValue}
      />
    );
  },
};

export const TwoOptions: Story = {
  render: () => {
    const [value, setValue] = useState(1);
    return <StepSlider options={[1, 2]} value={value} onChange={setValue} />;
  },
  name: 'Two options',
};

export const ManyOptions: Story = {
  render: () => {
    const [value, setValue] = useState(4);
    return (
      <div style={{ width: 'min(400px, calc(100vw - 48px))' }}>
        <StepSlider
          options={[1, 2, 3, 4, 5, 6, 7, 8]}
          value={value}
          onChange={setValue}
        />
      </div>
    );
  },
  name: 'Many options (8)',
};
