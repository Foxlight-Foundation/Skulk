import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { Button } from './Button';

const meta: Meta<typeof Button> = {
  title: 'Common/Button',
  component: Button,
  parameters: { layout: 'centered' },
  decorators: [
    (Story) => (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 24, padding: 24, maxWidth: '100%', flexWrap: 'wrap' }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof Button>;

const PlusIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);

const TrashIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
  </svg>
);

const RefreshIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="23 4 23 10 17 10" /><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
  </svg>
);

export const Variants: Story = {
  render: () => (
    <>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <Button variant="primary">Primary outline</Button>
        <Button variant="solid">Save changes</Button>
        <Button variant="approve">Approve action</Button>
        <Button variant="outline">Outline</Button>
        <Button variant="ghost">Ghost</Button>
        <Button variant="danger">Danger</Button>
      </div>
    </>
  ),
};

/** Screen-specific commit actions retain their solid treatment. */
export const SolidActions: Story = {
  render: () => <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
    <Button variant="solid">Launch</Button>
    <Button variant="solid">Save changes</Button>
    <Button variant="approve">Approve action</Button>
    <Button variant="solid" disabled>Launch unavailable</Button>
    <Button variant="solid" loading>Launching</Button>
    <Button variant="approve" loading>Approving</Button>
  </div>,
};

export const Sizes: Story = {
  render: () => (
    <>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <Button variant="primary" size="sm">Small</Button>
        <Button variant="primary" size="md">Medium</Button>
        <Button variant="primary" size="lg">Large</Button>
      </div>
    </>
  ),
};

export const WithIcons: Story = {
  render: () => (
    <>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <Button variant="primary"><PlusIcon /> New Chat</Button>
        <Button variant="outline"><RefreshIcon /> Refresh</Button>
        <Button variant="danger"><TrashIcon /> Delete</Button>
      </div>
    </>
  ),
};

export const IconOnly: Story = {
  render: () => (
    <>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <Button variant="primary" icon size="sm" aria-label="Add"><PlusIcon /></Button>
        <Button variant="outline" icon aria-label="Refresh"><RefreshIcon /></Button>
        <Button variant="ghost" icon aria-label="Delete"><TrashIcon /></Button>
        <Button variant="danger" icon aria-label="Delete"><TrashIcon /></Button>
      </div>
    </>
  ),
};

export const States: Story = {
  render: () => (
    <>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <Button variant="primary">Normal</Button>
        <Button variant="primary" disabled>Disabled</Button>
        <Button variant="primary" loading>Loading</Button>
      </div>
    </>
  ),
};

export const Block: Story = {
  render: () => (
    <div style={{ width: '100%', maxWidth: 300 }}>
      <Button variant="solid" block>▸ Launch Model</Button>
    </div>
  ),
};

export const InteractiveLoading: Story = {
  render: () => {
    const [loading, setLoading] = useState(false);
    return (
      <Button
        variant="solid"
        loading={loading}
        onClick={() => {
          setLoading(true);
          setTimeout(() => setLoading(false), 2000);
        }}
      >
        ▸ Launch
      </Button>
    );
  },
  name: 'Interactive (click to load)',
};
