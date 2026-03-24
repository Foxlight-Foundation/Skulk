import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { ConnectionBanner } from './ConnectionBanner';

const meta: Meta<typeof ConnectionBanner> = {
  title: 'Status/ConnectionBanner',
  component: ConnectionBanner,
  parameters: { layout: 'fullscreen' },
};

export default meta;
type Story = StoryObj<typeof ConnectionBanner>;

export const Disconnected: Story = {
  args: { connected: false },
};

export const Connected: Story = {
  args: { connected: true },
  name: 'Connected (hidden)',
};

export const Toggle: Story = {
  render: () => {
    const [connected, setConnected] = useState(true);
    return (
      <div style={{ background: '#111', minHeight: '100vh' }}>
        <ConnectionBanner connected={connected} />
        <div style={{ padding: 24 }}>
          <button
            style={{ padding: '8px 16px', background: connected ? '#ef4444' : '#22c55e', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontFamily: 'monospace' }}
            onClick={() => setConnected(!connected)}
          >
            {connected ? 'Simulate disconnect' : 'Simulate reconnect'}
          </button>
        </div>
      </div>
    );
  },
};
