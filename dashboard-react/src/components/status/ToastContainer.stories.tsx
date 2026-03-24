import type { Meta, StoryObj } from '@storybook/react-vite';
import { ToastContainer } from './ToastContainer';
import { addToast } from '../../hooks/useToast';

const meta: Meta<typeof ToastContainer> = {
  title: 'Status/ToastContainer',
  component: ToastContainer,
  parameters: { layout: 'fullscreen' },
};

export default meta;
type Story = StoryObj<typeof ToastContainer>;

const TriggerPanel = () => (
  <div style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 8, maxWidth: 300 }}>
    <button
      style={{ padding: '8px 16px', background: '#22c55e', color: '#000', border: 'none', borderRadius: 6, cursor: 'pointer', fontFamily: 'monospace' }}
      onClick={() => addToast({ type: 'success', message: 'Model launched successfully' })}
    >
      Success toast
    </button>
    <button
      style={{ padding: '8px 16px', background: '#ef4444', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontFamily: 'monospace' }}
      onClick={() => addToast({ type: 'error', message: 'Failed to connect to node kite1' })}
    >
      Error toast
    </button>
    <button
      style={{ padding: '8px 16px', background: '#eab308', color: '#000', border: 'none', borderRadius: 6, cursor: 'pointer', fontFamily: 'monospace' }}
      onClick={() => addToast({ type: 'warning', message: 'Node kite3 memory usage above 90%' })}
    >
      Warning toast
    </button>
    <button
      style={{ padding: '8px 16px', background: '#3b82f6', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontFamily: 'monospace' }}
      onClick={() => addToast({ type: 'info', message: 'Download started for Qwen3-30B-A3B-4bit' })}
    >
      Info toast
    </button>
    <button
      style={{ padding: '8px 16px', background: '#555', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontFamily: 'monospace' }}
      onClick={() => addToast({ type: 'error', message: 'This toast will not auto-dismiss', persistent: true })}
    >
      Persistent toast
    </button>
  </div>
);

export const Interactive: Story = {
  render: () => (
    <div style={{ height: '100vh', background: '#111' }}>
      <TriggerPanel />
      <ToastContainer />
    </div>
  ),
};
