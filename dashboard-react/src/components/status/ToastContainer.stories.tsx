import type { Meta, StoryObj } from '@storybook/react-vite';
import { Button } from '../common/Button';
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
    <Button
      onClick={() => addToast({ type: 'success', message: 'Model launched successfully' })}
    >
      Success toast
    </Button>
    <Button
      onClick={() => addToast({ type: 'error', message: 'Failed to connect to example node' })}
    >
      Error toast
    </Button>
    <Button
      onClick={() => addToast({ type: 'warning', message: 'Example node memory usage above 90%' })}
    >
      Warning toast
    </Button>
    <Button
      onClick={() => addToast({ type: 'info', message: 'Download started for Qwen3-30B-A3B-4bit' })}
    >
      Info toast
    </Button>
    <Button
      onClick={() => addToast({ type: 'error', message: 'This toast will not auto-dismiss', persistent: true })}
    >
      Persistent toast
    </Button>
  </div>
);

export const Interactive: Story = {
  render: () => (
    <div style={{ height: '100vh', }}>
      <TriggerPanel />
      <ToastContainer />
    </div>
  ),
};
