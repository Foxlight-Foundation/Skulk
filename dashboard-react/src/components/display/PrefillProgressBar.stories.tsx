import { useEffect, useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { PrefillProgressBar, type PrefillProgress } from './PrefillProgressBar';

const meta: Meta<typeof PrefillProgressBar> = {
  title: 'Display/PrefillProgressBar',
  component: PrefillProgressBar,
  parameters: { layout: 'centered' },
  decorators: [
    (Story) => (
      <div style={{ width: 'min(400px, calc(100vw - 48px))', padding: 24, }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof PrefillProgressBar>;

export const HalfWay: Story = {
  args: {
    progress: { processed: 5000, total: 10000, startedAt: performance.now() - 3000 },
  },
};

export const NearlyDone: Story = {
  args: {
    progress: { processed: 9500, total: 10000, startedAt: performance.now() - 8000 },
  },
};

export const JustStarted: Story = {
  args: {
    progress: { processed: 100, total: 10000, startedAt: performance.now() - 500 },
  },
};

export const Animated: Story = {
  render: () => {
    const [progress, setProgress] = useState<PrefillProgress>({
      processed: 0,
      total: 8000,
      startedAt: performance.now(),
    });

    useEffect(() => {
      const interval = setInterval(() => {
        setProgress((prev) => {
          const next = Math.min(prev.processed + 200, prev.total);
          if (next >= prev.total) clearInterval(interval);
          return { ...prev, processed: next };
        });
      }, 100);
      return () => clearInterval(interval);
    }, []);

    return <PrefillProgressBar progress={progress} />;
  },
};
