import type { Meta, StoryObj } from '@storybook/react-vite';
import { TokenHeatmap, type TokenData } from './TokenHeatmap';

const meta: Meta<typeof TokenHeatmap> = {
  title: 'Display/TokenHeatmap',
  component: TokenHeatmap,
  parameters: { layout: 'centered' },
  decorators: [
    (Story) => (
      <div style={{ maxWidth: 600, padding: 24, color: '#e5e5e5', fontFamily: 'monospace', fontSize: 14 }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof TokenHeatmap>;

const sampleTokens: TokenData[] = [
  { token: 'The', probability: 0.95, logprob: -0.05, topLogprobs: [{ token: 'The', logprob: -0.05 }, { token: 'A', logprob: -3.1 }] },
  { token: ' quick', probability: 0.82, logprob: -0.2, topLogprobs: [{ token: ' quick', logprob: -0.2 }, { token: ' fast', logprob: -1.8 }] },
  { token: ' brown', probability: 0.45, logprob: -0.8, topLogprobs: [{ token: ' brown', logprob: -0.8 }, { token: ' red', logprob: -1.2 }, { token: ' lazy', logprob: -1.5 }] },
  { token: ' fox', probability: 0.91, logprob: -0.09, topLogprobs: [{ token: ' fox', logprob: -0.09 }] },
  { token: ' jumped', probability: 0.12, logprob: -2.1, topLogprobs: [{ token: ' jumps', logprob: -0.8 }, { token: ' jumped', logprob: -2.1 }, { token: ' ran', logprob: -2.5 }] },
  { token: ' over', probability: 0.88, logprob: -0.13, topLogprobs: [{ token: ' over', logprob: -0.13 }] },
  { token: ' the', probability: 0.97, logprob: -0.03, topLogprobs: [{ token: ' the', logprob: -0.03 }] },
  { token: ' lazy', probability: 0.72, logprob: -0.33, topLogprobs: [{ token: ' lazy', logprob: -0.33 }, { token: ' sleeping', logprob: -1.5 }] },
  { token: ' dog', probability: 0.15, logprob: -1.9, topLogprobs: [{ token: ' cat', logprob: -1.2 }, { token: ' dog', logprob: -1.9 }, { token: ' bear', logprob: -2.8 }] },
  { token: '.', probability: 0.93, logprob: -0.07, topLogprobs: [{ token: '.', logprob: -0.07 }] },
];

export const Default: Story = {
  args: { tokens: sampleTokens },
};

export const WithRegenerate: Story = {
  args: {
    tokens: sampleTokens,
    onRegenerateFrom: (i: number) => alert(`Regenerate from token index ${i}`),
  },
};

export const HighConfidence: Story = {
  args: {
    tokens: sampleTokens.map((t) => ({ ...t, probability: 0.9 + Math.random() * 0.1 })),
  },
  name: 'All high confidence',
};

export const LowConfidence: Story = {
  args: {
    tokens: sampleTokens.map((t) => ({ ...t, probability: 0.05 + Math.random() * 0.15 })),
  },
  name: 'All low confidence',
};
