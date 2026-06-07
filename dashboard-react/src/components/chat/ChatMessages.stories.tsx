import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { ChatMessages } from './ChatMessages';
import type { ChatMessage } from '../../types/chat';

const now = Date.now();

const sampleMessages: ChatMessage[] = [
  {
    id: '1', role: 'user', content: 'How do I set up distributed inference with Skulk?', timestamp: now - 60000,
    attachments: [{ id: 'a1', name: 'config.json', type: 'application/json' }],
  },
  {
    id: '2', role: 'assistant', timestamp: now - 55000,
    content: `To set up distributed inference with **Skulk**, you need to:\n\n1. Install Skulk on all nodes:\n\`\`\`bash\nuv sync\n\`\`\`\n\n2. Start the cluster:\n\`\`\`bash\nuv run skulk\n\`\`\`\n\nThe system will automatically discover nodes and distribute the model using pipeline parallelism.\n\n> **Note:** For best performance with multiple Apple Silicon devices, use Thunderbolt connections and tensor parallelism.`,
    ttftMs: 245, tps: 42.3,
    thinkingContent: 'The user wants to set up distributed inference. I should explain the basic setup steps and mention the key configuration options for Apple Silicon clusters.',
    tokens: [
      { token: 'To', probability: 0.95, logprob: -0.05 },
      { token: ' set', probability: 0.88, logprob: -0.13 },
      { token: ' up', probability: 0.92, logprob: -0.08 },
      { token: ' distributed', probability: 0.45, logprob: -0.8, topLogprobs: [{ token: ' distributed', logprob: -0.8 }, { token: ' multi', logprob: -1.2 }] },
      { token: ' inference', probability: 0.85, logprob: -0.16 },
    ],
  },
  {
    id: '3', role: 'user', content: 'What about memory requirements for a 70B model?', timestamp: now - 30000,
  },
  {
    id: '4', role: 'assistant', timestamp: now - 25000,
    content: 'A 70B model at 4-bit quantization requires approximately **35GB** of memory. With a 3-node cluster of Mac Studios (64GB each), you have plenty of headroom.\n\nThe formula is: $M = P \\times Q$ where $P$ is parameters and $Q$ is the quantization factor (0.5 for 4-bit).',
    ttftMs: 180, tps: 45.1,
  },
];

const meta: Meta<typeof ChatMessages> = {
  title: 'Chat/ChatMessages',
  component: ChatMessages,
  parameters: { layout: 'fullscreen' },
  decorators: [
    (Story) => (
      <div style={{ height: '100vh', overflow: 'auto', background: '#000' }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof ChatMessages>;

export const Default: Story = {
  args: {
    messages: sampleMessages,
    onDelete: (id) => console.log('Delete', id),
    onEdit: (id, content) => console.log('Edit', id, content),
    onRegenerate: () => console.log('Regenerate'),
  },
};

export const Empty: Story = {
  args: { messages: [] },
};

export const Streaming: Story = {
  args: {
    messages: [sampleMessages[0], sampleMessages[1]],
    streamingContent: 'The memory requirements depend on several factors including the model architecture, quantization level, and',
    isLoading: true,
  },
};

export const WithImages: Story = {
  args: {
    messages: [
      { id: '1', role: 'user', content: 'Generate a mountain landscape', timestamp: now - 10000 },
      {
        id: '2', role: 'assistant', content: 'Here is your generated image:', timestamp: now - 5000,
        generatedImages: ['https://picsum.photos/seed/mountain/512/512'],
      },
    ],
  },
};
