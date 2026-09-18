import type { Meta, StoryObj } from '@storybook/react-vite';
import { HuggingFaceResultItem } from './HuggingFaceResultItem';

const meta: Meta<typeof HuggingFaceResultItem> = {
  title: 'Models/HuggingFaceResultItem',
  component: HuggingFaceResultItem,
  parameters: { layout: 'centered' },
  decorators: [
    (Story) => (
      <div style={{ width: 'min(500px, calc(100vw - 48px))', padding: 8, borderRadius: 8 }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof HuggingFaceResultItem>;

export const NotAdded: Story = {
  args: {
    model: { id: 'mlx-community/Qwen3-30B-A3B-4bit', author: 'mlx-community', downloads: 125000, likes: 340, last_modified: '2026-03-20', tags: [] },
    isAdded: false,
    isAdding: false,
    onAdd: () => {},
    onSelect: () => {},
  },
};

export const Added: Story = {
  args: {
    model: { id: 'mlx-community/Llama-3.1-8B-4bit', author: 'mlx-community', downloads: 2500000, likes: 1200, last_modified: '2026-03-15', tags: [] },
    isAdded: true,
    isAdding: false,
    onAdd: () => {},
    onSelect: () => {},
    downloadedOnNodes: ['kite3'],
  },
};

export const Adding: Story = {
  args: {
    model: { id: 'deepseek-ai/DeepSeek-V3-0324', author: 'deepseek-ai', downloads: 50000, likes: 890, last_modified: '2026-03-24', tags: [] },
    isAdded: false,
    isAdding: true,
    onAdd: () => {},
    onSelect: () => {},
  },
};
