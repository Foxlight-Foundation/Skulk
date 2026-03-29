import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { ChatSidebar } from './ChatSidebar';
import type { Conversation } from '../../types/chat';

const now = Date.now();

const sampleConversations: Conversation[] = [
  { id: '1', name: 'Distributed inference setup', modelId: 'qwen3-30b-4bit', createdAt: now - 2000, updatedAt: now - 1000, messages: [] },
  { id: '2', name: 'Python async patterns', modelId: 'llama-8b-4bit', createdAt: now - 7200000, updatedAt: now - 3600000, messages: [] },
  { id: '3', name: 'Debugging RDMA connections', modelId: 'qwen3-30b-4bit', createdAt: now - 90000000, updatedAt: now - 86400000, messages: [] },
  { id: '4', name: 'MLX optimization tips', modelId: 'qwen3-30b-4bit', createdAt: now - 200000000, updatedAt: now - 172800000, messages: [] },
  { id: '5', name: 'Kubernetes cluster design', modelId: 'llama-8b-4bit', createdAt: now - 700000000, updatedAt: now - 604800000, messages: [] },
];

const meta: Meta<typeof ChatSidebar> = {
  title: 'Chat/ChatSidebar',
  component: ChatSidebar,
  parameters: { layout: 'fullscreen' },
  decorators: [(Story) => <div style={{ height: '100vh', display: 'flex' }}><Story /></div>],
};

export default meta;
type Story = StoryObj<typeof ChatSidebar>;

export const Default: Story = {
  render: () => {
    const [active, setActive] = useState<string | null>('1');
    const [convs, setConvs] = useState(sampleConversations);
    return (
      <ChatSidebar
        conversations={convs}
        activeId={active}
        onNewChat={() => alert('New chat')}
        onSelectConversation={setActive}
        onRenameConversation={(id, name) => setConvs((c) => c.map((x) => x.id === id ? { ...x, name } : x))}
        onDeleteConversation={(id) => setConvs((c) => c.filter((x) => x.id !== id))}
        onDeleteAllConversations={() => setConvs([])}
      />
    );
  },
};

export const Empty: Story = {
  args: {
    conversations: [],
    activeId: null,
    onNewChat: () => {},
    onSelectConversation: () => {},
    onRenameConversation: () => {},
    onDeleteConversation: () => {},
    onDeleteAllConversations: () => {},
  },
};
