import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { ChatAttachments } from './ChatAttachments';
import type { ChatUploadedFile } from '../../types/chat';

const meta: Meta<typeof ChatAttachments> = {
  title: 'Chat/ChatAttachments',
  component: ChatAttachments,
  parameters: { layout: 'centered' },
  decorators: [(Story) => <div style={{ width: 'min(500px, calc(100vw - 48px))', padding: 16, }}><Story /></div>],
};

export default meta;
type Story = StoryObj<typeof ChatAttachments>;

const sampleFiles: ChatUploadedFile[] = [
  { id: '1', name: 'screenshot.png', type: 'image/png', size: 245000, preview: 'https://picsum.photos/seed/a/100/100' },
  { id: '2', name: 'notes.txt', type: 'text/plain', size: 1200, textContent: 'Hello world' },
  { id: '3', name: 'very-long-filename-report.pdf', type: 'application/pdf', size: 5200000 },
  { id: '4', name: 'recording.mp3', type: 'audio/mpeg', size: 3400000 },
];

export const Default: Story = {
  render: () => {
    const [files, setFiles] = useState(sampleFiles);
    return (
      <ChatAttachments
        files={files}
        onRemove={(id) => setFiles((f) => f.filter((x) => x.id !== id))}
      />
    );
  },
};

export const ReadOnly: Story = {
  args: { files: sampleFiles, readonly: true },
};

export const SingleImage: Story = {
  args: {
    files: [{ id: '1', name: 'photo.jpg', type: 'image/jpeg', size: 180000, preview: 'https://picsum.photos/seed/b/100/100' }],
    onRemove: () => {},
  },
};

export const Empty: Story = {
  args: { files: [] },
};
