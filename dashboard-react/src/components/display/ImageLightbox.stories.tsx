import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { ImageLightbox } from './ImageLightbox';

const meta: Meta<typeof ImageLightbox> = {
  title: 'Display/ImageLightbox',
  component: ImageLightbox,
  parameters: { layout: 'fullscreen' },
};

export default meta;
type Story = StoryObj<typeof ImageLightbox>;

const SAMPLE_IMAGE = 'https://picsum.photos/seed/skulk/1200/800';

export const Open: Story = {
  args: { src: SAMPLE_IMAGE, onClose: () => {} },
};

export const Interactive: Story = {
  render: () => {
    const [src, setSrc] = useState<string | null>(null);
    return (
      <div style={{ padding: 24, background: '#111', minHeight: '100vh' }}>
        <button
          style={{ padding: '8px 16px', background: '#333', color: '#FFD700', border: '1px solid rgba(255,215,0,0.3)', borderRadius: 6, cursor: 'pointer', fontFamily: 'monospace' }}
          onClick={() => setSrc(SAMPLE_IMAGE)}
        >
          Open lightbox
        </button>
        <ImageLightbox src={src} onClose={() => setSrc(null)} />
      </div>
    );
  },
};
