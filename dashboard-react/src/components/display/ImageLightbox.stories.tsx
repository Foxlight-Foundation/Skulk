import { Button } from '../common/Button';
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
      <div style={{ padding: 24, minHeight: '100vh' }}>
        <Button variant="primary"
          onClick={() => setSrc(SAMPLE_IMAGE)}
        >
          Open lightbox
        </Button>
        <ImageLightbox src={src} onClose={() => setSrc(null)} />
      </div>
    );
  },
};
