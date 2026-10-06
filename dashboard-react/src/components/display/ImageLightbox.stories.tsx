import { galleryImage } from '../../../.storybook/media';
import { expect, userEvent, within } from 'storybook/test';
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



export const Open: Story = {
  args: { src: galleryImage, onClose: () => {} },
};

export const Interactive: Story = {
  render: () => {
    const [src, setSrc] = useState<string | null>(null);
    return (
      <div style={{ padding: 24, minHeight: '100vh' }}>
        <Button variant="primary"
          onClick={() => setSrc(galleryImage)}
        >
          Open lightbox
        </Button>
        <ImageLightbox src={src} onClose={() => setSrc(null)} />
      </div>
    );
  },
};

export const Keyboard: Story = {
  ...Interactive,
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const opener = canvas.getByRole('button', { name: 'Open lightbox' });
    await userEvent.click(opener);
    const dialog = await canvas.findByRole('dialog', { name: 'Full size preview' });
    await expect(dialog).toHaveFocus();
    await userEvent.tab();
    await expect(canvas.getByRole('button', { name: 'Download image' })).toHaveFocus();
    await userEvent.tab({ shift: true });
    await expect(canvas.getByRole('button', { name: 'Close lightbox' })).toHaveFocus();
    await userEvent.keyboard('{Escape}');
    await expect(canvas.queryByRole('dialog')).not.toBeInTheDocument();
    await expect(opener).toHaveFocus();
  },
};
