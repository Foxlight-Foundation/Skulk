import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { expect, userEvent, within } from 'storybook/test';
import { ReadyModelSelect } from './ReadyModelSelect';
import { Gallery, GalleryHeader, Specimen } from '../../../.storybook/Gallery';

/** Local controlled fixture: selection never starts a request. */
function ModelChooserExample() {
  const [selected, setSelected] = useState('skulk/steward');
  return <Gallery><GalleryHeader title="Chat model menu">Fabric and ready-model groups share one controlled selection. Opening, focusing and dismissing the menu never submits a prompt.</GalleryHeader>
    <Specimen style={{ width: 'min(100%, 440px)', marginTop: 280 }}><ReadyModelSelect value={selected} onChange={setSelected} fabricEnabled models={[{ modelId: 'example/Chat-8B' }, { modelId: 'example/Reasoning-32B' }]} /></Specimen>
  </Gallery>;
}
const meta = { title: 'Chat/ReadyModelSelect', component: ModelChooserExample, parameters: { layout: 'fullscreen' } } satisfies Meta;
export default meta;
type Story = StoryObj<typeof meta>;
export const OpenMenu: Story = { render: () => <ModelChooserExample />, play: async ({ canvasElement }) => {
  const canvas = within(canvasElement);
  await userEvent.click(await canvas.findByRole('button', { name: 'Select chat model' }));
  await expect(within(canvasElement.ownerDocument.body).getByRole('listbox')).toBeVisible();
} };
