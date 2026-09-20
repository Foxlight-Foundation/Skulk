import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { expect, userEvent, within } from 'storybook/test';
import { Select } from './Select';
import { Gallery, GalleryHeader, Specimen } from '../../../.storybook/Gallery';

function SelectorExample() {
  const [value, setValue] = useState('automatic');
  return <Gallery><GalleryHeader title="Selectors">Chat menu styling shared across dashboard controls.</GalleryHeader>
    <Specimen><label>Mode <Select aria-label="Mode" value={value} onValueChange={setValue}>
      <option value="automatic">Automatic</option><option disabled value="unavailable">Unavailable</option>
      <option value="manual">Manual</option><option value="long">A deliberately long translated option that must wrap inside the menu without overflowing the viewport</option>
    </Select></label><Select aria-label="Disabled mode" value="locked" disabled><option value="locked">Managed by configuration</option></Select>
    <output aria-label="Selected mode">{value}</output></Specimen>
  </Gallery>;
}
const meta = { title: 'Design System/Select', component: SelectorExample, parameters: { layout: 'fullscreen' } } satisfies Meta;
export default meta;
type Story = StoryObj<typeof meta>;
export const KeyboardSelection: Story = { play: async ({ canvasElement }) => {
  const canvas = within(canvasElement);
  const body = within(canvasElement.ownerDocument.body);
  const trigger = canvas.getByRole('button', { name: 'Mode' });
  await userEvent.click(trigger);
  await expect(body.getByRole('listbox')).toBeVisible();
  await expect(canvas.getByLabelText('Selected mode')).toHaveTextContent('automatic');
  await userEvent.keyboard('{ArrowDown}{Enter}');
  await expect(canvas.getByLabelText('Selected mode')).toHaveTextContent('manual');
  await expect(trigger).toHaveFocus();
  await userEvent.click(trigger);
  await userEvent.keyboard('{Home}{Escape}');
  await expect(body.queryByRole('listbox')).not.toBeInTheDocument();
  await expect(canvas.getByLabelText('Selected mode')).toHaveTextContent('manual');
  await expect(trigger).toHaveFocus();
  await expect(canvas.getByRole('button', { name: 'Disabled mode' })).toBeDisabled();
} };
export const OpenMenu: Story = { play: async ({ canvasElement }) => {
  await userEvent.click(within(canvasElement).getByRole('button', { name: 'Mode' }));
} };
