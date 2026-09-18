import { expect, userEvent, within } from 'storybook/test';
import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import styled from 'styled-components';
import { FiCopy, FiSearch } from 'react-icons/fi';
import { Button } from './Button';
import { Field } from './Field';
import { SegmentedControl } from './SegmentedControl';
import { CollapsibleSection, Monogram, SectionLabel, StatusPill, Surface } from './Surfaces';

const Gallery = styled.div`
  padding: 24px; max-width: 980px; display: grid; gap: 24px;
  h1 { font-size: 34px; letter-spacing: -.025em; color: ${({ theme }) => theme.colors.text}; }
  h2 { font-size: 20px; color: ${({ theme }) => theme.colors.text}; }
`;
const Row = styled.div`display: flex; flex-wrap: wrap; gap: 12px; align-items: center;`;

/** Interactive production primitives with deterministic local-only state. */
function FoundationGallery() {
  const [tab, setTab] = useState('Local network');
  const [open, setOpen] = useState(true);
  const [search, setSearch] = useState('');
  return <Gallery>
    <div><SectionLabel>Skulk dashboard</SectionLabel><h1>Studio Night</h1><p>Operator components · Night and Noon Ridge</p></div>
    <Surface><h2>Actions</h2><Row>
      <Button variant="primary">Download</Button><Button variant="solid">Launch</Button>
      <Button variant="approve">Approve</Button><Button>Not now</Button>
      <Button variant="danger">Uninstall</Button><Button icon aria-label="Copy"><FiCopy /></Button>
      <Button disabled>Unavailable</Button><Button loading>Loading</Button>
    </Row></Surface>
    <Surface><h2>Fields and choices</h2><Row>
      <Field aria-label="Search models" placeholder="Search supported models…" icon={<FiSearch />} value={search} onChange={e => setSearch(e.target.value)} />
      <Field aria-label="Unavailable field" placeholder="Unavailable" disabled />
      <SegmentedControl options={['Local network', 'Tailscale']} value={tab} onChange={setTab} />
    </Row></Surface>
    <Surface><h2>Identity and evidence</h2><Row>
      <Monogram>VS</Monogram><StatusPill tone="healthy">healthy</StatusPill>
      <StatusPill tone="live">warming</StatusPill><StatusPill tone="danger">failed</StatusPill>
      <StatusPill>stale</StatusPill><StatusPill>uninstalled</StatusPill>
    </Row></Surface>
    <Surface style={{ padding: 0 }}><CollapsibleSection title="Inference" summary="Default" open={open} onOpenChange={setOpen}>
      <Field aria-label="Example setting" defaultValue="Default" /><p>Applies on the next model launch.</p>
    </CollapsibleSection></Surface>
  </Gallery>;
}
const meta = { title: 'Design System/Foundation', component: FoundationGallery, parameters: { layout: 'fullscreen' }, play: async ({ canvasElement }) => {
  const canvas = within(canvasElement);
  await userEvent.hover(await canvas.findByRole('button', { name: 'Download' }));
  await userEvent.click(canvas.getByRole('textbox', { name: 'Search models' }));
  await userEvent.type(canvas.getByRole('textbox', { name: 'Search models' }), 'Example model');
  await expect(canvas.getByRole('textbox', { name: 'Search models' })).toHaveValue('Example model');
  await userEvent.click(canvas.getByRole('button', { name: 'Tailscale' }));
  await expect(canvas.getByRole('button', { name: 'Tailscale' })).toHaveAttribute('aria-pressed', 'true');
  await expect(canvas.getByRole('button', { name: 'Unavailable' })).toBeDisabled();
} } satisfies Meta<typeof FoundationGallery>;
export default meta;
type Story = StoryObj<typeof meta>;
export const Night: Story = { globals: { theme: 'dark' } };
export const NoonRidge: Story = { globals: { theme: 'light' } };
