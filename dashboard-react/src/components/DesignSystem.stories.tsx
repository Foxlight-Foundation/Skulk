import { specimenIndex } from '../../.storybook/specimenIndex';
import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { expect, userEvent, waitFor, within } from 'storybook/test';
import { Gallery, GalleryHeader, GallerySection, Specimen, SpecimenGrid, SpecimenLabel, SpecimenRow } from '../../.storybook/Gallery';
import { RunningInstanceCard } from './cluster/RunningInstanceCard';
import { ToastNotification } from './status/ToastContainer';
import { ClusterNode } from './topology/ClusterNode';
import { PrefillProgressBar } from './display/PrefillProgressBar';
import { HeaderNav } from './layout/HeaderNav';
import { ReadyModelSelect } from './chat/ReadyModelSelect';
import { ChatForm } from './chat/ChatForm';
import { RightDrawer } from './common/RightDrawer';
import { Button } from './common/Button';
import type { NodeInfo } from '../types/topology';

const meta = { title: 'Design System/Component specimens', parameters: { layout: 'fullscreen' } } satisfies Meta;
export default meta;
type Story = StoryObj<typeof meta>;

/** Existing cards with representative observed states and stable fictional identity. */
function CardsGallery() {
  const [dismissed, setDismissed] = useState<string[]>([]);
  const [startedAt] = useState(() => performance.now());
  return <Gallery><GalleryHeader title="Cards and rows">Production instance cards, notifications and progress. Source labels identify what each specimen exercises.</GalleryHeader>
    <GallerySection title="Cards and rows"><SpecimenGrid>
      <Specimen><SpecimenLabel>RunningInstanceCard · ready / loading / failed</SpecimenLabel>{(['ready', 'loading', 'failed'] as const).map((status, index) => <RunningInstanceCard key={status} instanceId={['e8617e40-example', '3f0b91c2-example', 'a71d0e55-example'][index]} modelId={['example/Chat-8B', 'example/Video-Generation', 'example/Reasoning-32B'][index]} sharding="Pipeline" instanceType="MlxRing" engine="mlx" nodeStatuses={[{ name: 'Example workstation', state: status === 'failed' ? 'failed' : status === 'loading' ? 'loading' : 'ready' }]} status={status} onDelete={() => {}} onChat={status === 'ready' ? () => {} : undefined} statusMessage={status === 'loading' ? 'Loading layers 12/40…' : status === 'failed' ? 'Not enough available memory' : undefined} />)}</Specimen>
      <Specimen><SpecimenLabel>ToastNotification · status rail / dismiss / auto-dismiss track</SpecimenLabel>{(['success', 'error', 'warning', 'info'] as const).filter(type => !dismissed.includes(type)).map((type, index) => <ToastNotification key={type} toast={{ id: type, type, message: ['Settings saved — changes apply on the next launch.', 'Download failed: not enough disk space.', 'Pairing code ready.', 'Example workstation joined the cluster.'][index], createdAt: 0, duration: 60000 }} onDismiss={() => setDismissed([...dismissed, type])} />)}<Button onClick={() => setDismissed([])}>Reset notifications</Button></Specimen>
      <Specimen><SpecimenLabel>PrefillProgressBar · amber work in flight</SpecimenLabel><PrefillProgressBar progress={{ processed: 6400, total: 12000, startedAt }} /><SpecimenLabel>Empty progress · no fabricated completion</SpecimenLabel><PrefillProgressBar progress={{ processed: 0, total: 12000, startedAt }} /></Specimen>
    </SpecimenGrid></GallerySection></Gallery>;
}
export const CardsAndRows: Story = { render: () => <CardsGallery /> };

const node: NodeInfo = {
  last_mactop_update: 0,
  friendly_name: 'Example workstation', system_info: { model_id: 'Mac Studio', chip: 'M4 Max', memory: 64 * 1024 ** 3 },
  mactop_info: { memory: { ram_usage: 32 * 1024 ** 3, ram_total: 64 * 1024 ** 3 }, gpu_usage: [0, 0.54], temp: { gpu_temp_avg: 39 }, sys_power: 18 },
};
function TopologyGallery() {
  return <Gallery><GalleryHeader title="Topology">The existing SVG topology preserves its structure, memory fill, compute ring, vendor identity and selection controls.</GalleryHeader><GallerySection title="Topology"><SpecimenGrid>{(['Healthy', 'Selected', 'Syncing'] as const).map(state => <Specimen key={state}><SpecimenLabel>ClusterNode · {state}</SpecimenLabel><svg viewBox="0 0 280 260" width="100%" role="img" aria-label={`${state} example node`}><ClusterNode nodeId="example-workstation" nodeInfo={{ ...node, syncing: state === 'Syncing' }} x={140} y={85} selected={state === 'Selected'} onSelect={() => {}} /></svg></Specimen>)}</SpecimenGrid></GallerySection></Gallery>;
}
export const Topology: Story = { render: () => <TopologyGallery /> };

function ChromeGallery() {
  const [open, setOpen] = useState(false);
  const [width, setWidth] = useState(420);
  const [sent, setSent] = useState('');
  const [model, setModel] = useState('skulk/steward');
  return <Gallery><GalleryHeader title="Chrome and composers">Header navigation, ordinary and Steward composers, and the shared drawer. Actions below only change this story's state.</GalleryHeader>
    <GallerySection title="Header"><HeaderNav instanceCount={3} onOpenSettings={() => setOpen(true)} /></GallerySection>
    <GallerySection title="Chat composers"><SpecimenGrid><Specimen><SpecimenLabel>ChatForm · ordinary · solid starlight send</SpecimenLabel><ChatForm onSend={message => setSent(message)} placeholder="Ask anything…" modelLabel="Example Chat 8B" /></Specimen><Specimen><SpecimenLabel>ChatForm · Steward · solid amber send</SpecimenLabel><ChatForm steward onSend={message => setSent(message)} placeholder="Ask Skulk about the cluster…" modelLabel="Skulk" /></Specimen></SpecimenGrid><p role="status">{sent ? `Local preview: ${sent}` : 'No prompt submitted.'}</p></GallerySection>
    <GallerySection title="Drawer"><SpecimenRow><Button onClick={() => setOpen(true)}>Open settings drawer</Button></SpecimenRow></GallerySection>
    <RightDrawer open={open} onClose={() => setOpen(false)} width={width} minWidth={320} maxWidth={720} onWidthChange={setWidth} title="Settings" ariaLabel="Specimen settings" closeLabel="Close" resizeLabel="Resize"><div style={{ padding: '20px 24px', display: 'grid', gap: 16 }}><ReadyModelSelect value={model} onChange={setModel} fabricEnabled models={[{ modelId: 'example/Chat-8B' }]} /><p>Modal focus, Escape dismissal and keyboard resizing use the same implementation as the dashboard.</p><Button variant="solid" onClick={() => setOpen(false)}>Save changes</Button></div></RightDrawer>
  </Gallery>;
}
export const ChromeAndComposers: Story = { render: () => <ChromeGallery />, play: async ({ canvasElement }) => {
  const canvas = within(canvasElement);
  const open = await canvas.findByRole('button', { name: 'Open settings drawer' });
  await userEvent.click(open);
  await expect(canvas.getByRole('dialog', { name: 'Specimen settings' })).toBeVisible();
  const chooser = canvas.getByRole('button', { name: 'Select chat model' });
  await userEvent.click(chooser);
  const documentCanvas = within(canvasElement.ownerDocument.body);
  await expect(documentCanvas.getByRole('listbox')).toBeVisible();
  await waitFor(() => expect(documentCanvas.getByRole('option', { name: /Skulk/ })).toHaveFocus());
  await userEvent.keyboard('{ArrowDown}');
  await expect(documentCanvas.getByRole('option', { name: /Chat-8B/ })).toHaveFocus();
  await userEvent.keyboard('{Escape}');
  await expect(canvas.getByRole('dialog', { name: 'Specimen settings' })).toBeVisible();
  await expect(chooser).toHaveTextContent('Skulk');
  await userEvent.click(chooser);
  // FloatingFocusManager schedules initial focus after mounting its portal.
  await waitFor(() => expect(documentCanvas.getByRole('option', { name: /Skulk/ })).toHaveFocus());
  await userEvent.keyboard('{ArrowDown}{Enter}');
  await expect(chooser).toHaveTextContent('Chat-8B');
  await expect(documentCanvas.queryByRole('listbox')).not.toBeInTheDocument();
  await userEvent.keyboard('{Escape}');
  await expect(open).toHaveFocus();
} };

/** Every inventoried component remains reachable from one source-labelled index. */
export const ComponentIndex: Story = { render: () => <Gallery>
  <GalleryHeader title="Component inventory">Choose a component to inspect its dedicated states. Connected workflows use isolated fixtures, including unavailable and empty evidence.</GalleryHeader>
  <GallerySection title="Source specimens"><SpecimenGrid>{Array.from(new Set(specimenIndex.map(item => item.source.split('/')[0]))).map(folder => <Specimen key={folder}><SpecimenLabel>{folder}/</SpecimenLabel>{specimenIndex.filter(item => item.source.startsWith(folder + '/')).map(item => <a key={item.source} href={'/?path=/story/' + item.story} target="_top" style={{ overflowWrap: 'anywhere', textDecoration: 'underline', fontSize: 14 }}>{item.source.split('/')[1]}</a>)}</Specimen>)}</SpecimenGrid></GallerySection>
</Gallery> };
