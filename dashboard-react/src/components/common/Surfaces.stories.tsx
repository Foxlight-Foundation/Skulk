import { Select as DesignedSelect } from './Select';
import { expect, userEvent, within } from 'storybook/test';
import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import styled, { useTheme } from 'styled-components';
import { FiCopy, FiSearch, FiSettings, FiSun } from 'react-icons/fi';
import { MdAutoAwesome } from 'react-icons/md';
import { Button } from './Button';
import { Toggle } from './Toggle';
import { Field } from './Field';
import { SegmentedControl } from './SegmentedControl';
import { InfoTooltip } from './InfoTooltip';
import { CollapsibleSection, Monogram, StatusPill, Surface } from './Surfaces';
import { buildTagColors, CapabilityTagBadge } from './capabilityTags';
import { Gallery, GalleryHeader, GallerySection, Specimen, SpecimenGrid, SpecimenLabel, SpecimenRow } from '../../../.storybook/Gallery';

const Swatches = styled.div`display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 150px), 1fr)); gap: 10px;`;
const Swatch = styled.div`
  border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 10px; overflow: hidden;
  background: ${({ theme }) => theme.colors.surface}; min-width: 0;
  > div:first-child { height: 56px; }
  > div:last-child { padding: 8px 10px; }
  strong { font-size: 12.5px; font-weight: 600; color: ${({ theme }) => theme.colors.text}; }
  code { display: block; font: 10.5px ${({ theme }) => theme.fonts.mono}; color: ${({ theme }) => theme.colors.textMuted}; margin-top: 2px; overflow-wrap: anywhere; }
  p { margin-top: 3px; font-size: 11px; line-height: 1.4; color: ${({ theme }) => theme.colors.textSecondary}; }
`;
const TypeRow = styled.div`
  display: flex; align-items: baseline; gap: 16px; border-top: 1px solid ${({ theme }) => theme.colors.borderLight}; padding-top: 10px;
  > code { width: 150px; flex-shrink: 0; font: 10.5px ${({ theme }) => theme.fonts.mono}; color: ${({ theme }) => theme.colors.textMuted}; }
  > span { min-width: 0; overflow-wrap: anywhere; line-height: 1.2; }
  @media(max-width: 600px) { flex-direction: column; gap: 8px; > code { width: auto; } }
`;
const Hint = styled.p`font-size: 12px; line-height: 1.5; color: ${({ theme }) => theme.colors.textMuted};`;
const Fields = styled.div`display: grid; gap: 12px; width: min(100%, 280px);`;
const Select = styled(DesignedSelect)`
  width: 100%; height: 36px; padding: 0 10px; font: 14px ${({ theme }) => theme.fonts.body};
  border: 1px solid ${({ theme }) => theme.colors.borderControl}; border-radius: 8px;
  background: ${({ theme }) => theme.colors.surfaceHover}; color: ${({ theme }) => theme.colors.text};
`;
const tones = ['healthy', 'live', 'danger', 'neutral'] as const;

/** A reference-ordered gallery of real production primitives, with local-only interactions. */
function FoundationGallery() {
  const theme = useTheme();
  const [tab, setTab] = useState('Local network');
  const [format, setFormat] = useState('Shell');
  const [open, setOpen] = useState(true);
  const [loggingOpen, setLoggingOpen] = useState(false);
  const [enabled, setEnabled] = useState(true);
  const [search, setSearch] = useState('');
  const tags = buildTagColors(theme);
  const swatches = [
    ['void', theme.colors.bg, 'page'], ['ridge-1', theme.colors.surface, 'cards, panels'],
    ['ridge-2', theme.colors.surfaceHover, 'inputs, node body'], ['ridge-3', theme.colors.selected, 'selected, monograms'],
    ['elevated', theme.colors.surfaceElevated, 'drawers, modals, tables'], ['starlight', theme.colors.gold, 'accent, active nav, memory'],
    ['foxlight', theme.colors.live, 'live work, steward, warning pip'], ['healthy', theme.colors.healthy, 'node health, ready'],
    ['danger', theme.colors.error, 'failure, delete'], ['vision', theme.colors.tagVision, 'vision capability tag'],
    ['text-1', theme.colors.text, 'headings, values'], ['text-2', theme.colors.body, 'body'],
    ['text-3', theme.colors.textSecondary, 'secondary, labels'], ['text-4', theme.colors.textMuted, 'muted, section labels'],
    ['text-5', theme.colors.idle, 'idle, footnotes'], ['line-hair', theme.colors.borderLight, 'row dividers'],
    ['line-card', theme.colors.border, 'card borders'], ['line-control', theme.colors.borderControl, 'inputs'],
    ['line-strong', theme.colors.borderStrong, 'active, drawer edge'], ['fill-star', theme.colors.goldBg, 'active nav fill'],
  ];
  return <Gallery>
    <GalleryHeader title="Dashboard design system">Every component the dashboard renders today, retokened to the Studio Night system, plus the new pieces from the redesign. Night and Noon Ridge share one token set. Values trace to the production theme, controls, navigation, cards, drawers and topology. The inventory below links each source component to its reviewable states.</GalleryHeader>
    <GallerySection title="Colour"><Swatches>{swatches.map(([name, color, use]) => <Swatch key={name}>
      <div style={{ background: color }} /><div><strong>{name}</strong><code>{color}</code><p>{use}</p></div>
    </Swatch>)}</Swatches><SpecimenGrid>{[
      ['Amber is the living colour.', 'Live work, Steward actions and favourites retain the amber treatments specified by the screens.', theme.colors.live, theme.colors.liveBg, theme.colors.borderLive],
      ['Starlight is the everyday accent.', 'Active nav, focus, primary buttons, memory fill and information. Hairlines are palette-tinted.', theme.colors.gold, theme.colors.goldBg, theme.colors.borderStrong],
      ['Green means healthy.', 'Observed node health, ready instances and verified store entries. Unknown evidence remains explicit.', theme.colors.healthy, theme.colors.accentBg, theme.colors.borderHealthy],
    ].map(([title, copy, color, background, borderColor]) => <Specimen key={title} style={{ background, borderColor, padding: '12px 14px', borderRadius: 10, display: 'block', fontSize: 13, lineHeight: 1.5, color: theme.colors.textSecondary }}><strong style={{ color }}>{title}</strong> {copy}</Specimen>)}</SpecimenGrid></GallerySection>
    <GallerySection title="Type"><SpecimenGrid>
      <Specimen><SpecimenLabel>Instrument Sans · UI</SpecimenLabel>{([
        ['display 34 · 700 · −.025em', 34, 700, 'Dashboard design system'], ['page title 28 · 700', 28, 700, 'Integrations'],
        ['heading 18 · 700', 18, 700, 'Observability'], ['title 16 · 600', 16, 600, 'Active Instances'],
        ['body / table 15 · 400', 15, 400, 'example/Chat-8B'], ['nav / control 14 · 400', 14, 400, 'Model Store · Launch Model'],
        ['caption 12.5 · 400', 12.5, 400, 'Ready to chat! · 20d ago'],
      ] as const).map(([label, size, weight, text]) => <TypeRow key={label}><code>{label}</code><span style={{ fontSize: size, fontWeight: weight, color: theme.colors.text, letterSpacing: size === 34 ? '-.025em' : size === 28 ? '-.02em' : undefined }}>{text}</span></TypeRow>)}</Specimen>
      <Specimen><SpecimenLabel>JetBrains Mono · machine values</SpecimenLabel>{([
        ['section label 10 · 600', 10, 600, 'READY MODELS'], ['meta 11 · 400', 11, 400, 'managed.example · release v1.2.0'],
        ['value 13 · 400', 13, 400, 'api.example.test'], ['instance id 14 · 400', 14, 400, 'E8617E40'], ['stat 22 · 700', 22, 700, '208 GB'],
      ] as const).map(([label, size, weight, text]) => <TypeRow key={label}><code>{label}</code><span style={{ fontFamily: theme.fonts.mono, fontSize: size, fontWeight: weight, letterSpacing: size === 10 ? '.16em' : undefined }}>{text}</span></TypeRow>)}
      <Hint>Mono for machine values and section labels. Sans for prose and controls.</Hint></Specimen>
    </SpecimenGrid></GallerySection>
    <GallerySection title="Buttons, nav, toggles"><SpecimenGrid>
      <Specimen><SpecimenLabel>Button · primary / outline / ghost / danger · sm 30 · md 36 · lg 42</SpecimenLabel>
        <SpecimenRow><Button variant="primary">Download</Button><Button>Outline</Button><Button variant="ghost">Ghost</Button><Button variant="danger">Danger</Button><Button loading>Loading</Button></SpecimenRow>
        <SpecimenRow><Button size="sm">Small</Button><Button size="lg">Large</Button><Button icon aria-label="Copy"><FiCopy /></Button><Button variant="solid">Save changes</Button><Button variant="approve">Approve action</Button></SpecimenRow>
        <SpecimenLabel>Screen-specific commit actions · solid Launch / send</SpecimenLabel>
        <SpecimenRow><Button variant="solid">Launch</Button><Button variant="solid" disabled>Unavailable</Button><Button variant="solid" loading>Launching</Button><Button variant="approve" disabled>Approve unavailable</Button></SpecimenRow>
        <SpecimenLabel>Focus ring</SpecimenLabel><Button style={{ alignSelf: 'flex-start', boxShadow: theme.colors.focusRing }}>Focused</Button>
        <Hint>Tab to inspect focus; hover to inspect borders and fills. Solid starlight commits a change; solid amber approves a fabric proposal.</Hint>
      </Specimen>
      <Specimen><SpecimenLabel>Header icon controls · 42px</SpecimenLabel><SpecimenRow>
        <Button variant="ghost" size="lg" icon aria-label="Theme"><FiSun /></Button>
        <Button variant="ghost" size="lg" icon aria-label="Steward open" aria-pressed style={{ color: theme.colors.live, background: theme.colors.liveBg, borderColor: theme.colors.borderLive }}><MdAutoAwesome /></Button>
        <Button variant="ghost" size="lg" icon aria-label="Settings"><FiSettings /></Button>
      </SpecimenRow><SpecimenLabel>Segmented control · address / format</SpecimenLabel>
      <SpecimenRow><SegmentedControl size="sm" options={['Local network', 'Tailscale']} value={tab} onChange={setTab} /></SpecimenRow>
      <SpecimenRow><SegmentedControl options={['Shell', 'Settings file']} value={format} onChange={setFormat} /></SpecimenRow>
      <SpecimenLabel>Toggle · on / off / disabled</SpecimenLabel><SpecimenRow><Toggle aria-label="Example enabled" $on={enabled} onClick={() => setEnabled(!enabled)} /><Toggle aria-label="Example disabled" $on={!enabled} onClick={() => setEnabled(!enabled)} /><Toggle aria-label="Unavailable toggle" $on={false} disabled /></SpecimenRow>
      <SpecimenLabel>Disabled choices</SpecimenLabel><SegmentedControl<string> options={[{ value: 'Available', label: 'Available' }, { value: 'Unavailable', label: 'Unavailable', disabled: true }]} value="Available" onChange={() => {}} />
      </Specimen>
    </SpecimenGrid></GallerySection>
    <GallerySection title="Fields, selects, tags, pills"><SpecimenGrid>
      <Specimen><SpecimenLabel>Field · sm 30 / md 36 / lg 42 · icon / search / select</SpecimenLabel><Fields>
        <Field size="sm" aria-label="Example node name" defaultValue="Example workstation" />
        <Field aria-label="Search models" placeholder="Search supported models…" icon={<FiSearch size={14} />} value={search} onChange={e => setSearch(e.target.value)} />
        <Field size="lg" aria-label="Example token" type="password" defaultValue="fictional-token" />
        <Select aria-label="Example cache format"><option>TurboQuant Adaptive</option><option>Default</option></Select>
        <Field aria-label="Unavailable field" placeholder="Unavailable" disabled />
      </Fields><SpecimenRow>Enabled <InfoTooltip size={14} filled content="Changes apply on the next model launch." triggerLabel="About enabled" /></SpecimenRow>
      <Hint><i>Hint text · italic 12px text-4. Used under settings rows.</i></Hint></Specimen>
      <Specimen><SpecimenLabel>Capability tags · 10px 500 · radius 4 · 0×5</SpecimenLabel><SpecimenRow>{['thinking', 'vision', 'tensor', ...Object.keys(theme.capabilityTints)].map(name => <CapabilityTagBadge key={name} $color={tags[name].color} $bg={tags[name].bg} $border={tags[name].border}>{name}</CapabilityTagBadge>)}</SpecimenRow>
      <SpecimenLabel>Health pills · lowercase mono · observed evidence</SpecimenLabel><SpecimenRow>{tones.map((tone, i) => <StatusPill key={tone} tone={tone}>{['healthy', 'needs attention', 'failed', 'stale'][i]}</StatusPill>)}</SpecimenRow>
      <SpecimenLabel>Identity · ridge-3 monograms</SpecimenLabel><SpecimenRow><Monogram $size={36}>CC</Monogram><Monogram>VS</Monogram></SpecimenRow>
      <Hint>Verification, compatibility and availability are separate facts. Unknown evidence remains explicit.</Hint></Specimen>
    </SpecimenGrid></GallerySection>
    <GallerySection title="Collapsible settings rows"><Surface style={{ padding: 0 }}><CollapsibleSection title="Inference" summary="Default" open={open} onOpenChange={setOpen}>
      <Field aria-label="Example setting" defaultValue="Default" /><Hint>Applies on the next model launch.</Hint>
    </CollapsibleSection><CollapsibleSection title="Logging" summary="off" open={loggingOpen} onOpenChange={setLoggingOpen}><span>Logging disabled</span></CollapsibleSection></Surface></GallerySection>
    <GallerySection title="Component inventory"><SpecimenGrid>{[
      ['Cards and rows', 'design-system-component-specimens--cards-and-rows'],
      ['Topology', 'design-system-component-specimens--topology'],
      ['Chrome and composers', 'design-system-component-specimens--chrome-and-composers'],
      ['Domain components', 'design-system-domain-surfaces--night'],
      ['All source specimens', 'design-system-component-specimens--component-index'],
    ].map(([label, id]) => <Specimen key={id}><a href={'/?path=/story/' + id} target="_top">{label} →</a></Specimen>)}</SpecimenGrid></GallerySection>
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
  for (const button of canvas.getAllByRole('button', { name: /^Unavailable$/ })) await expect(button).toBeDisabled();
} } satisfies Meta<typeof FoundationGallery>;
export default meta;
type Story = StoryObj<typeof meta>;
export const Night: Story = { globals: { theme: 'dark' } };
export const NoonRidge: Story = { globals: { theme: 'light' } };
