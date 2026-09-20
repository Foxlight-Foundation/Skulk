import { IntegrationSetupStep } from '../integrations/IntegrationSetupStep';
import { PairingInvitationRow } from '../layout/PairingSettings';
import { ReadyModelSelect } from '../chat/ReadyModelSelect';
import type { Meta, StoryObj } from '@storybook/react-vite';
import styled from 'styled-components';
import { useState } from 'react';
import { Gallery, GalleryHeader, GallerySection, Specimen, SpecimenGrid, SpecimenLabel } from '../../../.storybook/Gallery';
import { StewardPrompt } from '../steward/StewardPrompt';
import { IntegrationToolCard } from '../integrations/IntegrationToolCard';
import { StewardProposalCard } from '../steward/StewardProposalCard';
import { PluginSummaryCard } from './PluginSummaryCard';
import { DeviceRow } from '../layout/DevicesPanel';

const Grid = styled.div`display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 160px), 1fr)); gap: 12px;`;
const Narrow = styled.div`max-width: 412px; width: 100%;`;

/** Domain specimens use inert callbacks, never cluster operations. */
function DomainGallery() {
  const [model, setModel] = useState('skulk/steward');
  return <Gallery>
    <GalleryHeader title="Domain components">Steward, integrations, plugins and devices. These are the production components with fictional evidence; requests cannot reach a real cluster.</GalleryHeader>
    <GallerySection title="Steward"><SpecimenGrid><Specimen style={{ gridColumn: '1 / -1' }}><SpecimenLabel>Cluster entry · 640×48 · opens without submitting</SpecimenLabel>

    <StewardPrompt onOpen={() => {}} />
    <SpecimenLabel>Chat · fabric / ready models</SpecimenLabel><ReadyModelSelect value={model} onChange={setModel} fabricEnabled models={[{ modelId: 'example/Local-chat-model' }]} /></Specimen>
    <Specimen><SpecimenLabel>Proposal · pending approval · drawer width</SpecimenLabel><Narrow>
    <StewardProposalCard busy={false} onDecision={() => {}} proposal={{ proposal_id: 'fixture-proposal', action: 'restart_model', target: 'example/model', rationale: 'The model stopped responding to health probes.', evidence: ['Three unsuccessful health probes were observed.'], expected_effect: 'Restart the selected instance.', created_at: '2026-01-01T12:00:00Z', expires_at: '2026-01-01T12:05:00Z', status: 'pending', decided_at: null, decided_by: null, outcome: null }} />
    </Narrow></Specimen></SpecimenGrid></GallerySection>
    <GallerySection title="Integrations"><SpecimenLabel>Tool cards · 36px monogram · set up does not imply connected</SpecimenLabel><Grid>
      <IntegrationToolCard name="Claude Code" monogram="CC" description="Anthropic-compatible coding agent in your terminal." method="anthropic · shell or settings.json" onOpen={() => {}} />
      <IntegrationToolCard name="OpenCode" monogram="OC" description="Open-source terminal agent with provider config." method="openai · config" onOpen={() => {}} />
      <IntegrationToolCard name="Codex" monogram="CX" description="Point your coding agent at a local model." method="openai · config.toml" onOpen={() => {}} />
    </Grid>
    <IntegrationSetupStep number={3} title="Check it reached the cluster"><p>Connection evidence unavailable.</p></IntegrationSetupStep>
    </GallerySection><GallerySection title="Plugins"><SpecimenLabel>Summary cards · explicit unknown / stale evidence · detail actions preserved</SpecimenLabel>
    <PluginSummaryCard name="Video Studio" description="Generative video on your fabric." tone="neutral" health="Status unavailable" release="1.2.0" nodes={['Video capability']} onOpen={() => {}} />
    <PluginSummaryCard name="A plugin with a longer translated display name" description="Health cannot be confirmed from the current observation." tone="neutral" health="Stale" release="1.2.0" nodes={[]} onOpen={() => {}} />
    </GallerySection><GallerySection title="Devices and invitations"><SpecimenGrid><Specimen><SpecimenLabel>Invitations · active / expired · metadata only</SpecimenLabel>
    <PairingInvitationRow busy={false} onRevoke={() => {}} invitation={{ invitationId: 'fictional-invitation', createdAt: '2026-01-01T12:00:00Z', expiresAt: '2026-01-01T12:05:00Z', successfulPairings: 1, maxPairings: 3, activeAttempts: 0, totalAttempts: 1, state: 'active' }} />
    <PairingInvitationRow busy={false} onRevoke={() => {}} invitation={{ invitationId: 'fictional-expired', createdAt: '2026-01-01T12:00:00Z', expiresAt: '2026-01-01T12:05:00Z', successfulPairings: 1, maxPairings: 1, activeAttempts: 0, totalAttempts: 1, state: 'expired' }} />
    </Specimen><Specimen><SpecimenLabel>Devices · paired / revoked · unknown presence</SpecimenLabel>
    <DeviceRow device={{ deviceId: 'fictional-device', name: 'Example tablet', pairedAt: '2026-01-01T12:00:00Z', refreshExpiresAt: null, state: 'active', current: false }} busy={false} onRevoke={() => {}} />
    <DeviceRow device={{ deviceId: 'fictional-revoked', name: 'Revoked device with a long translated name', pairedAt: '2026-01-01T12:00:00Z', refreshExpiresAt: null, state: 'revoked', current: false }} busy={false} onRevoke={() => {}} />
  </Specimen></SpecimenGrid></GallerySection></Gallery>;
}
const meta = { title: 'Design System/Domain surfaces', component: DomainGallery, parameters: { layout: 'fullscreen' } } satisfies Meta<typeof DomainGallery>;
export default meta;
type Story = StoryObj<typeof meta>;
export const Night: Story = { globals: { theme: 'dark' } };
export const NoonRidge: Story = { globals: { theme: 'light' } };
