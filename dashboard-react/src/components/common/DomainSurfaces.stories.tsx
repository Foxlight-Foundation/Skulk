import { IntegrationSetupStep } from '../integrations/IntegrationSetupStep';
import { PairingInvitationRow } from '../layout/PairingSettings';
import { ReadyModelSelect } from '../chat/ReadyModelSelect';
import type { Meta, StoryObj } from '@storybook/react-vite';
import styled from 'styled-components';
import { StewardPrompt } from '../steward/StewardPrompt';
import { IntegrationToolCard } from '../integrations/IntegrationToolCard';
import { StewardProposalCard } from '../steward/StewardProposalCard';
import { PluginSummaryCard } from './PluginSummaryCard';
import { DeviceRow } from '../layout/DevicesPanel';

const Stack = styled.div`display: grid; gap: 24px; padding: 24px; max-width: 1000px;`;
const Grid = styled.div`display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px;`;


/** Domain specimens use inert callbacks, never cluster operations. */
function DomainGallery() {
  return <Stack>
    <ReadyModelSelect value="skulk/steward" onChange={() => {}} fabricEnabled models={[{ modelId: 'example/Local-chat-model' }]} />
    <StewardPrompt onOpen={() => {}} />
    <StewardProposalCard busy={false} onDecision={() => {}} proposal={{ proposal_id: 'fixture-proposal', action: 'restart_model', target: 'example/model', rationale: 'The model stopped responding to health probes.', evidence: ['Three unsuccessful health probes were observed.'], expected_effect: 'Restart the selected instance.', created_at: '2026-01-01T12:00:00Z', expires_at: '2026-01-01T12:05:00Z', status: 'pending', decided_at: null, decided_by: null, outcome: null }} />
    <Grid>
      <IntegrationToolCard name="Claude Code" monogram="CC" description="Anthropic-compatible coding agent in your terminal." method="anthropic · shell or settings.json" onOpen={() => {}} />
      <IntegrationToolCard name="OpenCode" monogram="OC" description="Open-source terminal agent with provider config." method="openai · config" onOpen={() => {}} />
      <IntegrationToolCard name="Codex" monogram="CX" description="Point your coding agent at a local model." method="openai · config.toml" onOpen={() => {}} />
    </Grid>
    <IntegrationSetupStep number={3} title="Check it reached the cluster"><p>Connection evidence unavailable.</p></IntegrationSetupStep>
    <PluginSummaryCard name="Video Studio" description="Generative video on your fabric." tone="neutral" health="Status unavailable" release="1.2.0" nodes={['Video capability']} onOpen={() => {}} />
    <PluginSummaryCard name="A plugin with a longer translated display name" description="Health cannot be confirmed from the current observation." tone="neutral" health="Stale" release="1.2.0" nodes={[]} onOpen={() => {}} />
    <PairingInvitationRow busy={false} onRevoke={() => {}} invitation={{ invitationId: 'fictional-invitation', createdAt: '2026-01-01T12:00:00Z', expiresAt: '2026-01-01T12:05:00Z', successfulPairings: 1, maxPairings: 3, activeAttempts: 0, totalAttempts: 1, state: 'active' }} />
    <PairingInvitationRow busy={false} onRevoke={() => {}} invitation={{ invitationId: 'fictional-expired', createdAt: '2026-01-01T12:00:00Z', expiresAt: '2026-01-01T12:05:00Z', successfulPairings: 1, maxPairings: 1, activeAttempts: 0, totalAttempts: 1, state: 'expired' }} />
    <DeviceRow device={{ deviceId: 'fictional-device', name: 'Example tablet', pairedAt: '2026-01-01T12:00:00Z', refreshExpiresAt: null, state: 'active', current: false }} busy={false} onRevoke={() => {}} />
    <DeviceRow device={{ deviceId: 'fictional-revoked', name: 'Revoked device with a long translated name', pairedAt: '2026-01-01T12:00:00Z', refreshExpiresAt: null, state: 'revoked', current: false }} busy={false} onRevoke={() => {}} />
  </Stack>;
}
const meta = { title: 'Design System/Domain surfaces', component: DomainGallery, parameters: { layout: 'fullscreen' } } satisfies Meta<typeof DomainGallery>;
export default meta;
type Story = StoryObj<typeof meta>;
export const Night: Story = { globals: { theme: 'dark' } };
export const NoonRidge: Story = { globals: { theme: 'light' } };
