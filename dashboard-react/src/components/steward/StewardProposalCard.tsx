import styled from 'styled-components';
import type { StewardActionProposal } from '../../store/endpoints/steward';
import { useSkulkTranslation } from '../../i18n/tolgee';

const ProposalCard = styled.article`
  display: grid;
  gap: 8px;
  padding: 12px;
  border: 1px solid ${({ theme }) => theme.colors.borderLive};
  border-radius: 12px;
  background: ${({ theme }) => theme.colors.liveBg};
`;

const ProposalTitle = styled.div`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.text};
`;

const ProposalCopy = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  line-height: 1.45;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const ProposalEvidence = styled.ul`
  margin: 0;
  padding-left: 18px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  line-height: 1.45;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const ProposalActions = styled.div`
  display: flex;
  gap: 8px;
`;

const ProposalButton = styled.button<{ $reject?: boolean }>`
  border: 1px solid ${({ $reject, theme }) => $reject ? theme.colors.border : theme.colors.approvalFill};
  border-radius: ${({ theme }) => theme.radii.sm};
  background: ${({ $reject, theme }) => $reject ? theme.colors.surface : theme.colors.approvalFill};
  color: ${({ $reject, theme }) => $reject ? theme.colors.textSecondary : theme.colors.onLive};
  padding: 6px 10px;
  font: inherit;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  cursor: pointer;

  &:disabled {
    cursor: wait;
    opacity: 0.55;
  }
`;

/** Evidence and explicit decision controls for a server-owned pending proposal. */
export function StewardProposalCard({ proposal, busy, onDecision }: { proposal: StewardActionProposal; busy: boolean; onDecision: (approved: boolean) => void }) {
  const { t } = useSkulkTranslation();
  return (            <ProposalCard>
              <ProposalCopy>{t('stewardChat.proposals.pending', 'PROPOSED ACTION · needs your approval')}</ProposalCopy>
              <ProposalTitle>
                {t('stewardChat.proposals.title', '{action}: {target}', {
                  action: proposal.action.replaceAll('_', ' '),
                  target: proposal.target,
                })}
              </ProposalTitle>
              <ProposalCopy>{proposal.rationale}</ProposalCopy>
              <ProposalCopy>
                {t('stewardChat.proposals.evidence', 'Evidence')}
              </ProposalCopy>
              <ProposalEvidence>
                {proposal.evidence.map((item, index) => (
                  <li key={`${proposal.proposal_id}-${index}`}>{item}</li>
                ))}
              </ProposalEvidence>
              <ProposalCopy>
                {t('stewardChat.proposals.effect', 'Expected effect: {effect}', {
                  effect: proposal.expected_effect,
                })}
              </ProposalCopy>
              <ProposalCopy>
                {t('stewardChat.proposals.expiry', 'Expires: {time}', {
                  time: new Date(proposal.expires_at).toLocaleTimeString(),
                })}
              </ProposalCopy>
              <ProposalActions>
                <ProposalButton
                  disabled={busy}
                  onClick={() => onDecision(true)}
                >
                  {t('stewardChat.proposals.approve', 'Approve')}
                </ProposalButton>
                <ProposalButton
                  $reject
                  disabled={busy}
                  onClick={() => onDecision(false)}
                >
                  {t('stewardChat.proposals.notNow', 'Not now')}
                </ProposalButton>
              </ProposalActions>
            </ProposalCard>
  );
}
