import styled from 'styled-components';
import { Button } from '../common/Button';
import type { StewardActionProposal } from '../../store/endpoints/steward';
import { useSkulkTranslation } from '../../i18n/tolgee';

const ProposalCard = styled.article`
  display: grid;
  gap: 8px;
  padding: 12px 14px;
  border: 1px solid ${({ theme }) => theme.colors.borderLive};
  border-radius: 12px;
  background: ${({ theme }) => theme.colors.liveBg};
`;

const ProposalTitle = styled.div`
  font-family: ${({ theme }) => theme.fonts.body};
  font-weight: 600;
  overflow-wrap: anywhere;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.text};
`;

const ProposalCopy = styled.div`
  font-size: 13px;
  line-height: 1.45;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const ProposalEvidence = styled.ul`
  margin: 0;
  padding-left: 18px;
  font-size: 13px;
  line-height: 1.45;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const ProposalActions = styled.div`
  display: flex;
  gap: 8px;
`;

const ProposalHeading = styled.div`
  display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px;
  font: 11px ${({ theme }) => theme.fonts.mono}; color: ${({ theme }) => theme.colors.textMuted};
  strong { font-size: 10px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: ${({ theme }) => theme.colors.live}; }
`;
const ProposalButton = styled(Button)`
  height: 32px; font-size: 13px; padding: 0 14px;
`;

/** Evidence and explicit decision controls for a server-owned pending proposal. */
export function StewardProposalCard({ proposal, busy, onDecision }: { proposal: StewardActionProposal; busy: boolean; onDecision: (approved: boolean) => void }) {
  const { t } = useSkulkTranslation();
  return (            <ProposalCard>
              <ProposalHeading><strong>{t('stewardChat.proposals.proposedAction', 'Proposed action')}</strong><span>{t('stewardChat.proposals.approvalRequired', 'needs your approval')}</span></ProposalHeading>
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
                  variant="approve"
                  disabled={busy}
                  onClick={() => onDecision(true)}
                >
                  {t('stewardChat.proposals.approve', 'Approve')}
                </ProposalButton>
                <ProposalButton
                  variant="outline"
                  disabled={busy}
                  onClick={() => onDecision(false)}
                >
                  {t('stewardChat.proposals.notNow', 'Not now')}
                </ProposalButton>
              </ProposalActions>
            </ProposalCard>
  );
}
