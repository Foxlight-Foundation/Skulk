import styled, { css } from 'styled-components';
import { Button } from '../common/Button';
import type { StewardActionProposal } from '../../store/endpoints/steward';
import { useSkulkTranslation } from '../../i18n/tolgee';

/** Visual weight of a card: live while it waits, quieter once settled. */
type ProposalTone = 'live' | 'settled' | 'warning' | 'error';

const ProposalCard = styled.article<{ $tone: ProposalTone }>`
  display: grid;
  gap: 8px;
  padding: 12px 14px;
  border: 1px solid ${({ theme }) => theme.colors.borderLive};
  border-radius: 12px;
  background: ${({ theme }) => theme.colors.liveBg};
  ${({ $tone, theme }) => $tone === 'settled' && css`
    border-color: ${theme.colors.borderLight};
    background: transparent;
  `}
  ${({ $tone, theme }) => $tone === 'warning' && css`
    border-color: ${theme.colors.warning};
    background: ${theme.colors.warningBg};
  `}
  ${({ $tone, theme }) => $tone === 'error' && css`
    border-color: ${theme.colors.error};
    background: ${theme.colors.errorBg};
  `}
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

const ProposalReason = styled.div<{ $tone: ProposalTone }>`
  font-size: 13px;
  line-height: 1.45;
  overflow-wrap: anywhere;
  color: ${({ $tone, theme }) =>
    $tone === 'warning' ? theme.colors.warningText
      : $tone === 'error' ? theme.colors.errorText
        : theme.colors.text};
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

const ProposalHeading = styled.div<{ $tone: ProposalTone }>`
  display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px;
  font: 11px ${({ theme }) => theme.fonts.mono}; color: ${({ theme }) => theme.colors.textMuted};
  strong {
    font-size: 10px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase;
    color: ${({ $tone, theme }) =>
      $tone === 'warning' ? theme.colors.warningText
        : $tone === 'error' ? theme.colors.errorText
          : $tone === 'settled' ? theme.colors.textMuted
            : theme.colors.live};
  }
`;

const ProposalButton = styled(Button)`
  height: 32px; font-size: 13px; padding: 0 14px;
`;

/** Props for one steward proposal, pending or recently settled. */
export interface StewardProposalCardProps {
  /** Server-owned proposal; its status decides what the card offers. */
  proposal: StewardActionProposal;
  /** True while a decision request is in flight, to prevent double clicks. */
  busy: boolean;
  /** Approve (true) or decline (false) a pending proposal. */
  onDecision: (approved: boolean) => void;
  /** Hide a settled card from this view; omitted for pending proposals. */
  onDismiss?: () => void;
}

function toneFor(status: StewardActionProposal['status']): ProposalTone {
  if (status === 'pending') return 'live';
  if (status === 'stale') return 'warning';
  if (status === 'failed') return 'error';
  return 'settled';
}

/**
 * One steward proposal. A pending proposal shows its evidence and the
 * Approve / Not now controls. Once it settles, the card stays briefly and
 * says what happened: carried out, refused with the server's reason, out of
 * date with what changed, expired, or declined. An operator should never
 * approve something and see it silently vanish.
 */
export function StewardProposalCard({ proposal, busy, onDecision, onDismiss }: StewardProposalCardProps) {
  const { t } = useSkulkTranslation();
  const tone = toneFor(proposal.status);
  const title = t('stewardChat.proposals.title', '{action}: {target}', {
    action: proposal.action.replaceAll('_', ' '),
    target: proposal.target,
  });

  if (proposal.status !== 'pending') {
    const label = {
      approved: t('stewardChat.proposals.status.approved', 'Approved, in progress'),
      dispatched: t('stewardChat.proposals.status.dispatched', 'Approved and sent to the cluster'),
      rejected: t('stewardChat.proposals.status.rejected', 'Not approved'),
      expired: t('stewardChat.proposals.status.expired', 'Expired'),
      failed: t('stewardChat.proposals.status.failed', 'Not carried out'),
      stale: t('stewardChat.proposals.status.stale', 'Out of date'),
    }[proposal.status];
    // Only refusals carry a reason worth reading; the other outcomes are
    // internal bookkeeping already summarized by the label.
    const reason = proposal.status === 'stale' || proposal.status === 'failed'
      ? proposal.outcome
      : proposal.status === 'expired'
        ? t('stewardChat.proposals.expiredReason', 'No decision was made before it expired.')
        : null;
    return (
      <ProposalCard $tone={tone} data-testid="steward-proposal-settled" data-status={proposal.status}>
        <ProposalHeading $tone={tone}><strong>{label}</strong></ProposalHeading>
        <ProposalTitle>{title}</ProposalTitle>
        {reason ? <ProposalReason $tone={tone}>{reason}</ProposalReason> : null}
        {proposal.status === 'stale' ? (
          <ProposalCopy>
            {t('stewardChat.proposals.staleHint', 'Ask the steward again if this is still needed; it will look at the cluster as it is now.')}
          </ProposalCopy>
        ) : null}
        {onDismiss ? (
          <ProposalActions>
            <ProposalButton variant="ghost" onClick={onDismiss}>
              {t('stewardChat.proposals.dismiss', 'Dismiss')}
            </ProposalButton>
          </ProposalActions>
        ) : null}
      </ProposalCard>
    );
  }

  return (
    <ProposalCard $tone={tone} data-testid="steward-proposal-pending">
      <ProposalHeading $tone={tone}><strong>{t('stewardChat.proposals.proposedAction', 'Proposed action')}</strong><span>{t('stewardChat.proposals.approvalRequired', 'needs your approval')}</span></ProposalHeading>
      <ProposalTitle>{title}</ProposalTitle>
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
