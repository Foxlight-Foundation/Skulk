import type { StewardActionProposal } from '../../store/endpoints/steward';

/**
 * How long a settled proposal stays on screen after it leaves pending. It
 * matches a proposal's own ten-minute lifetime: long enough to read why an
 * approval was refused or a proposal went out of date, short enough that the
 * tray does not become an audit log.
 */
export const RECENT_SETTLED_PROPOSAL_MS = 10 * 60 * 1000;

/**
 * Select the proposals the steward tray shows: every pending proposal, then
 * recently settled ones the operator has not dismissed, newest first.
 *
 * @param proposals Server-owned proposals, newest first.
 * @param nowMs Current time in epoch milliseconds.
 * @param dismissed Proposal ids the operator hid in this view.
 * @returns Pending proposals followed by recently settled ones.
 */
export function visibleStewardProposals(
  proposals: readonly StewardActionProposal[],
  nowMs: number,
  dismissed: ReadonlySet<string>,
): StewardActionProposal[] {
  const pending = proposals.filter((proposal) => proposal.status === 'pending');
  const settled = proposals
    .filter((proposal) => {
      if (proposal.status === 'pending' || dismissed.has(proposal.proposal_id)) return false;
      if (proposal.decided_at === null) return false;
      const settledAt = Date.parse(proposal.decided_at);
      return Number.isFinite(settledAt) && nowMs - settledAt <= RECENT_SETTLED_PROPOSAL_MS;
    })
    .sort((left, right) => Date.parse(right.decided_at ?? '') - Date.parse(left.decided_at ?? ''));
  return [...pending, ...settled];
}

/**
 * Pull the server's refusal text out of a failed decision request, if any.
 *
 * @param error Rejection thrown by the decision mutation.
 * @returns The API's `detail` string, or null when the error carries none.
 */
export function decisionErrorDetail(error: unknown): string | null {
  if (typeof error !== 'object' || error === null || !('data' in error)) return null;
  const data = (error as { data?: unknown }).data;
  if (typeof data !== 'object' || data === null) return null;
  const detail = (data as { detail?: unknown }).detail;
  return typeof detail === 'string' && detail.length > 0 ? detail : null;
}
