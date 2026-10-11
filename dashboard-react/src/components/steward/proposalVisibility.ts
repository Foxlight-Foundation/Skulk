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

/** Browser storage key for the steward proposals this viewer dismissed. */
export const DISMISSED_PROPOSALS_KEY = 'skulk-steward-dismissed-proposals';

/**
 * Read the proposals this viewer dismissed earlier, so a dismissed card stays
 * gone after a reload.
 *
 * Dismissal is a per-viewer convenience, so it lives in this browser only.
 * Storage can be missing or throw (private windows, blocked site data); an
 * empty set is the safe answer, since a card that reappears is only a
 * nuisance.
 *
 * @returns Dismissed proposal ids, or an empty set when none can be read.
 */
export function loadDismissedProposalIds(): Set<string> {
  try {
    const stored = window.localStorage.getItem(DISMISSED_PROPOSALS_KEY);
    if (!stored) return new Set();
    const parsed = JSON.parse(stored) as unknown;
    if (!Array.isArray(parsed)) return new Set();
    return new Set(parsed.filter((value): value is string => typeof value === 'string'));
  } catch {
    return new Set();
  }
}

/**
 * Remember dismissals, keeping only proposals the server still reports.
 *
 * The server retains a bounded proposal audit, so pruning to the ids it still
 * returns keeps the stored list bounded without any separate expiry.
 *
 * @param dismissed Every id the viewer has dismissed, including the new one.
 * @param knownProposalIds Ids in the server's current proposal list.
 * @returns The pruned set that was stored; it is still returned when storage
 *   is unavailable, so the dismissal holds for this page view.
 */
export function saveDismissedProposalIds(
  dismissed: ReadonlySet<string>,
  knownProposalIds: ReadonlySet<string>,
): Set<string> {
  const kept = new Set([...dismissed].filter((id) => knownProposalIds.has(id)));
  try {
    window.localStorage.setItem(DISMISSED_PROPOSALS_KEY, JSON.stringify([...kept]));
  } catch {
    // Storage unavailable: the dismissal lasts for this page view only.
  }
  return kept;
}
