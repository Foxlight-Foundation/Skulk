import { afterEach, describe, expect, it, vi } from 'vitest';
import type { StewardActionProposal } from '../../store/endpoints/steward';
import {
  DISMISSED_PROPOSALS_KEY,
  RECENT_SETTLED_PROPOSAL_MS,
  decisionErrorDetail,
  loadDismissedProposalIds,
  saveDismissedProposalIds,
  visibleStewardProposals,
} from './proposalVisibility';

const NOW = Date.parse('2026-10-11T12:00:00Z');

function proposal(
  id: string,
  status: StewardActionProposal['status'],
  decidedMsAgo: number | null = null,
): StewardActionProposal {
  return {
    proposal_id: id,
    action: 'restart_model',
    target: 'org/model',
    rationale: 'The runner is degraded.',
    evidence: ['Three failed probes.'],
    expected_effect: 'Replace the ordinary model instance.',
    created_at: new Date(NOW - 120_000).toISOString(),
    expires_at: new Date(NOW + 480_000).toISOString(),
    status,
    decided_at: decidedMsAgo === null ? null : new Date(NOW - decidedMsAgo).toISOString(),
    decided_by: decidedMsAgo === null ? null : 'fabric_staleness',
    outcome: status === 'stale' ? 'The instance changed after review: its runners were replaced.' : null,
  };
}

describe('visibleStewardProposals', () => {
  it('shows pending first, then recently settled newest first', () => {
    const visible = visibleStewardProposals(
      [
        proposal('older-stale', 'stale', 60_000),
        proposal('pending', 'pending'),
        proposal('newer-failed', 'failed', 5_000),
      ],
      NOW,
      new Set(),
    );
    expect(visible.map((item) => item.proposal_id)).toEqual(['pending', 'newer-failed', 'older-stale']);
  });

  it('drops settled proposals past the window and those the operator dismissed', () => {
    const visible = visibleStewardProposals(
      [
        proposal('aged-out', 'stale', RECENT_SETTLED_PROPOSAL_MS + 1),
        proposal('dismissed', 'expired', 1_000),
        proposal('kept', 'dispatched', 1_000),
      ],
      NOW,
      new Set(['dismissed']),
    );
    expect(visible.map((item) => item.proposal_id)).toEqual(['kept']);
  });
});

describe('decisionErrorDetail', () => {
  it('returns the API refusal text when present', () => {
    expect(decisionErrorDetail({ status: 409, data: { detail: 'Steward proposal is already stale: moved.' } }))
      .toBe('Steward proposal is already stale: moved.');
  });

  it('returns null for errors without a detail string', () => {
    expect(decisionErrorDetail(new Error('network'))).toBeNull();
    expect(decisionErrorDetail({ status: 500, data: 'oops' })).toBeNull();
    expect(decisionErrorDetail({ status: 409, data: { detail: '' } })).toBeNull();
  });
});

describe('dismissed proposal storage', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.removeItem(DISMISSED_PROPOSALS_KEY);
  });

  it('round-trips dismissals and prunes ids the server no longer reports', () => {
    const kept = saveDismissedProposalIds(new Set(['current', 'pruned']), new Set(['current', 'other']));
    expect([...kept]).toEqual(['current']);
    expect([...loadDismissedProposalIds()]).toEqual(['current']);
  });

  it('reads missing, corrupt, or foreign values as nothing dismissed', () => {
    expect(loadDismissedProposalIds().size).toBe(0);
    window.localStorage.setItem(DISMISSED_PROPOSALS_KEY, '{not json');
    expect(loadDismissedProposalIds().size).toBe(0);
    window.localStorage.setItem(DISMISSED_PROPOSALS_KEY, '{"a": 1}');
    expect(loadDismissedProposalIds().size).toBe(0);
    window.localStorage.setItem(DISMISSED_PROPOSALS_KEY, '["kept", 7]');
    expect([...loadDismissedProposalIds()]).toEqual(['kept']);
  });

  it('keeps the dismissal for the page view when storage refuses writes', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    expect([...saveDismissedProposalIds(new Set(['current']), new Set(['current']))]).toEqual(['current']);
  });
});
