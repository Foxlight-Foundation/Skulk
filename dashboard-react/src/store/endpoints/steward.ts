import { apiSlice } from '../api';

/**
 * One-word lifecycle summary of the steward, derived server-side from the
 * boolean fields plus liveness-canary history:
 *
 * - `disabled`: intelligent-fabric mode is off.
 * - `downloading`: placed, weights still staging (the long first wait).
 * - `starting`: being placed, or placed and loading.
 * - `ready`: serving, with a clean canary history.
 * - `degraded`: serving, but a liveness probe has failed and the fabric may
 *   repair the placement.
 */
export type StewardState = 'disabled' | 'downloading' | 'starting' | 'ready' | 'degraded';
/** Controlled best-brain convergence activity. */
export type StewardTransition = 'idle' | 'prestaging' | 'replacing' | 'repairing';

/** Steward availability as reported by `GET /v1/steward`. */
export interface StewardStatus {
  /** Whether intelligent-fabric mode is enabled in Settings. */
  enabled: boolean;
  /** Whether a steward placement currently exists. */
  present: boolean;
  /** Whether every steward runner is Ready/Running (model loaded and serving). */
  ready: boolean;
  /** Model id of the steward brain when present. */
  steward_model: string | null;
  /** The steward instance id when present. */
  instance_id: string | null;
  /** Preferred brain being prepared, or the currently serving brain. */
  desired_model: string | null;
  /** Best-brain lifecycle activity. */
  transition: StewardTransition;
  /** Aggregate prestaging ratio from 0 to 1, when measurable. */
  progress: number | null;
  /**
   * Renderable lifecycle word. The booleans above stay authoritative; this
   * saves every client from re-deriving the same precedence rules.
   */
  state: StewardState;
}

/** Safe action summary returned by the steward proposal API. */
export interface StewardActionProposal {
  proposal_id: string;
  action: 'place_model' | 'stop_model' | 'restart_model' | 'cancel_download';
  target: string;
  rationale: string;
  evidence: string[];
  expected_effect: string;
  created_at: string;
  expires_at: string;
  status: 'pending' | 'approved' | 'dispatched' | 'rejected' | 'expired' | 'failed';
  decided_at: string | null;
  decided_by: string | null;
  outcome: string | null;
}

/** Explicit operator decision submitted for one pending proposal. */
export interface StewardActionDecision {
  proposalId: string;
  approved: boolean;
}

/**
 * Steward status endpoint. Conversation rides the standard streaming
 * chat-completions surface with the reserved virtual model id (see
 * StewardChatView); only presence/readiness needs a dedicated query.
 */
export const stewardApi = apiSlice.injectEndpoints({
  endpoints: (build) => ({
    getStewardStatus: build.query<StewardStatus, void>({
      query: () => ({ url: '/v1/steward' }),
      providesTags: ['StewardStatus'],
    }),
    getStewardProposals: build.query<StewardActionProposal[], void>({
      query: () => ({ url: '/v1/steward/proposals' }),
      providesTags: ['StewardProposals'],
    }),
    decideStewardProposal: build.mutation<void, StewardActionDecision>({
      query: ({ proposalId, approved }) => ({
        url: `/v1/steward/proposals/${encodeURIComponent(proposalId)}/decision`,
        method: 'POST',
        body: { approved },
      }),
      invalidatesTags: ['StewardProposals'],
    }),
  }),
});

export const {
  useDecideStewardProposalMutation,
  useGetStewardProposalsQuery,
  useGetStewardStatusQuery,
} = stewardApi;
