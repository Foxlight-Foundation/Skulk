import { apiSlice } from '../api';

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
  }),
});

export const { useGetStewardStatusQuery } = stewardApi;
