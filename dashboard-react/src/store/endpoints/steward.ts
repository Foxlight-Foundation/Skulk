import { apiSlice } from '../api';

/** Steward availability as reported by `GET /v1/steward`. */
export interface StewardStatus {
  /** Whether intelligent-fabric mode is enabled in Settings. */
  enabled: boolean;
  /** Whether a steward placement currently exists. */
  present: boolean;
  /** Model id of the steward brain when present. */
  steward_model: string | null;
  /** The steward instance id when present. */
  instance_id: string | null;
}

/** One turn of steward conversation history sent to the server. */
export interface StewardChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

/** One tool call the steward made while investigating. */
export interface StewardToolStep {
  tool: string;
  arguments: Record<string, unknown>;
}

/** The steward's reply plus its investigation trace. */
export interface StewardChatResponse {
  reply: string;
  steps: StewardToolStep[];
  steward_model: string;
  instance_id: string;
}

/**
 * Steward endpoints: status polling and the non-streaming chat turn.
 * The chat call can run for minutes on small hardware (each turn is a
 * bounded multi-step investigation), so callers own their loading UX.
 */
export const stewardApi = apiSlice.injectEndpoints({
  endpoints: (build) => ({
    getStewardStatus: build.query<StewardStatus, void>({
      query: () => ({ url: '/v1/steward' }),
      providesTags: ['StewardStatus'],
    }),
    stewardChat: build.mutation<StewardChatResponse, StewardChatMessage[]>({
      query: (messages) => ({
        url: '/v1/steward/chat',
        method: 'POST',
        body: { messages },
      }),
    }),
  }),
});

export const { useGetStewardStatusQuery, useStewardChatMutation } = stewardApi;
