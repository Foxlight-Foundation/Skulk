import { apiSlice } from '../api';
import type { ProposalOperation, ProposalReference } from './plugins';

/** Readiness of one installed burst provider on this host (`GET /v1/burst/providers`). */
export interface BurstProvider {
  pluginId: string;
  nodeId: string;
  bundleId: string;
  version: string;
  title: string | null;
  memoryPools: ('vram' | 'ram')[];
  /** available, configuration_required, unavailable, or disabled (installation turned off). */
  state: 'available' | 'configuration_required' | 'unavailable' | 'disabled';
  /** Plain text naming the next step. */
  message: string;
  /** Owner ceiling on one rental's total cost, as an exact dollar string. */
  maximumTotalUsd: string | null;
  maximumLifetimeSeconds: number | null;
}

/** Installed burst providers on this API host. */
export interface BurstProviders {
  scope: 'local_api_host';
  observedAt: number;
  providers: BurstProvider[];
}

/** Offer terms; money as exact US dollar strings. */
export interface BurstOffer {
  gpu: string;
  region: string;
  /** Quoted hourly price; advisory, never the spending limit. */
  hourlyPriceUsd: string;
  maximumHourlyPriceUsd: string;
  /** Worst case the approval permits. */
  maximumTotalUsd: string;
  /** UTC Unix seconds after which the offer can no longer be approved. */
  planExpiresAt: number;
  /** UTC Unix seconds when the rental ends at the latest. */
  resourceExpiresAt: number;
  modelId: string;
  contextTokens: number;
  backend: string;
  skulkBuild: string;
}

/** Outcome of `POST /v1/burst/recommendations`. */
export interface BurstRecommendation {
  state:
    | 'offer'
    | 'not_required'
    | 'blocked'
    | 'no_provider'
    | 'provider_required'
    | 'provider_unavailable'
    | 'no_eligible_capacity'
    | 'skulk_build_unavailable'
    | 'refused';
  message: string;
  modelId: string;
  contextTokens: number;
  assessmentStatus: 'local' | 'external_required' | 'blocked' | 'insufficient_evidence';
  assessmentReasons: string[];
  provider: BurstProvider | null;
  reference: ProposalReference | null;
  offer: BurstOffer | null;
  approvalRevision: string | null;
  review: { label: string; value: string }[];
}

/** One rental's summary phase, absence first. */
export type BurstPhase =
  | 'approving'
  | 'refused'
  | 'attention'
  | 'uncertain'
  | 'starting'
  | 'installing'
  | 'joining'
  | 'downloading'
  | 'loading'
  | 'ready'
  | 'releasing'
  | 'released';

/** One approved rental (`/v1/burst/operations`). */
export interface BurstOperation {
  pluginId: string;
  nodeId: string;
  operationId: string;
  phase: BurstPhase;
  message: string;
  /** Skulk instance serving the model, once observed. */
  instanceId: string | null;
  modelId: string | null;
  resourceExpiresAt: number | null;
  acquisition: ProposalOperation;
  release: ProposalOperation | null;
}

/** Active rentals on this API host (`GET /v1/burst/operations`). */
export interface BurstRentals {
  scope: 'local_api_host';
  observedAt: number;
  rentals: BurstOperation[];
}

/** Address of one rental: its provider and approval operation. */
export interface BurstOperationAddress {
  pluginId: string;
  nodeId: string;
  operationId: string;
}

/** Phases after which a rental needs no more polling. `ready` is not one: the
 *  provider can still release an idle rental, or its expiry end it. */
export const SETTLED_BURST_PHASES: readonly BurstPhase[] = ['refused', 'released'];

const headers = { 'X-Skulk-Dashboard': 'pairing-v1' };
const operationPath = ({ operationId }: BurstOperationAddress) => `/v1/burst/operations/${encodeURIComponent(operationId)}`;
const operationParams = ({ pluginId, nodeId }: BurstOperationAddress) => ({ plugin_id: pluginId, node_id: nodeId });

const burstApi = apiSlice.injectEndpoints({
  endpoints: (build) => ({
    getBurstProviders: build.query<BurstProviders, void>({
      query: () => ({ url: '/v1/burst/providers', headers, cache: 'no-store' }),
      keepUnusedDataFor: 0,
    }),
    recommendBurst: build.mutation<BurstRecommendation, { modelId: string; contextTokens: number; excludedNodeIds?: string[] }>({
      query: (body) => ({ url: '/v1/burst/recommendations', headers, method: 'POST', body }),
    }),
    approveBurst: build.mutation<BurstOperation, { operationId: string; reference: ProposalReference; approvalRevision: string }>({
      query: (body) => ({ url: '/v1/burst/operations', headers, method: 'POST', body }),
      invalidatesTags: ['BurstRentals'],
    }),
    getBurstOperation: build.query<BurstOperation, BurstOperationAddress>({
      query: (address) => ({ url: operationPath(address), params: operationParams(address), headers, cache: 'no-store' }),
      keepUnusedDataFor: 0,
    }),
    releaseBurst: build.mutation<BurstOperation, BurstOperationAddress>({
      query: (address) => ({ url: operationPath(address), params: operationParams(address), headers, method: 'DELETE' }),
      invalidatesTags: ['BurstRentals'],
    }),
    getBurstRentals: build.query<BurstRentals, void>({
      query: () => ({ url: '/v1/burst/operations', headers, cache: 'no-store' }),
      providesTags: ['BurstRentals'],
    }),
  }),
});

export const {
  useGetBurstProvidersQuery,
  useRecommendBurstMutation,
  useApproveBurstMutation,
  useGetBurstOperationQuery,
  useReleaseBurstMutation,
  useGetBurstRentalsQuery,
} = burstApi;
