import { apiSlice } from '../api';

/** Whether this host's plugin manager is set up and answering (`GET /v1/plugins/managed/service`). */
export interface PluginServiceStatus {
  /** absent: never set up; setting_up; ready; unavailable: set up but not answering; failed; unsupported. */
  state: 'absent' | 'setting_up' | 'ready' | 'unavailable' | 'failed' | 'unsupported';
  /** `user` (a per-user service) or `system`, once set up. */
  scope: 'user' | 'system' | null;
  /** The current step while setup runs. */
  progress: string | null;
  /** Why the last setup failed or the host is unsupported. */
  error: string | null;
}

/** Stable installed node identity, independent of its capabilities and transport. */
export interface ConfigurableNode {
  nodeId: string;
  bundleId: string;
  version: string;
  status: string;
  configurable: boolean;
  credentialsConfigurable?: boolean;
  preflightAvailable?: boolean;
  setupAvailable?: boolean;
  setupActionsAvailable?: boolean;
  proposalsAvailable?: boolean;
  proposalActionsAvailable?: boolean;
}

/** A plugin's installed nodes remain visible while disabled or unavailable. */
export interface PluginNodes {
  pluginId: string;
  nodes: ConfigurableNode[];
  available: boolean;
}

/** Ordinary settings only; credential values use a separate write-only contract. */
export interface NodeConfiguration {
  nodeId: string;
  revision: number;
  schemaDigest: string;
  configurationSchema: Record<string, unknown>;
  values: Record<string, unknown>;
  enabled: boolean;
}

/** An exact node, rather than a capability method or transient peer address. */
export interface NodeAddress { pluginId: string; nodeId: string }

/** Opaque provider journal identity; it contains no executable input or proof. */
export interface ProposalReference extends NodeAddress { proposalId: string; proposalDigest: string }

/** Bounded plain-text metadata for selecting an immutable proposal. */
export interface ProposalSummary { reference: ProposalReference; summary: string; expiresAt: number; state: 'pending_approval' | 'approved' | 'execution_recorded' | 'expired' | 'unavailable' }

/** Exact reviewed terms; a missing fence means approval is unavailable. */
export interface ProposalReview { proposal: ProposalSummary; observedAt: number; approvalRevision: string | null; fields: { label: string; value: string }[] }

/** Original owner intent, accepted once under a durable operation identity. */
export interface ProposalApproval { operationId: string; reference: ProposalReference; reviewRevision: string }

/** Safe operation observation; uncertain dispatch must never be replayed. */
/** Historical cleanup evidence, independent of the original submission result. */
export interface ProposalReconciliation { state: 'pending' | 'active' | 'releasing' | 'absent' | 'attention' | 'unknown'; observedAt: number | null; stale: boolean; code: string | null }
export interface ProposalOperation { operationId: string; reference: ProposalReference; phase: 'accepted' | 'approving' | 'approved' | 'dispatching' | 'acknowledged' | 'succeeded' | 'refused' | 'approval_interrupted' | 'uncertain'; reconciliation?: ProposalReconciliation | null; updatedAt: number; code: string | null; correctiveAction: string | null }

/** Public setup files from the installed owner; never private keys or credentials. */
export interface NodeSetup {
  nodeId: string;
  revision: number;
  schemaDigest: string;
  credentialRevision: number;
  artifacts: { name: string; title: string; mediaType: 'application/json' | 'text/plain'; content: string }[];
}

/** Fixed nonbillable setup form supplied by the installed owner. */
export interface SetupAction { requiresApproval?: boolean; actionId: string; title: string; description: string; parametersSchema: Record<string, unknown>; schemaDigest: string }

/** Safe retained setup observation; complete is separate from preflight and enablement. */
export interface SetupOperation { requiresApproval?: boolean; operationId: string; nodeId: string; actionId: string; phase: 'queued' | 'running' | 'complete' | 'failed'; code: string | null; correctiveAction: string | null }

/** Current setup forms and revision fences, read without starting work. */
export interface SetupActions { nodeId: string; revision: number; schemaDigest: string; credentialRevision: number; credentialSchemaDigest: string; actions: SetupAction[]; operations: SetupOperation[] }

/** Immutable ordinary setup intent; credentials use their existing separate endpoint. */
export interface SetupMutation { expectedRequiresApproval?: boolean; operationId: string; actionId: string; expectedRevision: number; expectedSchemaDigest: string; expectedCredentialRevision: number; expectedCredentialSchemaDigest: string; expectedActionSchemaDigest: string; values: Record<string, unknown> }

/** Declared reference readiness only; credential values are never query/cache data. */
export interface NodeCredentials {
  nodeId: string;
  revision: number;
  schemaDigest: string;
  credentials: { credentialId: string; title: string; description: string; required: boolean; ready: boolean }[];
}

/** Safe prerequisite observations; enabling always performs fresh server checks. */
export interface NodePreflight {
  nodeId: string;
  revision: number;
  schemaDigest: string;
  valuesDigest: string;
  credentialRevision: number | null;
  observedAt: number;
  checks: { code: string; passed: boolean; correctiveAction: string | null }[];
}

/** Compare-and-set mutation shared with the plugin's terminal configuration store. */
export interface ConfigurationMutation extends NodeAddress {
  operation: 'validate' | 'edit' | 'enable' | 'disable';
  expectedRevision: number;
  expectedSchemaDigest: string;
  values?: Record<string, unknown>;
}

/** Name, version and publisher of an installation's selected signed release. Display only. */
export interface InstalledRelease {
  bundle_id: string;
  /** Title from the signed manifest; null when the manifest declares none. */
  title: string | null;
  bundle_version: string;
  publisher: string;
  sequence: number;
}

/** Public process observation; a running process alone does not establish readiness. */
export interface ManagedRuntime {
  plugin_id: string;
  /** The selected signed release, when its staged metadata is readable. */
  release?: InstalledRelease | null;
  selected_digest: string | null;
  selection_revision: number;
  enabled: boolean;
  /** Uninstalled runtimes retain cleanup state and can be explicitly reinstalled. */
  uninstalled?: boolean;
  stale: boolean;
  error_code: string | null;
  operation_id: string | null;
  operation_state: ManagedOperation['state'] | null;
  /** `error_code` names why a failed process stopped, such as `verification_failed`. */
  service: { state: string; active_digest: string | null; observed_at: number; error_code?: string | null } | null;
}

/** Durable local operation reference; provider submissions and approvals are separate. */
export interface ManagedOperation {
  request: { operation_id: string; action: 'activate' | 'select' | 'disable' | 'uninstall' };
  state: 'accepted' | 'applying' | 'complete' | 'failed' | 'recovery_required' | 'superseded';
  error_code: string | null;
  withdraws_operation_id?: string | null;
}

/** Signed review returned before artifact download; no credentials or executable paths. */
export interface RuntimeRelease {
  runtime_digest: string;
  source_revision: number;
  publisher: string;
  bundle_id: string;
  version: string;
  sequence: number;
  platform: string;
  python_requires: string;
  skulk_build_sha256: string;
  permissions: string[];
  artifact_bytes: number;
  expires_at: number;
}

/** Original server-owned staging intent, separate from activation and paid approval. */
export interface RuntimeInstallation {
  attempt?: number;
  attempt_source_revision?: number | null;
  request: { operation_id: string; runtime_digest: string; expected_source_revision: number };
  review: RuntimeRelease;
  state: 'accepted' | 'downloading' | 'staging' | 'staged' | 'recovery_required';
  downloaded_bytes: number;
  error_code: string | null;
}

/** Address an operation already retained by the installed manager. */
export interface ManagedOperationAddress { pluginId: string; operationId: string }

/** Source readiness only; no stored credential values or local paths. */
export interface RuntimeSourceStatus {
  revision: number;
  configured: boolean;
  credential_reference: string | null;
  credential_ready: boolean;
  trust_revision: number | null;
}

/** One release a signed catalog lists: the consent facts, never an address or credential. */
export interface CatalogEntry {
  bundle_id: string;
  bundle_version: string;
  /** Title from the signed record; null when the release declares none. */
  title: string | null;
  publisher: string;
  sequence: number;
  release_digest: string;
  /** Exact artifact family of a runtime-bearing release, such as `macos-arm64`. */
  runtime_platform: string | null;
  artifact_sha256: string;
  artifact_bytes: number;
  /** The artifact plus every wheel the runtime needs: what a download costs. */
  transfer_bytes: number;
  platforms: string[];
  /** Skulk build the publisher qualified the release against; provenance only. */
  skulk_build_sha256: string;
  /** Skulk versions the release runs on; null for an older listing tied to one build. */
  skulk_requires: string | null;
  permissions: string[];
  /** Capability ids it serves, such as `images.edit@1.0.0`. */
  descriptors: string[];
  /** Titles of the screens it opens. */
  surfaces: string[];
  /** Whether it runs durable operations that survive restarts. */
  operations: boolean;
  /** Steward risk classes; `billable` means an action can spend money. */
  steward_risks: string[];
  expires_at: number;
  /** Whether this release fits this host: platform, and this host's Skulk version in its range. */
  matches_host: boolean;
}

/** A catalog read and verified by the host against its discovery trust. */
export interface CatalogListing {
  publisher: string;
  revision: number;
  created_at: number;
  expires_at: number;
  /** Digest of the verified document; an install names it as the reviewed listing. */
  catalog_sha256: string;
  entries: CatalogEntry[];
}

/** Owner consent to bind one installation to one reviewed listing. */
export interface CatalogInstallRequest {
  catalog_sha256: string;
  bundle_id: string;
  sequence: number;
  runtime_platform?: string | null;
  /** An existing installation of the same bundle to upgrade; omitted registers a new one. */
  plugin_id?: string;
}

/** The bound installation, its source readiness and the signed release it will stage. */
export interface CatalogInstallation {
  plugin_id: string;
  listing: CatalogEntry;
  source: RuntimeSourceStatus;
  review: RuntimeRelease;
}

/** Owner catalog address and discovery trust; the credential is write-only. */
export interface CatalogSourceUpdate {
  expected_revision: number;
  base_url?: string;
  document_filename?: string;
  trust?: { revision: number; expires_at: number; publishers: Record<string, string> };
  token?: string;
  clear_token?: boolean;
}

/**
 * The server's own sentence for a refused plugin request, when it sent one.
 *
 * Catalog refusals, owner-access refusals and protocol refusals carry a
 * sentence that names the cause and the next step; everything else returns
 * null so the caller shows its own conservative text.
 */
export function pluginRefusalDetail(error: unknown): string | null {
  if (!error || typeof error !== 'object' || !('data' in error)) return null;
  const data = (error as { data?: unknown }).data;
  if (!data || typeof data !== 'object') return null;
  // Route errors carry `detail`; request-validation errors carry `error.message`.
  const detail = 'detail' in data ? (data as { detail?: unknown }).detail
    : 'error' in data && (data as { error?: unknown }).error && typeof (data as { error?: unknown }).error === 'object'
      ? ((data as { error: { message?: unknown } }).error.message) : undefined;
  return typeof detail === 'string' && detail.length > 0 && detail.length <= 600 ? detail : null;
}

/**
 * Whether a failed plugin request was definitely answered with a refusal.
 *
 * A 4xx means the host decided and nothing changed, so the caller reports
 * it. A 5xx, a timeout or a lost connection may have landed, so the caller
 * reads the operation back instead of assuming either way.
 */
export function pluginRequestRefused(error: unknown): number | null {
  if (!error || typeof error !== 'object' || !('status' in error)) return null;
  const status = (error as { status?: unknown }).status;
  return typeof status === 'number' && status >= 400 && status < 500 ? status : null;
}

const headers = { 'X-Skulk-Dashboard': 'pairing-v1' };
const nodePath = ({ pluginId, nodeId }: NodeAddress) =>
  `/v1/plugins/${encodeURIComponent(pluginId)}/nodes/${encodeURIComponent(nodeId)}/configuration`;

const pluginsApi = apiSlice.injectEndpoints({
  endpoints: (build) => ({
    getNodeProposals: build.query<{ proposals: ProposalSummary[]; nextOffset: number | null }, NodeAddress & { offset: number }>({
      query: ({ offset, ...address }) => ({ url: nodePath(address).replace(/configuration$/, 'proposals'), params: { offset }, headers, cache: 'no-store' }),
      keepUnusedDataFor: 0,
    }),
    getNodeProposalReview: build.query<ProposalReview, ProposalReference>({
      query: ({ proposalId, proposalDigest, ...address }) => ({ url: nodePath(address).replace(/configuration$/, `proposals/${encodeURIComponent(proposalId)}`), params: { proposal_digest: proposalDigest }, headers, cache: 'no-store' }),
      keepUnusedDataFor: 0,
    }),
    approveNodeProposal: build.mutation<ProposalOperation, ProposalApproval>({
      query: (body) => ({ url: nodePath(body.reference).replace(/configuration$/, `proposals/${encodeURIComponent(body.reference.proposalId)}/approve`), headers, method: 'POST', body }),
    }),
    getNodeProposalOperation: build.query<ProposalOperation, NodeAddress & { operationId: string }>({
      query: ({ operationId, ...address }) => ({ url: nodePath(address).replace(/configuration$/, `proposal-operations/${encodeURIComponent(operationId)}`), headers, cache: 'no-store' }),
      keepUnusedDataFor: 0,
    }),
    resumeNodeProposal: build.mutation<ProposalOperation, NodeAddress & { operationId: string }>({
      query: ({ operationId, ...address }) => ({ url: nodePath(address).replace(/configuration$/, `proposal-operations/${encodeURIComponent(operationId)}/resume`), headers, method: 'POST' }),
    }),
    getNodeSetupActions: build.query<SetupActions, NodeAddress>({
      query: (address) => ({ url: nodePath(address).replace(/configuration$/, 'setup-actions'), headers, cache: 'no-store' }),
      providesTags: ['Plugins'],
      keepUnusedDataFor: 0,
    }),
    startNodeSetup: build.mutation<SetupOperation, NodeAddress & { mutation: SetupMutation }>({
      query: ({ mutation, ...address }) => ({ url: nodePath(address).replace(/configuration$/, 'setup-operations'), headers, method: 'POST', body: mutation }),
      invalidatesTags: ['Plugins'],
    }),
    resumeNodeSetup: build.mutation<SetupOperation, NodeAddress & { operationId: string }>({
      query: ({ operationId, ...address }) => ({ url: nodePath(address).replace(/configuration$/, `setup-operations/${encodeURIComponent(operationId)}/resume`), headers, method: 'POST' }),
      invalidatesTags: ['Plugins'],
    }),
    getNodeSetup: build.query<NodeSetup, NodeAddress>({
      query: ({ pluginId, nodeId }) => ({ url: `/v1/plugins/${encodeURIComponent(pluginId)}/nodes/${encodeURIComponent(nodeId)}/setup`, headers, cache: 'no-store' }),
      providesTags: ['Plugins'],
    }),
    getNodePreflight: build.query<NodePreflight, NodeAddress>({
      query: ({ pluginId, nodeId }) => ({ url: `/v1/plugins/${encodeURIComponent(pluginId)}/nodes/${encodeURIComponent(nodeId)}/preflight`, headers, cache: 'no-store' }),
    }),
    getNodeCredentials: build.query<NodeCredentials, NodeAddress>({
      query: ({ pluginId, nodeId }) => ({ url: `/v1/plugins/${encodeURIComponent(pluginId)}/nodes/${encodeURIComponent(nodeId)}/credentials`, headers, cache: 'no-store' }),
      providesTags: ['Plugins'],
    }),
    registerManagedRuntime: build.mutation<ManagedRuntime, string>({
      query: (pluginId) => ({ url: '/v1/plugins/managed/installations', method: 'POST', headers, body: { plugin_id: pluginId } }),
      invalidatesTags: ['Plugins'],
    }),
    getRuntimeSourceStatus: build.query<RuntimeSourceStatus, string>({
      query: (pluginId) => ({ url: `/v1/plugins/managed/installations/${encodeURIComponent(pluginId)}/source`, headers, cache: 'no-store' }),
      providesTags: ['Plugins'],
    }),
    recoverRuntimeInstallation: build.mutation<RuntimeInstallation, ManagedOperationAddress & { expectedSourceRevision: number }>({
      query: ({ pluginId, operationId, expectedSourceRevision }) => ({
        url: `/v1/plugins/managed/installations/${encodeURIComponent(pluginId)}/install/${encodeURIComponent(operationId)}/recover`, method: 'POST', headers,
        body: { expected_source_revision: expectedSourceRevision },
      }),
      invalidatesTags: ['Plugins'],
    }),
    getRuntimeRelease: build.query<RuntimeRelease, string>({
      query: (pluginId) => ({ url: `/v1/plugins/managed/installations/${encodeURIComponent(pluginId)}/release`, headers, cache: 'no-store' }),
    }),
    getRuntimeInstallation: build.query<{ operation: RuntimeInstallation | null }, string>({
      query: (pluginId) => ({ url: `/v1/plugins/managed/installations/${encodeURIComponent(pluginId)}/install`, headers, cache: 'no-store' }),
      providesTags: ['Plugins'],
    }),
    installRuntimeRelease: build.mutation<RuntimeInstallation, { pluginId: string; request: RuntimeInstallation['request'] }>({
      query: ({ pluginId, request }) => ({ url: `/v1/plugins/managed/installations/${encodeURIComponent(pluginId)}/install`, method: 'POST', headers, body: request }),
      invalidatesTags: ['Plugins'],
    }),
    activateRuntimeRelease: build.mutation<ManagedOperation, { pluginId: string; operationId: string; expectedRevision: number; runtimeDigest: string; rollback: boolean; action?: 'activate' | 'select' }>({
      query: ({ pluginId, operationId, expectedRevision, runtimeDigest, rollback, action = 'activate' }) => ({
        url: `/v1/plugins/managed/installations/${encodeURIComponent(pluginId)}/operations`, method: 'POST', headers,
        body: { operation_id: operationId, action, expected_revision: expectedRevision, runtime_digest: runtimeDigest, rollback, accept_permissions: true },
      }),
      invalidatesTags: ['Plugins'],
    }),
    getManagedRuntimes: build.query<{ installations: ManagedRuntime[] }, void>({
      query: () => ({ url: '/v1/plugins/managed', headers, cache: 'no-store' }),
      providesTags: ['Plugins'],
    }),
    getManagedOperation: build.query<ManagedOperation, ManagedOperationAddress>({
      query: ({ pluginId, operationId }) => ({ url: `/v1/plugins/managed/installations/${encodeURIComponent(pluginId)}/operations/${encodeURIComponent(operationId)}`, headers, cache: 'no-store' }),
      providesTags: ['Plugins'],
    }),
    withdrawManagedRuntime: build.mutation<ManagedOperation, { pluginId: string; operationId: string; expectedRevision: number; action: 'disable' | 'uninstall' }>({
      query: ({ pluginId, operationId, expectedRevision, action }) => ({
        url: `/v1/plugins/managed/installations/${encodeURIComponent(pluginId)}/operations`, method: 'POST', headers,
        body: { operation_id: operationId, action, expected_revision: expectedRevision },
      }),
      // Even an uncertain response needs a fresh server operation reference;
      // invalidation reads state and never replays the mutation.
      invalidatesTags: ['Plugins'],
    }),
    purgeManagedRuntime: build.mutation<{ plugin_id: string; purged: boolean }, string>({
      query: (pluginId) => ({ url: `/v1/plugins/managed/installations/${encodeURIComponent(pluginId)}`, method: 'DELETE', headers }),
      invalidatesTags: ['Plugins'],
    }),
    recoverManagedOperation: build.mutation<ManagedOperation, ManagedOperationAddress>({
      query: ({ pluginId, operationId }) => ({ url: `/v1/plugins/managed/installations/${encodeURIComponent(pluginId)}/operations/${encodeURIComponent(operationId)}/recover`, method: 'POST', headers }),
      invalidatesTags: ['Plugins'],
    }),
    getPluginCatalog: build.query<CatalogListing, void>({
      // Each read fetches and verifies the signed document on the host, so it
      // is read on demand and never polled.
      query: () => ({ url: '/v1/plugins/managed/catalog', headers, cache: 'no-store' }),
      providesTags: ['PluginCatalog'],
      keepUnusedDataFor: 0,
    }),
    getCatalogSource: build.query<RuntimeSourceStatus, void>({
      query: () => ({ url: '/v1/plugins/managed/catalog/source', headers, cache: 'no-store' }),
      providesTags: ['PluginCatalog'],
    }),
    installFromCatalog: build.mutation<CatalogInstallation, CatalogInstallRequest>({
      query: (body) => ({ url: '/v1/plugins/managed/catalog/install', method: 'POST', headers, body }),
      invalidatesTags: ['Plugins'],
    }),
    getPluginService: build.query<PluginServiceStatus, void>({
      query: () => ({ url: '/v1/plugins/managed/service', headers, cache: 'no-store' }),
      providesTags: ['PluginService'],
    }),
    startPluginServiceSetup: build.mutation<PluginServiceStatus, void>({
      query: () => ({ url: '/v1/plugins/managed/service/setup', method: 'POST', headers }),
      invalidatesTags: ['PluginService'],
    }),
    getPluginNodes: build.query<PluginNodes[], void>({
      query: () => ({ url: '/v1/plugins', headers, cache: 'no-store' }),
      providesTags: ['Plugins'],
    }),
    getNodeConfiguration: build.query<NodeConfiguration, NodeAddress>({
      query: (address) => ({ url: nodePath(address), headers, cache: 'no-store' }),
      providesTags: (_result, _error, address) => [{ type: 'PluginConfiguration', id: `${address.pluginId}/${address.nodeId}` }],
    }),
    configurePluginNode: build.mutation<{ configuration: NodeConfiguration; validated: boolean }, ConfigurationMutation>({
      query: ({ pluginId, nodeId, ...body }) => ({
        url: nodePath({ pluginId, nodeId }), method: 'POST', headers, body,
      }),
      invalidatesTags: (_result, error, action) => error || action.operation === 'validate' ? [] : [
        'Plugins', { type: 'PluginConfiguration', id: `${action.pluginId}/${action.nodeId}` },
      ],
    }),
  }),
});

export const { useGetPluginServiceQuery, useStartPluginServiceSetupMutation, useGetPluginCatalogQuery, useGetCatalogSourceQuery, useInstallFromCatalogMutation, useLazyGetManagedOperationQuery, useLazyGetRuntimeInstallationQuery, useLazyGetManagedRuntimesQuery, useGetNodeProposalsQuery, useGetNodeProposalReviewQuery, useApproveNodeProposalMutation, useGetNodeProposalOperationQuery, useResumeNodeProposalMutation, useGetNodeSetupActionsQuery, useStartNodeSetupMutation, useResumeNodeSetupMutation, useLazyGetNodeSetupQuery, useLazyGetNodePreflightQuery, useGetNodeCredentialsQuery, useRegisterManagedRuntimeMutation, useGetRuntimeSourceStatusQuery, useRecoverRuntimeInstallationMutation, useLazyGetRuntimeReleaseQuery, useGetRuntimeInstallationQuery, useInstallRuntimeReleaseMutation, useActivateRuntimeReleaseMutation, useGetManagedRuntimesQuery, useGetManagedOperationQuery, useWithdrawManagedRuntimeMutation, useRecoverManagedOperationMutation, usePurgeManagedRuntimeMutation, useGetPluginNodesQuery, useGetNodeConfigurationQuery, useConfigurePluginNodeMutation } = pluginsApi;
