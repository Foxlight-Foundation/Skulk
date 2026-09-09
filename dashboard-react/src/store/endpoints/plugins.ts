import { apiSlice } from '../api';

/** Stable installed node identity, independent of its capabilities and transport. */
export interface ConfigurableNode {
  nodeId: string;
  bundleId: string;
  version: string;
  status: string;
  configurable: boolean;
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

/** Compare-and-set mutation shared with the plugin's terminal configuration store. */
export interface ConfigurationMutation extends NodeAddress {
  operation: 'validate' | 'edit' | 'enable' | 'disable';
  expectedRevision: number;
  expectedSchemaDigest: string;
  values?: Record<string, unknown>;
}

/** Public process observation; a running process alone does not establish readiness. */
export interface ManagedRuntime {
  plugin_id: string;
  selected_digest: string | null;
  selection_revision: number;
  enabled: boolean;
  stale: boolean;
  error_code: string | null;
  operation_id: string | null;
  operation_state: ManagedOperation['state'] | null;
  service: { state: string; active_digest: string | null; observed_at: number } | null;
}

/** Durable local operation reference; provider submissions and approvals are separate. */
export interface ManagedOperation {
  request: { operation_id: string; action: 'activate' | 'disable' };
  state: 'accepted' | 'applying' | 'complete' | 'failed' | 'recovery_required';
  error_code: string | null;
}

/** Address an operation already retained by the installed manager. */
export interface ManagedOperationAddress { pluginId: string; operationId: string }

const headers = { 'X-Skulk-Dashboard': 'pairing-v1' };
const nodePath = ({ pluginId, nodeId }: NodeAddress) =>
  `/v1/plugins/${encodeURIComponent(pluginId)}/nodes/${encodeURIComponent(nodeId)}/configuration`;

const pluginsApi = apiSlice.injectEndpoints({
  endpoints: (build) => ({
    getManagedRuntimes: build.query<{ installations: ManagedRuntime[] }, void>({
      query: () => ({ url: '/v1/plugins/managed', headers, cache: 'no-store' }),
      providesTags: ['Plugins'],
    }),
    getManagedOperation: build.query<ManagedOperation, ManagedOperationAddress>({
      query: ({ pluginId, operationId }) => ({ url: `/v1/plugins/managed/installations/${encodeURIComponent(pluginId)}/operations/${encodeURIComponent(operationId)}`, headers, cache: 'no-store' }),
      providesTags: ['Plugins'],
    }),
    disableManagedRuntime: build.mutation<ManagedOperation, { pluginId: string; operationId: string; expectedRevision: number }>({
      query: ({ pluginId, operationId, expectedRevision }) => ({
        url: `/v1/plugins/managed/installations/${encodeURIComponent(pluginId)}/operations`, method: 'POST', headers,
        body: { operation_id: operationId, action: 'disable', expected_revision: expectedRevision },
      }),
      // Even an uncertain response needs a fresh server operation reference;
      // invalidation reads state and never replays the mutation.
      invalidatesTags: ['Plugins'],
    }),
    recoverManagedOperation: build.mutation<ManagedOperation, ManagedOperationAddress>({
      query: ({ pluginId, operationId }) => ({ url: `/v1/plugins/managed/installations/${encodeURIComponent(pluginId)}/operations/${encodeURIComponent(operationId)}/recover`, method: 'POST', headers }),
      invalidatesTags: ['Plugins'],
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

export const { useGetManagedRuntimesQuery, useGetManagedOperationQuery, useDisableManagedRuntimeMutation, useRecoverManagedOperationMutation, useGetPluginNodesQuery, useGetNodeConfigurationQuery, useConfigurePluginNodeMutation } = pluginsApi;
