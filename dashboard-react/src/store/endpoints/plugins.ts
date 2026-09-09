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

const headers = { 'X-Skulk-Dashboard': 'pairing-v1' };
const nodePath = ({ pluginId, nodeId }: NodeAddress) =>
  `/v1/plugins/${encodeURIComponent(pluginId)}/nodes/${encodeURIComponent(nodeId)}/configuration`;

const pluginsApi = apiSlice.injectEndpoints({
  endpoints: (build) => ({
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

export const { useGetPluginNodesQuery, useGetNodeConfigurationQuery, useConfigurePluginNodeMutation } = pluginsApi;
