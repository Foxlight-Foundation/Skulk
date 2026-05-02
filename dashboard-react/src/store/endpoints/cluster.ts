import { apiSlice } from '../api';

/* ── Raw response types (camelCase from Python CamelCaseModel) ── */

export interface RawNodeIdentity {
  modelId?: string;
  chipId?: string;
  friendlyName?: string;
  osVersion?: string;
  osBuildVersion?: string;
  exoVersion?: string;
  exoCommit?: string;
}

export interface RawMemoryUsage {
  ramTotal?: { inBytes: number };
  ramAvailable?: { inBytes: number };
}

export interface RawSystemPerformanceProfile {
  gpuUsage?: number;
  temp?: number;
  sysPower?: number;
  pcpuUsage?: number;
  ecpuUsage?: number;
}

export interface RawNetworkInterfaceInfo {
  name?: string;
  ipAddress?: string;
  addresses?: Array<{ address?: string } | string>;
  ipAddresses?: string[];
  ips?: string[];
}

export interface RawNodeNetworkInfo {
  interfaces?: RawNetworkInterfaceInfo[];
}

export interface RawConnectionEdge {
  sinkMultiaddr?: { address?: string; ipAddress?: string };
  sourceRdmaIface?: string;
  sinkRdmaIface?: string;
}

export interface RawTopology {
  nodes?: string[];
  connections?: Record<string, Record<string, RawConnectionEdge[]>>;
}

export interface RawThunderboltBridge {
  enabled: boolean;
  exists: boolean;
  serviceName?: string | null;
}

export interface RawThunderboltInterface {
  rdmaInterface: string;
  domainUuid: string;
  linkSpeed: string;
}

export interface RawThunderboltInfo {
  interfaces: RawThunderboltInterface[];
}

export interface RawRdmaCtl {
  enabled: boolean;
  interfacesPresent?: boolean;
}

export interface RawStateResponse {
  topology?: RawTopology;
  instances?: Record<string, unknown>;
  runners?: Record<string, unknown>;
  downloads?: Record<string, unknown[]>;
  nodeIdentities?: Record<string, RawNodeIdentity>;
  nodeMemory?: Record<string, RawMemoryUsage>;
  nodeSystem?: Record<string, RawSystemPerformanceProfile>;
  nodeNetwork?: Record<string, RawNodeNetworkInfo>;
  nodeDisk?: Record<string, { total: { inBytes: number }; available: { inBytes: number } }>;
  nodeThunderbolt?: Record<string, RawThunderboltInfo>;
  nodeThunderboltBridge?: Record<string, RawThunderboltBridge>;
  nodeRdmaCtl?: Record<string, RawRdmaCtl>;
  thunderboltBridgeCycles?: string[][];
}

export interface RawLocalNodeIdentityResponse {
  nodeId?: string;
  hostname?: string;
  ipAddress?: string;
}

/* ── Endpoints ── */

/**
 * Cluster-state endpoints injected into the root API slice. Polling is
 * controlled by the caller (LiveTab uses { skip: !panelOpen }, topology view
 * polls always). The transform stays out of the cache so consumers can read
 * the raw shape and run their own derivations.
 */
export const clusterApi = apiSlice.injectEndpoints({
  endpoints: (build) => ({
    getRawState: build.query<RawStateResponse, void>({
      query: () => '/state',
      providesTags: ['ClusterState'],
    }),
    getLocalNodeId: build.query<string, void>({
      // The /node_id endpoint returns the bare node id as a JSON string. RTK
      // Query parses JSON automatically; the response type is therefore a
      // plain string.
      query: () => '/node_id',
    }),
    getLocalNodeIdentity: build.query<RawLocalNodeIdentityResponse, void>({
      query: () => '/node/identity',
    }),
  }),
});

export const {
  useGetRawStateQuery,
  useGetLocalNodeIdQuery,
  useGetLocalNodeIdentityQuery,
} = clusterApi;
