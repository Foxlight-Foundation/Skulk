import { useEffect, useMemo, useRef, useState } from 'react';
import type { TopologyData, NodeInfo, TopologyEdge } from '../types/topology';
import {
  useGetLocalNodeIdQuery,
  useGetLocalNodeIdentityQuery,
  useGetRawStateQuery,
  type RawLocalNodeIdentityResponse,
  type RawNetworkInterfaceInfo,
  type RawNodeIdentity,
  type RawNodeNetworkInfo,
  type RawMemoryUsage,
  type RawSystemPerformanceProfile,
  type RawTopology,
  type RawThunderboltBridge,
  type RawThunderboltInfo,
  type RawRdmaCtl,
  type RawConnectionEdge,
} from '../store/endpoints/cluster';

/* ── Transforms ──────────────────────────────────────────── */

function extractAddresses(iface: RawNetworkInterfaceInfo): string[] {
  const addrs: string[] = [];
  if (iface.ipAddress) addrs.push(iface.ipAddress);
  if (iface.addresses) {
    for (const a of iface.addresses) {
      if (typeof a === 'string') addrs.push(a);
      else if (a?.address) addrs.push(a.address);
    }
  }
  if (iface.ipAddresses) addrs.push(...iface.ipAddresses);
  if (iface.ips) addrs.push(...iface.ips);
  return [...new Set(addrs)];
}

function extractIpFromMultiaddr(addr?: string): string | undefined {
  if (!addr) return undefined;
  const match = addr.match(/\/ip[46]\/([\d.]+|[a-fA-F0-9:]+)/);
  return match?.[1];
}

function normalizeNodeLabel(value?: string): string {
  return (value ?? '')
    .trim()
    .toLowerCase()
    .replace(/\.local$/, '')
    .replace(/[\s._-]+/g, '');
}

function transformTopology(
  raw: RawTopology,
  identities: Record<string, RawNodeIdentity>,
  memory: Record<string, RawMemoryUsage>,
  system: Record<string, RawSystemPerformanceProfile>,
  network: Record<string, RawNodeNetworkInfo>,
  tbBridge: Record<string, RawThunderboltBridge>,
  rdmaCtl: Record<string, RawRdmaCtl>,
): TopologyData {
  const nodes: Record<string, NodeInfo> = {};
  const edges: TopologyEdge[] = [];

  for (const nodeId of raw.nodes ?? []) {
    if (!nodeId) continue;

    const identity = identities[nodeId];
    const mem = memory[nodeId];
    const sys = system[nodeId];
    const net = network[nodeId];

    const ramTotal = mem?.ramTotal?.inBytes ?? 0;
    const ramAvailable = mem?.ramAvailable?.inBytes ?? 0;
    const ramUsage = Math.max(ramTotal - ramAvailable, 0);

    const rawIfaces = net?.interfaces ?? [];
    const networkInterfaces = rawIfaces.map((iface) => ({
      name: iface.name,
      addresses: extractAddresses(iface),
    }));

    const ipToInterface: Record<string, string> = {};
    for (const iface of networkInterfaces) {
      for (const addr of iface.addresses ?? []) {
        ipToInterface[addr] = iface.name ?? '';
      }
    }

    nodes[nodeId] = {
      system_info: {
        model_id: identity?.modelId ?? 'Unknown',
        chip: identity?.chipId,
        memory: ramTotal,
      },
      network_interfaces: networkInterfaces,
      ip_to_interface: ipToInterface,
      macmon_info: {
        memory: { ram_usage: ramUsage, ram_total: ramTotal },
        temp: sys?.temp != null ? { gpu_temp_avg: Math.max(30, sys.temp) } : undefined,
        gpu_usage: sys?.gpuUsage != null && sys.gpuUsage > 0 ? [0, sys.gpuUsage] : undefined,
        sys_power: sys?.sysPower,
      },
      last_macmon_update: Date.now() / 1000,
      friendly_name: identity?.friendlyName,
      os_version: identity?.osVersion,
      os_build_version: identity?.osBuildVersion,
      exo_version: identity?.exoVersion,
      exo_commit: identity?.exoCommit,
      thunderbolt_bridge: tbBridge[nodeId]?.enabled ?? false,
      rdma_enabled: rdmaCtl[nodeId]?.enabled ?? false,
      rdma_interfaces_present: rdmaCtl[nodeId]?.interfacesPresent ?? true,
    };
  }

  const connections = raw.connections;
  if (connections) {
    for (const [source, sinks] of Object.entries(connections)) {
      if (!sinks || typeof sinks !== 'object') continue;
      for (const [sink, edgeList] of Object.entries(sinks)) {
        if (!Array.isArray(edgeList)) continue;
        for (const edge of edgeList) {
          if (!edge || typeof edge !== 'object') continue;

          let sendBackIp: string | undefined;
          let sourceRdmaIface: string | undefined;
          let sinkRdmaIface: string | undefined;

          if ('sinkMultiaddr' in edge && edge.sinkMultiaddr) {
            const ma = edge.sinkMultiaddr as { ipAddress?: string; address?: string };
            sendBackIp = ma.ipAddress ?? extractIpFromMultiaddr(ma.address);
          } else if ('sourceRdmaIface' in edge) {
            sourceRdmaIface = (edge as RawConnectionEdge).sourceRdmaIface;
            sinkRdmaIface = (edge as RawConnectionEdge).sinkRdmaIface;
          }

          if (nodes[source] && nodes[sink] && source !== sink) {
            let sendBackInterface: string | undefined;
            if (sendBackIp) {
              sendBackInterface =
                nodes[source]?.ip_to_interface?.[sendBackIp] ??
                nodes[sink]?.ip_to_interface?.[sendBackIp];
            }
            edges.push({ source, target: sink, sendBackIp, sendBackInterface, sourceRdmaIface, sinkRdmaIface });
          }
        }
      }
    }
  }

  return { nodes, edges };
}

function ensureLocalNodePresent(
  topology: TopologyData,
  localNodeId: string | null,
  localNodeIdentity: RawLocalNodeIdentityResponse | null,
  identities: Record<string, RawNodeIdentity>,
): TopologyData {
  if (!localNodeId) return topology;
  if (topology.nodes[localNodeId]) return topology;

  const normalizedLocalHostname = normalizeNodeLabel(localNodeIdentity?.hostname);
  const localIpAddress = localNodeIdentity?.ipAddress?.trim();
  const matchedRealLocalNode = Object.values(topology.nodes).some((node) => {
    const normalizedFriendlyName = normalizeNodeLabel(node.friendly_name);
    if (
      normalizedLocalHostname.length > 0 &&
      normalizedFriendlyName.length > 0 &&
      normalizedFriendlyName === normalizedLocalHostname
    ) {
      return true;
    }
    if (!localIpAddress || localIpAddress.length === 0) return false;
    return (node.network_interfaces ?? []).some((iface) =>
      (iface.addresses ?? []).includes(localIpAddress),
    );
  });

  if (matchedRealLocalNode) return topology;

  const localIdentity = identities[localNodeId];
  return {
    nodes: {
      ...topology.nodes,
      [localNodeId]: {
        system_info: {
          model_id: localIdentity?.modelId ?? 'Unknown',
          chip: localIdentity?.chipId,
          memory: 0,
        },
        network_interfaces: [],
        ip_to_interface: {},
        macmon_info: {
          memory: { ram_usage: 0, ram_total: 0 },
        },
        last_macmon_update: Date.now() / 1000,
        friendly_name: localIdentity?.friendlyName ?? 'Local node (syncing)',
        os_version: localIdentity?.osVersion,
        os_build_version: localIdentity?.osBuildVersion,
        exo_version: localIdentity?.exoVersion,
        exo_commit: localIdentity?.exoCommit,
        syncing: true,
      },
    },
    edges: topology.edges,
  };
}

/* ── Public types ────────────────────────────────────────── */

export type RawDownloads = Record<string, unknown[]>;
export type NodeDiskInfo = Record<string, { total: { inBytes: number }; available: { inBytes: number } }>;

export interface RawShardAssignments {
  modelId?: string;
  nodeToRunner?: Record<string, string>;
  runnerToShard?: Record<string, Record<string, unknown>>;
}

export interface RawInstanceInner {
  instanceId?: string;
  shardAssignments?: RawShardAssignments;
}

export type RawInstances = Record<
  string,
  { MlxRingInstance?: RawInstanceInner; MlxJacclInstance?: RawInstanceInner }
>;

export type RawRunners = Record<string, Record<string, unknown>>;

export interface ClusterState {
  topology: TopologyData | null;
  connected: boolean;
  lastUpdate: number | null;
  downloads: RawDownloads;
  nodeDisk: NodeDiskInfo;
  instances: RawInstances;
  runners: RawRunners;
  nodeThunderbolt: Record<string, RawThunderboltInfo>;
  nodeThunderboltBridge: Record<string, RawThunderboltBridge>;
  nodeRdmaCtl: Record<string, RawRdmaCtl>;
  thunderboltBridgeCycles: string[][];
}

const CONNECTION_LOST_THRESHOLD = 3;
const POLL_INTERVAL = 1000;

/**
 * Polls `/state` and exposes a normalized view of the cluster.
 *
 * Backed by RTK Query under the hood — components calling this share a single
 * cache entry, so opening multiple topology surfaces doesn't fan out into
 * multiple HTTP requests. Polling is driven by RTK Query's `pollingInterval`.
 *
 * `connected` flips false after `CONNECTION_LOST_THRESHOLD` consecutive
 * failures so the dashboard's connection indicator doesn't oscillate on a
 * single bad poll.
 */
export function useClusterState(): ClusterState {
  const stateQuery = useGetRawStateQuery(undefined, { pollingInterval: POLL_INTERVAL });
  const nodeIdQuery = useGetLocalNodeIdQuery();
  const nodeIdentityQuery = useGetLocalNodeIdentityQuery();

  // Failure counter rides on top of the query state so we can require a
  // sustained outage before flipping `connected` to false.
  const failuresRef = useRef(0);
  const [connected, setConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);

  useEffect(() => {
    if (stateQuery.isError) {
      failuresRef.current += 1;
      if (failuresRef.current >= CONNECTION_LOST_THRESHOLD) {
        setConnected(false);
      }
    } else if (stateQuery.isSuccess && !stateQuery.isFetching) {
      failuresRef.current = 0;
      setConnected(true);
      setLastUpdate(Date.now());
    }
    // We only care about the transitions of the query; the query object
    // itself is replaced each render but its boolean status fields are stable.
  }, [stateQuery.isError, stateQuery.isSuccess, stateQuery.isFetching]);

  // Resolved local node id: prefer the value from /node/identity (which
  // includes hostname + ip + node id) over /node_id alone.
  const resolvedLocalNodeId = useMemo(() => {
    if (nodeIdentityQuery.data?.nodeId) return nodeIdentityQuery.data.nodeId;
    if (nodeIdQuery.data) return nodeIdQuery.data;
    return null;
  }, [nodeIdentityQuery.data, nodeIdQuery.data]);

  const data = stateQuery.data;

  const topology = useMemo<TopologyData | null>(() => {
    if (!data?.topology) return null;
    const transformed = transformTopology(
      data.topology,
      data.nodeIdentities ?? {},
      data.nodeMemory ?? {},
      data.nodeSystem ?? {},
      data.nodeNetwork ?? {},
      data.nodeThunderboltBridge ?? {},
      data.nodeRdmaCtl ?? {},
    );
    return ensureLocalNodePresent(
      transformed,
      resolvedLocalNodeId,
      nodeIdentityQuery.data ?? null,
      data.nodeIdentities ?? {},
    );
  }, [data, resolvedLocalNodeId, nodeIdentityQuery.data]);

  return {
    topology,
    connected,
    lastUpdate,
    downloads: (data?.downloads ?? {}) as RawDownloads,
    nodeDisk: (data?.nodeDisk ?? {}) as NodeDiskInfo,
    instances: (data?.instances ?? {}) as RawInstances,
    runners: (data?.runners ?? {}) as RawRunners,
    nodeThunderbolt: data?.nodeThunderbolt ?? {},
    nodeThunderboltBridge: data?.nodeThunderboltBridge ?? {},
    nodeRdmaCtl: data?.nodeRdmaCtl ?? {},
    thunderboltBridgeCycles: data?.thunderboltBridgeCycles ?? [],
  };
}
