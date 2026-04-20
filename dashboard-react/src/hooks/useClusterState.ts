import { useCallback, useEffect, useRef, useState } from 'react';
import type { TopologyData, NodeInfo, TopologyEdge } from '../types/topology';

/* ================================================================
   Raw response types (camelCase from Python CamelCaseModel)
   ================================================================ */

interface RawNodeIdentity {
  modelId?: string;
  chipId?: string;
  friendlyName?: string;
  osVersion?: string;
  osBuildVersion?: string;
  exoVersion?: string;
  exoCommit?: string;
}

interface RawMemoryUsage {
  ramTotal?: { inBytes: number };
  ramAvailable?: { inBytes: number };
}

interface RawSystemPerformanceProfile {
  gpuUsage?: number;
  temp?: number;
  sysPower?: number;
  pcpuUsage?: number;
  ecpuUsage?: number;
}

interface RawNetworkInterfaceInfo {
  name?: string;
  ipAddress?: string;
  addresses?: Array<{ address?: string } | string>;
  ipAddresses?: string[];
  ips?: string[];
}

interface RawNodeNetworkInfo {
  interfaces?: RawNetworkInterfaceInfo[];
}

interface RawConnectionEdge {
  sinkMultiaddr?: { address?: string; ipAddress?: string };
  sourceRdmaIface?: string;
  sinkRdmaIface?: string;
}

interface RawTopology {
  nodes?: string[];
  connections?: Record<string, Record<string, RawConnectionEdge[]>>;
}

interface RawThunderboltBridge {
  enabled: boolean;
  exists: boolean;
  serviceName?: string | null;
}

interface RawThunderboltInterface {
  rdmaInterface: string;
  domainUuid: string;
  linkSpeed: string;
}

interface RawThunderboltInfo {
  interfaces: RawThunderboltInterface[];
}

interface RawRdmaCtl {
  enabled: boolean;
  interfacesPresent?: boolean;
}

interface RawStateResponse {
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

/* ================================================================
   Transformation
   ================================================================ */

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

  // Build edges from nested connection map
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
            // Resolve interface name from IP
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

/* ================================================================
   Hook
   ================================================================ */

const CONNECTION_LOST_THRESHOLD = 3;
const POLL_INTERVAL = 1000;

/** Raw downloads map as returned by `/state`. */
export type RawDownloads = Record<string, unknown[]>;

/** Disk-capacity information keyed by node id. */
export type NodeDiskInfo = Record<string, { total: { inBytes: number }; available: { inBytes: number } }>;

/** Raw shard-assignment payload returned by the Skulk state API. */
export interface RawShardAssignments {
  modelId?: string;
  nodeToRunner?: Record<string, string>;
  runnerToShard?: Record<string, Record<string, unknown>>;
}

/** Raw inner instance shape returned by the Skulk state API. */
export interface RawInstanceInner {
  instanceId?: string;
  shardAssignments?: RawShardAssignments;
}

/** Raw instances keyed by instance id with backend-specific tagged variants. */
export type RawInstances = Record<string, { MlxRingInstance?: RawInstanceInner; MlxJacclInstance?: RawInstanceInner }>;

/** Runner status is a tagged union: e.g. { "RunnerReady": {} } or { "RunnerLoading": { layersLoaded: 5, totalLayers: 32 } } */
export type RawRunners = Record<string, Record<string, unknown>>;

/** Normalized cluster-state data exposed to the dashboard UI. */
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

/** Poll the Skulk `/state` endpoint and normalize it into dashboard-friendly topology and status data. */
export function useClusterState(): ClusterState {
  const [topology, setTopology] = useState<TopologyData | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);
  const [downloads, setDownloads] = useState<RawDownloads>({});
  const [nodeDisk, setNodeDisk] = useState<NodeDiskInfo>({});
  const [instances, setInstances] = useState<RawInstances>({});
  const [runners, setRunners] = useState<RawRunners>({});
  const [nodeThunderbolt, setNodeThunderbolt] = useState<Record<string, RawThunderboltInfo>>({});
  const [nodeThunderboltBridge, setNodeThunderboltBridge] = useState<Record<string, RawThunderboltBridge>>({});
  const [nodeRdmaCtl, setNodeRdmaCtl] = useState<Record<string, RawRdmaCtl>>({});
  const [thunderboltBridgeCycles, setThunderboltBridgeCycles] = useState<string[][]>([]);
  const failuresRef = useRef(0);

  const fetchState = useCallback(async () => {
    try {
      const res = await fetch('/state');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: RawStateResponse = await res.json();

      if (data.topology) {
        const topo = transformTopology(
          data.topology,
          data.nodeIdentities ?? {},
          data.nodeMemory ?? {},
          data.nodeSystem ?? {},
          data.nodeNetwork ?? {},
          data.nodeThunderboltBridge ?? {},
          data.nodeRdmaCtl ?? {},
        );
        setTopology(topo);
      }

      setDownloads(data.downloads ?? {});
      setNodeDisk(data.nodeDisk ?? {});
      setInstances((data.instances ?? {}) as RawInstances);
      setRunners((data.runners ?? {}) as RawRunners);
      setNodeThunderbolt(data.nodeThunderbolt ?? {});
      setNodeThunderboltBridge(data.nodeThunderboltBridge ?? {});
      setNodeRdmaCtl(data.nodeRdmaCtl ?? {});
      setThunderboltBridgeCycles(data.thunderboltBridgeCycles ?? []);
      setLastUpdate(Date.now());
      failuresRef.current = 0;
      setConnected(true);
    } catch {
      failuresRef.current++;
      if (failuresRef.current >= CONNECTION_LOST_THRESHOLD) {
        setConnected(false);
      }
    }
  }, []);

  useEffect(() => {
    fetchState();
    const id = setInterval(fetchState, POLL_INTERVAL);
    return () => clearInterval(id);
  }, [fetchState]);

  return {
    topology,
    connected,
    lastUpdate,
    downloads,
    nodeDisk,
    instances,
    runners,
    nodeThunderbolt,
    nodeThunderboltBridge,
    nodeRdmaCtl,
    thunderboltBridgeCycles,
  };
}
