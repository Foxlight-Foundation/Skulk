/**
 * Topology Store
 *
 * Polls /state every 2 seconds and maintains:
 * - Topology graph (nodes + edges)
 * - Instance and runner data
 * - Node identities (chip, OS version, friendly name)
 * - Memory usage, performance profiles
 * - Download progress per node/model
 * - Thunderbolt / RDMA connectivity info
 * - Connection status
 *
 * Replaces the topology/state slices of app.svelte.ts.
 */
import { create } from 'zustand';
import { fetchState } from '../api/client';
import type {
  TopologyData,
  Instance,
  RawNodeIdentity,
  RawMemoryUsage,
  RawSystemPerformanceProfile,
  RawNetworkConnectivity,
  RawNodeDownloads,
  RawThunderboltInfo,
  RawRdmaCtlInfo,
  NodeDiskInfo,
} from '../api/types';

const POLL_INTERVAL_MS = 2000;

// ─── Derived / computed helpers ────────────────────────────────────────────────

export interface MacosVersionMismatch {
  nodeId: string;
  friendlyName: string;
  version: string;
  buildVersion: string;
}

// ─── Store shape ───────────────────────────────────────────────────────────────

interface TopologyState {
  // Connection
  isConnected: boolean;
  lastUpdate: number | null;

  // Raw topology
  topology: TopologyData | null;
  instances: Record<string, Instance>;
  runners: Record<string, unknown>;

  // Granular node state
  nodeIdentities: Record<string, RawNodeIdentity>;
  nodeMemoryUsages: Record<string, RawMemoryUsage>;
  nodePerformanceProfiles: Record<string, RawSystemPerformanceProfile>;
  nodeNetworkConnectivity: Record<string, RawNetworkConnectivity>;
  nodeDownloads: Record<string, RawNodeDownloads>;
  /** Raw tagged download entries per node (e.g. DownloadCompleted, DownloadOngoing) */
  downloads: Record<string, unknown[]>;
  nodeDisk: Record<string, NodeDiskInfo>;
  nodeThunderbolt: Record<string, RawThunderboltInfo>;
  nodeThunderboltBridge: Record<string, unknown>;
  nodeRdmaCtl: Record<string, RawRdmaCtlInfo>;
  thunderboltBridgeCycles: number | null;

  // Polling control
  _pollTimer: ReturnType<typeof setInterval> | null;
  startPolling: () => void;
  stopPolling: () => void;

  // Derived getters
  getNodeName: (nodeId: string) => string;
  getMacosVersionMismatch: () => MacosVersionMismatch[] | null;
  getHasTb5WithoutRdma: () => boolean;
  getActiveDownloadSummary: () => { count: number; percentage: number } | null;
}

export const useTopologyStore = create<TopologyState>((set, get) => ({
  isConnected: false,
  lastUpdate: null,
  topology: null,
  instances: {},
  runners: {},
  nodeIdentities: {},
  nodeMemoryUsages: {},
  nodePerformanceProfiles: {},
  nodeNetworkConnectivity: {},
  nodeDownloads: {},
  downloads: {},
  nodeDisk: {},
  nodeThunderbolt: {},
  nodeThunderboltBridge: {},
  nodeRdmaCtl: {},
  thunderboltBridgeCycles: null,
  _pollTimer: null,

  // ── Polling ──────────────────────────────────────────────────────────────────

  startPolling: () => {
    const { _pollTimer } = get();
    if (_pollTimer !== null) return; // Already polling

    const poll = async () => {
      try {
        const state = await fetchState();
        set({
          isConnected: true,
          lastUpdate: Date.now(),
          topology: state.topology ?? null,
          instances: state.instances ?? {},
          runners: state.runners ?? {},
          nodeIdentities: state.nodeIdentities ?? {},
          nodeMemoryUsages: state.nodeMemoryUsages ?? {},
          nodePerformanceProfiles: state.nodePerformanceProfiles ?? {},
          nodeNetworkConnectivity: state.nodeNetworkConnectivity ?? {},
          nodeDownloads: state.nodeDownloads ?? {},
          downloads: state.downloads ?? {},
          nodeDisk: state.nodeDisk ?? {},
          nodeThunderbolt: state.nodeThunderbolt ?? {},
          nodeThunderboltBridge: state.nodeThunderboltBridge ?? {},
          nodeRdmaCtl: state.nodeRdmaCtl ?? {},
          thunderboltBridgeCycles: state.thunderboltBridgeCycles ?? null,
        });
      } catch {
        set({ isConnected: false });
      }
    };

    void poll(); // Immediate first fetch
    const timer = setInterval(() => void poll(), POLL_INTERVAL_MS);
    set({ _pollTimer: timer });
  },

  stopPolling: () => {
    const { _pollTimer } = get();
    if (_pollTimer !== null) {
      clearInterval(_pollTimer);
      set({ _pollTimer: null });
    }
  },

  // ── Derived helpers ───────────────────────────────────────────────────────────

  getNodeName: (nodeId) => {
    const { nodeIdentities, topology } = get();
    return (
      nodeIdentities[nodeId]?.friendlyName ??
      topology?.nodes[nodeId]?.friendly_name ??
      nodeId.slice(0, 8)
    );
  },

  getMacosVersionMismatch: () => {
    const { nodeIdentities } = get();
    const entries = Object.entries(nodeIdentities);
    const macosNodes = entries.filter(([, id]) => {
      const v = id.osVersion;
      return v && v !== 'Unknown' && /^\d/.test(v);
    });
    if (macosNodes.length < 2) return null;
    const buildVersions = new Set(
      macosNodes.map(([, id]) => id.osBuildVersion ?? id.osVersion ?? ''),
    );
    if (buildVersions.size <= 1) return null;
    const { getNodeName } = get();
    return macosNodes.map(([nodeId, id]) => ({
      nodeId,
      friendlyName: getNodeName(nodeId),
      version: id.osVersion ?? 'Unknown',
      buildVersion: id.osBuildVersion ?? 'Unknown',
    }));
  },

  getHasTb5WithoutRdma: () => {
    const { nodeRdmaCtl, nodeThunderbolt } = get();
    // Find nodes with any TB interfaces
    const tb5NodeIds = Object.entries(nodeThunderbolt)
      .filter(([, tb]) => (tb.interfaces?.length ?? 0) > 0)
      .map(([nodeId]) => nodeId);
    if (tb5NodeIds.length === 0) return false;
    // Check if RDMA is disabled on any TB5 node
    return tb5NodeIds.some((nodeId) => !nodeRdmaCtl[nodeId]?.enabled);
  },

  getActiveDownloadSummary: () => {
    const { nodeDownloads } = get();
    let totalBytes = 0;
    let downloadedBytes = 0;
    let count = 0;

    for (const nodeProgress of Object.values(nodeDownloads)) {
      for (const progress of Object.values(nodeProgress)) {
        if (progress.status === 'downloading') {
          count++;
          totalBytes += progress.totalBytes ?? 0;
          downloadedBytes += progress.downloadedBytes ?? 0;
        }
      }
    }

    if (count === 0) return null;
    return {
      count,
      percentage: totalBytes > 0 ? (downloadedBytes / totalBytes) * 100 : 0,
    };
  },
}));
