/** Memory sample returned by the topology polling layer. */
export interface MactopMemory {
  ram_usage: number;
  ram_total: number;
}

/** Temperature sample returned by the topology polling layer. */
export interface MactopTemp {
  gpu_temp_avg: number;
}

/** Node monitoring snapshot consumed by dashboard topology components. */
export interface MactopInfo {
  memory?: MactopMemory;
  temp?: MactopTemp;
  gpu_usage?: [number, number];
  sys_power?: number;
}

/** Basic hardware identity information for a node. */
export interface SystemInfo {
  model_id?: string;
  chip?: string;
  memory?: number;
}

/** One network interface known for a node. */
export interface NetworkInterfaceInfo {
  name?: string;
  addresses?: string[];
}

/** Severity of a node's derived health (#388). */
export type NodeHealthLevel = 'ok' | 'warn' | 'error';

/** One concrete node problem and the operator's path to fix it (#388). */
export interface NodeHealthReason {
  code: string;
  message: string;
  remediation: string;
}

/** Aggregate derived health for a node, rendered as a topology indicator. */
export interface NodeHealth {
  level: NodeHealthLevel;
  reasons: NodeHealthReason[];
}

/** Normalized node record used by the topology graph and cards. */
export interface NodeInfo {
  system_info?: SystemInfo;
  network_interfaces?: NetworkInterfaceInfo[];
  ip_to_interface?: Record<string, string>;
  mactop_info?: MactopInfo;
  last_mactop_update: number;
  friendly_name?: string;
  os_version?: string;
  os_build_version?: string;
  skulk_version?: string;
  skulk_commit?: string;
  thunderbolt_bridge?: boolean;
  rdma_enabled?: boolean;
  rdma_interfaces_present?: boolean;
  syncing?: boolean;
  /** Derived health summary for this node (#388); absent when unknown. */
  node_health?: NodeHealth;
}

/** Directed edge between two nodes in the cluster topology graph. */
export interface TopologyEdge {
  source: string;
  target: string;
  sendBackIp?: string;
  sendBackInterface?: string;
  sourceRdmaIface?: string;
  sinkRdmaIface?: string;
}

/** Complete normalized topology graph returned by the dashboard data layer. */
export interface TopologyData {
  nodes: Record<string, NodeInfo>;
  edges: TopologyEdge[];
}

export type DeviceModel = 'macbook-pro' | 'mac-studio' | 'mac-mini' | 'unknown';

/** Best-effort device-family classifier used for dashboard hardware icons. */
export function detectDeviceModel(modelId?: string): DeviceModel {
  if (!modelId) return 'unknown';
  const lower = modelId.toLowerCase();
  if (lower === 'macbook pro' || lower.includes('macbook')) return 'macbook-pro';
  if (lower === 'mac studio') return 'mac-studio';
  if (lower === 'mac mini') return 'mac-mini';
  return 'unknown';
}
