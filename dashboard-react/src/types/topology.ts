/** Memory sample returned by the topology polling layer. */
export interface MacmonMemory {
  ram_usage: number;
  ram_total: number;
}

/** Temperature sample returned by the topology polling layer. */
export interface MacmonTemp {
  gpu_temp_avg: number;
}

/** Node monitoring snapshot consumed by dashboard topology components. */
export interface MacmonInfo {
  memory?: MacmonMemory;
  temp?: MacmonTemp;
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

/** Normalized node record used by the topology graph and cards. */
export interface NodeInfo {
  system_info?: SystemInfo;
  network_interfaces?: NetworkInterfaceInfo[];
  ip_to_interface?: Record<string, string>;
  macmon_info?: MacmonInfo;
  last_macmon_update: number;
  friendly_name?: string;
  os_version?: string;
  os_build_version?: string;
  exo_version?: string;
  exo_commit?: string;
  thunderbolt_bridge?: boolean;
  rdma_enabled?: boolean;
  rdma_interfaces_present?: boolean;
  syncing?: boolean;
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
