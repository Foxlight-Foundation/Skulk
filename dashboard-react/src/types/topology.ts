export interface MacmonMemory {
  ram_usage: number;
  ram_total: number;
}

export interface MacmonTemp {
  gpu_temp_avg: number;
}

export interface MacmonInfo {
  memory?: MacmonMemory;
  temp?: MacmonTemp;
  gpu_usage?: [number, number];
  sys_power?: number;
}

export interface SystemInfo {
  model_id?: string;
  chip?: string;
  memory?: number;
}

export interface NetworkInterfaceInfo {
  name?: string;
  addresses?: string[];
}

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
}

export interface TopologyEdge {
  source: string;
  target: string;
  sendBackIp?: string;
  sendBackInterface?: string;
  sourceRdmaIface?: string;
  sinkRdmaIface?: string;
}

export interface TopologyData {
  nodes: Record<string, NodeInfo>;
  edges: TopologyEdge[];
}

export type DeviceModel = 'macbook-pro' | 'mac-studio' | 'mac-mini' | 'unknown';

export function detectDeviceModel(modelId?: string): DeviceModel {
  if (!modelId) return 'unknown';
  const lower = modelId.toLowerCase();
  if (lower === 'macbook pro' || lower.includes('macbook')) return 'macbook-pro';
  if (lower === 'mac studio') return 'mac-studio';
  if (lower === 'mac mini') return 'mac-mini';
  return 'unknown';
}
