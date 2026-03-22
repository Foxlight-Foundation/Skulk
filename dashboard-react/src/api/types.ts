/**
 * API Types
 *
 * TypeScript types for all data returned by the exo backend.
 * These mirror the server's Pydantic models and the existing
 * TypeScript interfaces in the Svelte app.svelte.ts.
 */

// ─── Topology ─────────────────────────────────────────────────────────────────

export interface NodeSystemInfo {
  model_id?: string;
  chip?: string;
  memory?: number;
}

export interface NodeNetworkInterface {
  name?: string;
  addresses?: string[];
}

export interface NodeMacmonInfo {
  memory?: {
    ram_usage: number;
    ram_total: number;
  };
  temp?: {
    gpu_temp_avg: number;
  };
  gpu_usage?: [number, number];
  sys_power?: number;
}

export interface NodeInfo {
  system_info?: NodeSystemInfo;
  network_interfaces?: NodeNetworkInterface[];
  ip_to_interface?: Record<string, string>;
  macmon_info?: NodeMacmonInfo;
  last_macmon_update: number;
  friendly_name?: string;
  os_version?: string;
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

// ─── Instances & Runners ──────────────────────────────────────────────────────

export interface ShardAssignments {
  modelId?: string;
  runnerToShard?: Record<string, unknown>;
  nodeToRunner?: Record<string, string>;
}

export interface Instance {
  shardAssignments?: ShardAssignments;
}

// ─── Granular Node State (new state structure) ────────────────────────────────

export interface RawNodeIdentity {
  modelId?: string;
  chipId?: string;
  friendlyName?: string;
  osVersion?: string;
  osBuildVersion?: string;
}

export interface RawMemoryUsage {
  ramTotal?: { inBytes: number };
  ramAvailable?: { inBytes: number };
  swapTotal?: { inBytes: number };
  swapAvailable?: { inBytes: number };
}

export interface RawSystemPerformanceProfile {
  gpuUsage?: number;
  temp?: number;
  sysPower?: number;
  pcpuUsage?: number;
  ecpuUsage?: number;
}

export interface RawNetworkConnectivity {
  connectedNodeIds?: string[];
}

export interface RawDownloadProgress {
  downloadedBytes?: number;
  totalBytes?: number;
  status?: 'pending' | 'downloading' | 'complete' | 'error';
  speed?: number;
}

export interface RawNodeDownloads {
  [modelId: string]: RawDownloadProgress;
}

export interface RawThunderboltInfo {
  interfaces?: string[];
  bridgeCycles?: number;
}

export interface RawRdmaCtlInfo {
  enabled?: boolean;
  interfaces?: string[];
}

export interface StateResponse {
  topology?: TopologyData;
  instances?: Record<string, Instance>;
  runners?: Record<string, unknown>;
  nodeIdentities?: Record<string, RawNodeIdentity>;
  nodeMemoryUsages?: Record<string, RawMemoryUsage>;
  nodePerformanceProfiles?: Record<string, RawSystemPerformanceProfile>;
  nodeNetworkConnectivity?: Record<string, RawNetworkConnectivity>;
  nodeDownloads?: Record<string, RawNodeDownloads>;
  nodeThunderbolt?: Record<string, RawThunderboltInfo>;
  nodeThunderboltBridge?: Record<string, unknown>;
  nodeRdmaCtl?: Record<string, RawRdmaCtlInfo>;
  thunderboltBridgeCycles?: number;
}

// ─── Download Progress (aggregated) ──────────────────────────────────────────

export interface DownloadProgress {
  modelId: string;
  nodeId: string;
  downloadedBytes: number;
  totalBytes: number;
  status: 'pending' | 'downloading' | 'complete' | 'error';
  speed?: number;
}

// ─── Placement Preview ────────────────────────────────────────────────────────

export interface PlacementPreview {
  modelId: string;
  nodeAssignments: Record<string, { shardStart: number; shardEnd: number }>;
  totalLayers: number;
  estimatedVram?: number;
}

// ─── Chat / OpenAI-compatible ─────────────────────────────────────────────────

export type MessageRole = 'user' | 'assistant' | 'system';

export interface ImageContent {
  type: 'image_url';
  image_url: { url: string };
}

export interface TextContent {
  type: 'text';
  text: string;
}

export type MessageContent = string | Array<TextContent | ImageContent>;

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: MessageContent;
  createdAt: number;
  /** Token uncertainty heatmap data, if available */
  tokenHeatmap?: Array<{ token: string; logprob: number }>;
  /** Whether this message is currently being streamed */
  isStreaming?: boolean;
  /** Time-to-first-token in ms */
  ttft?: number;
  /** Alias for ttft used in persisted data */
  ttftMs?: number;
  /** Tokens per second */
  tps?: number;
  /** Whether prefill is in progress */
  isPrefilling?: boolean;
  /** Image URL if this is an image generation result */
  imageUrl?: string;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
  modelId?: string;
  /** Sharding strategy label, e.g. "Pipeline" */
  sharding?: string;
  /** Instance type label, e.g. "MLX Ring" */
  instanceType?: string;
  /** Whether thinking/reasoning mode is enabled for this conversation */
  thinkingEnabled?: boolean;
}

export interface ChatCompletionRequest {
  model: string;
  messages: Array<{ role: MessageRole; content: MessageContent }>;
  stream: true;
  temperature?: number;
  max_tokens?: number;
}

export interface ImageGenerationRequest {
  model: string;
  prompt: string;
  n?: number;
  size?: string;
}

export interface ImageEditRequest {
  model: string;
  image: string;
  prompt: string;
  n?: number;
  size?: string;
}

// ─── Config / Store ───────────────────────────────────────────────────────────

export interface ModelStoreConfig {
  storeDir?: string;
  registeredStores?: string[];
}

// ─── HuggingFace search ───────────────────────────────────────────────────────

export interface HuggingFaceModel {
  id: string;
  modelId: string;
  downloads?: number;
  likes?: number;
  tags?: string[];
  pipeline_tag?: string;
}

// ─── Traces ───────────────────────────────────────────────────────────────────

export interface TraceEntry {
  taskId: string;
  timestamp: number;
  model?: string;
  durationMs?: number;
  tokens?: number;
}
