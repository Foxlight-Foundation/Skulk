/** Types for read-only Skulk node diagnostics API responses. */

export interface MemoryValue {
  /** Memory quantity in raw bytes. */
  inBytes: number;
}

export interface DiagnosticsRuntime {
  nodeId: string;
  hostname: string;
  friendlyName?: string | null;
  isMaster: boolean;
  masterNodeId?: string | null;
  cwd: string;
  configPath: string;
  configFileExists: boolean;
  skulkVersion: string;
  skulkCommit: string;
  libp2pNamespace?: string | null;
  pythonUnbuffered: boolean;
  tracingEnabled: boolean;
  structuredLoggingConfigured: boolean;
  loggingIngestUrl?: string | null;
}

export interface DiagnosticsProcess {
  pid: number;
  parentPid?: number | null;
  role: 'skulk' | 'runner' | 'vector' | 'python' | 'other';
  command: string;
  status?: string | null;
  cpuPercent?: number | null;
  memoryPercent?: number | null;
  rss?: MemoryValue | null;
  elapsedSeconds?: number | null;
  isChildOfSkulk: boolean;
}

export interface RunnerTaskDiagnostics {
  taskId: string;
  taskKind: string;
  taskStatus: string;
  instanceId: string;
  commandId?: string | null;
  runnerId?: string | null;
  modelId?: string | null;
}

export interface RunnerLifecycleMilestone {
  at: string;
  name: string;
  detail?: string | null;
}

/** Runner phase names reported by the local flight recorder. */
export type RunnerPhaseName =
  | 'created'
  | 'idle'
  | 'connect_group'
  | 'load_model'
  | 'warmup'
  | 'task_submission'
  | 'task_agreement'
  | 'prompt_build'
  | 'vision_preprocess'
  | 'kv_cache_lookup'
  | 'prefill_barrier'
  | 'prefill_pipeline'
  | 'prefill_stream'
  | 'decode_barrier'
  | 'decode_wait_first_token'
  | 'decode_stream'
  | 'parser'
  | 'cancel_requested'
  | 'cancel_observed'
  | 'completion'
  | 'error'
  | 'shutdown_cleanup';

/** Best-effort memory counters reported by MLX/Metal inside a runner process. */
export interface MlxMemorySnapshot {
  generatedAt: string;
  active?: MemoryValue | null;
  cache?: MemoryValue | null;
  peak?: MemoryValue | null;
  wiredLimit?: MemoryValue | null;
  source: string;
}

/** JSON-safe value type used for runner diagnostic attributes. */
export type RunnerDiagnosticValue = string | number | boolean | string[];

/** Stable runner identity attached to every flight-recorder entry. */
export interface RunnerDiagnosticContext {
  nodeId: string;
  runnerId: string;
  pid?: number | null;
  instanceId: string;
  modelId: string;
  rank: number;
  worldSize: number;
  startLayer: number;
  endLayer: number;
  nLayers: number;
}

/** One bounded local-only flight-recorder event emitted by a runner process. */
export interface RunnerFlightRecorderEntry {
  at: string;
  phase: RunnerPhaseName;
  event: string;
  detail?: string | null;
  attrs: Record<string, RunnerDiagnosticValue>;
  context: RunnerDiagnosticContext;
  taskId?: string | null;
  commandId?: string | null;
  mlxMemory?: MlxMemorySnapshot | null;
}

/** Live supervisor diagnostics for one local runner process. */
export interface RunnerSupervisorDiagnostics {
  runnerId: string;
  instanceId: string;
  nodeId: string;
  modelId: string;
  deviceRank: number;
  worldSize: number;
  startLayer: number;
  endLayer: number;
  nLayers: number;
  pid?: number | null;
  processAlive: boolean;
  exitCode?: number | null;
  statusKind: string;
  statusSince: string;
  secondsInStatus: number;
  phase: RunnerPhaseName;
  phaseStartedAt: string;
  secondsInPhase: number;
  lastProgressAt?: string | null;
  activeTaskId?: string | null;
  activeCommandId?: string | null;
  phaseDetail?: string | null;
  lastMlxMemory?: MlxMemorySnapshot | null;
  flightRecorder: RunnerFlightRecorderEntry[];
  pendingTaskIds: string[];
  inProgressTasks: RunnerTaskDiagnostics[];
  completedTaskCount: number;
  cancelledTaskIds: string[];
  lastTaskSentAt?: string | null;
  lastEventReceivedAt?: string | null;
  lastEventType?: string | null;
  milestones: RunnerLifecycleMilestone[];
}

export interface PlacementRunnerDiagnostics {
  runnerId: string;
  nodeId: string;
  friendlyName?: string | null;
  statusKind?: string | null;
  deviceRank: number;
  worldSize: number;
  startLayer: number;
  endLayer: number;
  nLayers: number;
  isLocal: boolean;
  isMaster: boolean;
  tasks: RunnerTaskDiagnostics[];
}

export interface InstancePlacementDiagnostics {
  instanceId: string;
  modelId: string;
  masterNodeId?: string | null;
  masterIsPlacementNode: boolean;
  localNodeIsPlacementNode: boolean;
  placementNodeIds: string[];
  runners: PlacementRunnerDiagnostics[];
  warnings: string[];
}

export interface NodeResourceDiagnostics {
  gatheredMemory?: {
    ramTotal: MemoryValue;
    ramAvailable: MemoryValue;
    swapTotal: MemoryValue;
    swapAvailable: MemoryValue;
  } | null;
  currentMemory?: {
    ramTotal: MemoryValue;
    ramAvailable: MemoryValue;
    swapTotal: MemoryValue;
    swapAvailable: MemoryValue;
  } | null;
  system?: {
    gpuUsage?: number;
    temp?: number;
    sysPower?: number;
    pcpuUsage?: number;
    ecpuUsage?: number;
  } | null;
}

export interface NodeDiagnostics {
  generatedAt: string;
  runtime: DiagnosticsRuntime;
  resources: NodeResourceDiagnostics;
  processes: DiagnosticsProcess[];
  supervisorRunners: RunnerSupervisorDiagnostics[];
  placements: InstancePlacementDiagnostics[];
  warnings: string[];
}

/** Heavyweight process sampling result inside an on-demand capture bundle. */
export interface DiagnosticProcessSample {
  name: string;
  command: string[];
  ok: boolean;
  exitCode?: number | null;
  durationSeconds: number;
  stdout?: string | null;
  stderr?: string | null;
  error?: string | null;
}

/** On-demand local or proxied diagnostic capture bundle. */
export interface DiagnosticCaptureResponse {
  generatedAt: string;
  nodeId: string;
  nodeDiagnostics: NodeDiagnostics;
  runner?: RunnerSupervisorDiagnostics | null;
  flightRecorder: RunnerFlightRecorderEntry[];
  mlxMemory?: MlxMemorySnapshot | null;
  processSamples: DiagnosticProcessSample[];
  warnings: string[];
}

/** Per-runner synopsis emitted as part of the cross-rank cluster timeline. */
export interface ClusterTimelineRunner {
  nodeId: string;
  runnerId: string;
  instanceId: string;
  modelId: string;
  deviceRank: number;
  worldSize: number;
  pid?: number | null;
  processAlive: boolean;
  statusKind: string;
  phase: RunnerPhaseName;
  phaseDetail?: string | null;
  secondsInPhase: number;
  lastProgressAt?: string | null;
  activeTaskId?: string | null;
  activeCommandId?: string | null;
  lastMlxMemory?: MlxMemorySnapshot | null;
}

/**
 * Single flight-recorder entry annotated with cluster identity. The cross-rank
 * merged timeline is a list of these sorted by `at` ascending so a deadlock's
 * rank-disagreement signature reads top-to-bottom.
 */
export interface ClusterTimelineEntry {
  at: string;
  nodeId: string;
  runnerId: string;
  deviceRank: number;
  worldSize: number;
  phase: RunnerPhaseName;
  event: string;
  detail?: string | null;
  attrs: Record<string, RunnerDiagnosticValue>;
  taskId?: string | null;
  commandId?: string | null;
  mlxMemory?: MlxMemorySnapshot | null;
}

/** One peer that the cluster timeline could not reach. */
export interface ClusterTimelineUnreachable {
  nodeId: string;
  url?: string | null;
  error: string;
}

/** Cross-rank chronological view of runner activity, served by the cluster API. */
export interface ClusterTimeline {
  generatedAt: string;
  localNodeId: string;
  masterNodeId?: string | null;
  runners: ClusterTimelineRunner[];
  timeline: ClusterTimelineEntry[];
  unreachableNodes: ClusterTimelineUnreachable[];
}
