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
