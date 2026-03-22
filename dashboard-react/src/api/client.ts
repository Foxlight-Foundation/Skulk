/**
 * API Client
 *
 * Typed fetch wrappers for all exo backend endpoints.
 * All requests are relative so they work with both the Vite dev proxy
 * and when served statically from the exo server.
 */

import type {
  StateResponse,
  ModelStoreConfig,
  PlacementPreview,
  TraceEntry,
  TraceListResponse,
  TraceStatsResponse,
  HuggingFaceModel,
  ConfigResponse,
  ConfigUpdateResponse,
  StoreHealthResponse,
  StoreRegistryEntry,
  StoreDownloadProgress,
  BrowseResponse,
  NodeIdentityResponse,
} from './types';

// ─── Base helpers ──────────────────────────────────────────────────────────────

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path, {
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) {
    throw new Error(`GET ${path} failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`POST ${path} failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

async function del(path: string): Promise<void> {
  const response = await fetch(path, { method: 'DELETE' });
  if (!response.ok) {
    throw new Error(`DELETE ${path} failed: ${response.status} ${response.statusText}`);
  }
}

async function put<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({})) as { detail?: string };
    throw new Error(detail.detail ?? `PUT ${path} failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

// ─── State endpoint ────────────────────────────────────────────────────────────

export async function fetchState(): Promise<StateResponse> {
  return get<StateResponse>('/state');
}

// ─── Model downloads ───────────────────────────────────────────────────────────

export async function startDownload(modelId: string): Promise<void> {
  await post('/download/start', { model_id: modelId });
}

export async function cancelDownload(modelId: string): Promise<void> {
  await post('/download/cancel', { model_id: modelId });
}

// ─── Placement previews ────────────────────────────────────────────────────────

export async function fetchPlacementPreview(
  modelId: string,
): Promise<PlacementPreview> {
  return get<PlacementPreview>(`/models/${encodeURIComponent(modelId)}/placement`);
}

// ─── Config / Store ────────────────────────────────────────────────────────────

export async function fetchConfig(): Promise<ConfigResponse> {
  return get<ConfigResponse>('/config');
}

export async function updateConfig(
  config: Record<string, unknown>,
): Promise<ConfigUpdateResponse> {
  return put<ConfigUpdateResponse>('/config', { config });
}

/** @deprecated Use fetchConfig / updateConfig instead */
export async function saveConfig(config: ModelStoreConfig): Promise<void> {
  await post('/config', config);
}

export async function fetchStore(): Promise<ModelStoreConfig> {
  return get<ModelStoreConfig>('/store');
}

export async function addModelStore(path: string): Promise<void> {
  await post('/store/add', { path });
}

export async function removeModelStore(path: string): Promise<void> {
  await post('/store/remove', { path });
}

export async function fetchStoreHealth(): Promise<StoreHealthResponse | null> {
  try {
    const response = await fetch('/store/health');
    if (!response.ok) return null;
    return response.json() as Promise<StoreHealthResponse>;
  } catch {
    return null;
  }
}

export async function fetchStoreRegistry(): Promise<StoreRegistryEntry[]> {
  try {
    const response = await fetch('/store/registry');
    if (!response.ok) return [];
    const data = await response.json() as { entries?: StoreRegistryEntry[] };
    return data.entries ?? [];
  } catch {
    return [];
  }
}

export async function fetchStoreDownloads(): Promise<StoreDownloadProgress[]> {
  try {
    const response = await fetch('/store/downloads');
    if (!response.ok) return [];
    const data = await response.json() as { downloads?: StoreDownloadProgress[] };
    return data.downloads ?? [];
  } catch {
    return [];
  }
}

export async function requestStoreDownload(
  modelId: string,
): Promise<{ status: string; progress?: number; error?: string }> {
  const response = await fetch(`/store/models/${encodeURIComponent(modelId)}/download`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw new Error(`Store download request failed: ${response.status}`);
  }
  return response.json() as Promise<{ status: string; progress?: number; error?: string }>;
}

export async function deleteStoreModel(modelId: string): Promise<boolean> {
  const response = await fetch(`/store/models/${encodeURIComponent(modelId)}`, {
    method: 'DELETE',
  });
  return response.ok;
}

// ─── Traces ────────────────────────────────────────────────────────────────────

export async function fetchTraces(): Promise<TraceEntry[]> {
  return get<TraceEntry[]>('/v1/traces');
}

export async function listTraces(): Promise<TraceListResponse> {
  return get<TraceListResponse>('/v1/traces');
}

export async function deleteTrace(taskId: string): Promise<void> {
  await del(`/v1/traces/${encodeURIComponent(taskId)}`);
}

export async function deleteAllTraces(): Promise<void> {
  await del('/v1/traces');
}

export async function deleteTracesBatch(
  taskIds: string[],
): Promise<{ deleted: string[]; notFound: string[] }> {
  const response = await fetch('/v1/traces/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ taskIds }),
  });
  if (!response.ok) {
    throw new Error(`Failed to delete traces: ${response.status}`);
  }
  return response.json() as Promise<{ deleted: string[]; notFound: string[] }>;
}

export async function fetchTraceData(taskId: string): Promise<unknown> {
  return get<unknown>(`/v1/traces/${encodeURIComponent(taskId)}`);
}

export async function fetchTraceStats(taskId: string): Promise<TraceStatsResponse> {
  return get<TraceStatsResponse>(`/v1/traces/${encodeURIComponent(taskId)}/stats`);
}

export function getTraceRawUrl(taskId: string): string {
  return `/v1/traces/${encodeURIComponent(taskId)}/raw`;
}

// ─── Node identity ─────────────────────────────────────────────────────────────

export async function fetchNodeInfo(): Promise<unknown> {
  return get<unknown>('/node');
}

export async function fetchNodeIdentity(): Promise<NodeIdentityResponse> {
  return get<NodeIdentityResponse>('/node/identity');
}

// ─── Filesystem browser ────────────────────────────────────────────────────────

export async function browseFilesystem(path = '/Volumes'): Promise<BrowseResponse> {
  const response = await fetch(`/filesystem/browse?path=${encodeURIComponent(path)}`);
  if (!response.ok) throw new Error(`Browse failed: ${response.status}`);
  return response.json() as Promise<BrowseResponse>;
}

// ─── Download management ───────────────────────────────────────────────────────

export async function startDownloadForNode(
  nodeId: string,
  shardMetadata: object,
): Promise<void> {
  await post('/download/start', { targetNodeId: nodeId, shardMetadata });
}

export async function deleteDownload(nodeId: string, modelId: string): Promise<void> {
  await del(`/download/${encodeURIComponent(nodeId)}/${encodeURIComponent(modelId)}`);
}

// ─── HuggingFace search (proxied through exo backend) ─────────────────────────

export async function searchHuggingFace(query: string): Promise<HuggingFaceModel[]> {
  return get<HuggingFaceModel[]>(
    `/models/search?q=${encodeURIComponent(query)}`,
  );
}
