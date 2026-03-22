/**
 * Shared utilities for parsing and querying download state.
 *
 * The download state from `/state` is shaped as:
 *   Record<NodeId, Array<TaggedDownloadEntry>>
 *
 * Each entry is a tagged union object like:
 *   { "DownloadCompleted": { shard_metadata: { "PipelineShardMetadata": { model_card: { model_id: "..." }, ... } }, ... } }
 *
 * Ported from dashboard/src/lib/utils/downloads.ts
 */

/** Unwrap one level of tagged-union envelope, returning [tag, payload]. */
function unwrapTagged(
  obj: Record<string, unknown>,
): [string, Record<string, unknown>] | null {
  const keys = Object.keys(obj);
  if (keys.length !== 1) return null;
  const tag = keys[0];
  if (!tag) return null;
  const payload = obj[tag];
  if (!payload || typeof payload !== 'object') return null;
  return [tag, payload as Record<string, unknown>];
}

/** Extract the model ID string from a download entry's nested shard_metadata. */
export function extractModelIdFromDownload(
  downloadPayload: Record<string, unknown>,
): string | null {
  const shardMetadata =
    downloadPayload.shard_metadata ?? downloadPayload.shardMetadata;
  if (!shardMetadata || typeof shardMetadata !== 'object') return null;

  const unwrapped = unwrapTagged(shardMetadata as Record<string, unknown>);
  if (!unwrapped) return null;
  const [, shardData] = unwrapped;

  const modelMeta = shardData.model_card ?? shardData.modelCard;
  if (!modelMeta || typeof modelMeta !== 'object') return null;

  const meta = modelMeta as Record<string, unknown>;
  return (meta.model_id as string) ?? (meta.modelId as string) ?? null;
}

/** Extract the shard_metadata object from a download entry payload. */
export function extractShardMetadata(
  downloadPayload: Record<string, unknown>,
): Record<string, unknown> | null {
  const shardMetadata =
    downloadPayload.shard_metadata ?? downloadPayload.shardMetadata;
  if (!shardMetadata || typeof shardMetadata !== 'object') return null;
  return shardMetadata as Record<string, unknown>;
}

/** Get the download tag (DownloadCompleted, DownloadOngoing, etc.) from a wrapped entry. */
export function getDownloadTag(
  entry: unknown,
): [string, Record<string, unknown>] | null {
  if (!entry || typeof entry !== 'object') return null;
  return unwrapTagged(entry as Record<string, unknown>);
}

/**
 * Iterate over all download entries for a given node, yielding [tag, payload, modelId].
 */
function* iterNodeDownloads(
  nodeDownloads: unknown[],
): Generator<[string, Record<string, unknown>, string]> {
  for (const entry of nodeDownloads) {
    const tagged = getDownloadTag(entry);
    if (!tagged) continue;
    const [tag, payload] = tagged;
    const modelId = extractModelIdFromDownload(payload);
    if (!modelId) continue;
    yield [tag, payload, modelId];
  }
}

/** Check if a specific model is fully downloaded (DownloadCompleted) on a specific node. */
export function isModelDownloadedOnNode(
  downloadsData: Record<string, unknown[]>,
  nodeId: string,
  modelId: string,
): boolean {
  const nodeDownloads = downloadsData[nodeId];
  if (!Array.isArray(nodeDownloads)) return false;

  for (const [tag, , entryModelId] of iterNodeDownloads(nodeDownloads)) {
    if (tag === 'DownloadCompleted' && entryModelId === modelId) return true;
  }
  return false;
}

/** Get all node IDs where a model is fully downloaded (DownloadCompleted). */
export function getNodesWithModelDownloaded(
  downloadsData: Record<string, unknown[]>,
  modelId: string,
): string[] {
  const result: string[] = [];
  for (const nodeId of Object.keys(downloadsData)) {
    if (isModelDownloadedOnNode(downloadsData, nodeId, modelId)) {
      result.push(nodeId);
    }
  }
  return result;
}

/**
 * Find shard metadata for a model from any download entry across all nodes.
 */
export function getShardMetadataForModel(
  downloadsData: Record<string, unknown[]>,
  modelId: string,
): Record<string, unknown> | null {
  let fallback: Record<string, unknown> | null = null;

  for (const nodeDownloads of Object.values(downloadsData)) {
    if (!Array.isArray(nodeDownloads)) continue;

    for (const [tag, payload, entryModelId] of iterNodeDownloads(nodeDownloads)) {
      if (entryModelId !== modelId) continue;
      const shard = extractShardMetadata(payload);
      if (!shard) continue;

      if (tag === 'DownloadCompleted') return shard;
      if (!fallback) fallback = shard;
    }
  }
  return fallback;
}

export interface ModelCardInfo {
  family: string;
  quantization: string;
  baseModel: string;
  capabilities: string[];
  storageSize: number;
  nLayers: number;
  supportsTensor: boolean;
}

function getBytes(value: unknown): number {
  if (typeof value === 'number') return value;
  if (value && typeof value === 'object') {
    const v = value as Record<string, unknown>;
    if (typeof v.inBytes === 'number') return v.inBytes;
  }
  return 0;
}

export function extractModelCard(payload: Record<string, unknown>): {
  prettyName: string | null;
  card: ModelCardInfo | null;
} {
  const shardMetadata = payload.shard_metadata ?? payload.shardMetadata;
  if (!shardMetadata || typeof shardMetadata !== 'object')
    return { prettyName: null, card: null };
  const shardObj = shardMetadata as Record<string, unknown>;
  const shardKeys = Object.keys(shardObj);
  if (shardKeys.length !== 1) return { prettyName: null, card: null };
  const firstKey = shardKeys[0];
  if (!firstKey) return { prettyName: null, card: null };
  const shardData = shardObj[firstKey] as Record<string, unknown>;
  const modelMeta = shardData?.model_card ?? shardData?.modelCard;
  if (!modelMeta || typeof modelMeta !== 'object')
    return { prettyName: null, card: null };
  const meta = modelMeta as Record<string, unknown>;

  const prettyName = (meta.prettyName as string) ?? null;

  const card: ModelCardInfo = {
    family: (meta.family as string) ?? '',
    quantization: (meta.quantization as string) ?? '',
    baseModel: (meta.base_model as string) ?? (meta.baseModel as string) ?? '',
    capabilities: Array.isArray(meta.capabilities)
      ? (meta.capabilities as string[])
      : [],
    storageSize: getBytes(meta.storage_size ?? meta.storageSize),
    nLayers: (meta.n_layers as number) ?? (meta.nLayers as number) ?? 0,
    supportsTensor:
      (meta.supports_tensor as boolean) ??
      (meta.supportsTensor as boolean) ??
      false,
  };

  return { prettyName, card };
}

export type CellStatus =
  | { kind: 'completed'; totalBytes: number; modelDirectory?: string }
  | {
      kind: 'downloading';
      percentage: number;
      downloadedBytes: number;
      totalBytes: number;
      speed: number;
      etaMs: number;
      modelDirectory?: string;
    }
  | {
      kind: 'pending';
      downloaded: number;
      total: number;
      modelDirectory?: string;
    }
  | { kind: 'failed'; modelDirectory?: string }
  | { kind: 'not_present' };

const CELL_PRIORITY: Record<CellStatus['kind'], number> = {
  completed: 4,
  downloading: 3,
  pending: 2,
  failed: 1,
  not_present: 0,
};

export function shouldUpgradeCell(
  existing: CellStatus,
  candidate: CellStatus,
): boolean {
  return CELL_PRIORITY[candidate.kind] > CELL_PRIORITY[existing.kind];
}

export interface ModelRow {
  modelId: string;
  prettyName: string | null;
  cells: Record<string, CellStatus>;
  shardMetadata: Record<string, unknown> | null;
  modelCard: ModelCardInfo | null;
}

export interface NodeColumn {
  nodeId: string;
  label: string;
  diskAvailable?: number;
  diskTotal?: number;
}

/**
 * Build model rows and node columns from the raw downloads state.
 */
export function buildDownloadGrid(
  downloadsData: Record<string, unknown[]>,
  nodeDiskData: Record<string, { total: { inBytes: number }; available: { inBytes: number } }>,
  getNodeLabel: (nodeId: string) => string,
): { modelRows: ModelRow[]; nodeColumns: NodeColumn[] } {
  if (!downloadsData || Object.keys(downloadsData).length === 0) {
    return { modelRows: [], nodeColumns: [] };
  }

  const allNodeIds = Object.keys(downloadsData);
  const nodeColumns: NodeColumn[] = allNodeIds.map((nodeId) => {
    const diskInfo = nodeDiskData?.[nodeId];
    return {
      nodeId,
      label: getNodeLabel(nodeId),
      diskAvailable: diskInfo?.available?.inBytes,
      diskTotal: diskInfo?.total?.inBytes,
    };
  });

  const rowMap = new Map<string, ModelRow>();

  for (const [nodeId, nodeDownloads] of Object.entries(downloadsData)) {
    const entries = Array.isArray(nodeDownloads)
      ? nodeDownloads
      : nodeDownloads && typeof nodeDownloads === 'object'
        ? Object.values(nodeDownloads as Record<string, unknown>)
        : [];

    for (const entry of entries) {
      const tagged = getDownloadTag(entry);
      if (!tagged) continue;
      const [tag, payload] = tagged;

      const modelId = extractModelIdFromDownload(payload) ?? 'unknown-model';
      const { prettyName, card } = extractModelCard(payload);

      if (!rowMap.has(modelId)) {
        rowMap.set(modelId, {
          modelId,
          prettyName,
          cells: {},
          shardMetadata: extractShardMetadata(payload),
          modelCard: card,
        });
      }
      const row = rowMap.get(modelId)!;
      if (prettyName && !row.prettyName) row.prettyName = prettyName;
      if (!row.shardMetadata) row.shardMetadata = extractShardMetadata(payload);
      if (!row.modelCard && card) row.modelCard = card;

      const modelDirectory =
        ((payload.model_directory ?? payload.modelDirectory) as string) ||
        undefined;

      let cell: CellStatus;
      if (tag === 'DownloadCompleted') {
        const totalBytes = getBytes(
          (payload.total_bytes ?? payload.totalBytes) ?? 0,
        );
        cell = { kind: 'completed', totalBytes, modelDirectory };
      } else if (tag === 'DownloadOngoing') {
        const raw = payload as Record<string, unknown>;
        const dl = raw.download_progress ?? raw.downloadProgress ?? raw;
        const dlObj = dl as Record<string, unknown>;
        const percentage = (dlObj.percentage as number) ?? 0;
        const downloadedBytes = getBytes(
          dlObj.downloaded_bytes ?? dlObj.downloadedBytes ?? 0,
        );
        const totalBytes = getBytes(dlObj.total_bytes ?? dlObj.totalBytes ?? 0);
        const speed = getBytes(dlObj.speed ?? 0);
        const etaMs = (dlObj.eta_ms ?? dlObj.etaMs ?? 0) as number;
        cell = {
          kind: 'downloading',
          percentage,
          downloadedBytes,
          totalBytes,
          speed,
          etaMs,
          modelDirectory,
        };
      } else if (tag === 'DownloadPending') {
        const raw = payload as Record<string, unknown>;
        const downloaded = (raw.downloaded ?? 0) as number;
        const total = (raw.total ?? 0) as number;
        cell = { kind: 'pending', downloaded, total, modelDirectory };
      } else if (tag === 'DownloadFailed') {
        cell = { kind: 'failed', modelDirectory };
      } else {
        cell = { kind: 'not_present' };
      }

      const existing = row.cells[nodeId];
      if (!existing || shouldUpgradeCell(existing, cell)) {
        row.cells[nodeId] = cell;
      }
    }
  }

  const modelRows = Array.from(rowMap.values());
  return { modelRows, nodeColumns };
}

export function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return '0B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const val = bytes / Math.pow(1024, i);
  return `${val.toFixed(val >= 10 ? 0 : 1)}${units[i]}`;
}

export function formatEta(ms: number): string {
  if (!ms || ms <= 0) return '--';
  const totalSeconds = Math.round(ms / 1000);
  const s = totalSeconds % 60;
  const m = Math.floor(totalSeconds / 60) % 60;
  const h = Math.floor(totalSeconds / 3600);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function formatSpeed(bytesPerSecond: number): string {
  if (!bytesPerSecond || bytesPerSecond <= 0) return '--';
  const units = ['B/s', 'KB/s', 'MB/s', 'GB/s'];
  const i = Math.min(
    Math.floor(Math.log(bytesPerSecond) / Math.log(1024)),
    units.length - 1,
  );
  const val = bytesPerSecond / Math.pow(1024, i);
  return `${val.toFixed(val >= 10 ? 0 : 1)}${units[i]}`;
}
