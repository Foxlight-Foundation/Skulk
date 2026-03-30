import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled, { css } from 'styled-components';
import type { TopologyData } from '../../types/topology';
import type { RawDownloads, NodeDiskInfo, RawInstances, RawRunners } from '../../hooks/useClusterState';
import { StoreRegistryTable, type StoreRegistryEntry, type StoreDownloadProgress, type ModelCardInfo } from '../layout/StoreRegistryTable';
import type { ClusterCardProps, ClusterCardNode } from '../cluster/ClusterCard';
import { ModelSearchModal } from './ModelSearchModal';
import { FiTrash2, FiSearch, FiCheck } from 'react-icons/fi';
import { Button } from '../common/Button';
import { addToast } from '../../hooks/useToast';
import { PlacementManager } from '../cluster/PlacementManager';

interface ModelStorePageProps {
  topology: TopologyData | null;
  downloads: RawDownloads;
  nodeDisk: NodeDiskInfo;
  instances: RawInstances;
  runners: RawRunners;
  onChat?: (modelId: string) => void;
}

/* ── Data extraction helpers ──────────────────────────── */

type CellKind = 'completed' | 'downloading' | 'pending' | 'failed' | 'not_present';

interface CellStatus {
  kind: CellKind;
  totalBytes: number;
  downloadedBytes?: number;
  percentage?: number;
  speed?: number;
}

interface ModelRow {
  modelId: string;
  cells: Record<string, CellStatus>;
}

interface NodeColumn {
  nodeId: string;
  label: string;
  diskFreeBytes?: number;
}

function getBytes(v: unknown): number {
  if (typeof v === 'number') return v;
  if (v && typeof v === 'object') {
    const obj = v as Record<string, unknown>;
    if (typeof obj.inBytes === 'number') return obj.inBytes;
  }
  return 0;
}

function getTag(entry: unknown): [string, Record<string, unknown>] | null {
  if (!entry || typeof entry !== 'object') return null;
  const obj = entry as Record<string, unknown>;
  for (const key of ['DownloadCompleted', 'DownloadOngoing', 'DownloadPending', 'DownloadFailed']) {
    if (key in obj && obj[key] && typeof obj[key] === 'object') {
      return [key, obj[key] as Record<string, unknown>];
    }
  }
  return null;
}

function extractModelId(payload: Record<string, unknown>): string | null {
  const shard = (payload.shardMetadata ?? payload.shard_metadata) as Record<string, unknown> | undefined;
  if (!shard) return null;
  for (const key of Object.keys(shard)) {
    const inner = shard[key] as Record<string, unknown> | undefined;
    const card = inner?.modelCard ?? inner?.model_card;
    if (card && typeof card === 'object') {
      const c = card as Record<string, unknown>;
      if (typeof c.modelId === 'string') return c.modelId;
      if (typeof c.model_id === 'string') return c.model_id;
    }
  }
  return null;
}

function buildGrid(
  downloads: RawDownloads,
  topology: TopologyData | null,
  nodeDisk: NodeDiskInfo,
): { rows: ModelRow[]; columns: NodeColumn[] } {
  const allNodeIds = Object.keys(downloads);
  if (allNodeIds.length === 0) return { rows: [], columns: [] };

  const columns: NodeColumn[] = allNodeIds.map((nodeId) => ({
    nodeId,
    label: topology?.nodes[nodeId]?.friendly_name ?? nodeId.slice(0, 8),
    diskFreeBytes: nodeDisk[nodeId]?.available?.inBytes,
  }));

  const rowMap = new Map<string, ModelRow>();

  for (const [nodeId, entries] of Object.entries(downloads)) {
    const list = Array.isArray(entries) ? entries : Object.values(entries);
    for (const entry of list) {
      const tagged = getTag(entry);
      if (!tagged) continue;
      const [tag, payload] = tagged;
      const modelId = extractModelId(payload) ?? 'unknown';

      if (!rowMap.has(modelId)) {
        rowMap.set(modelId, { modelId, cells: {} });
      }
      const row = rowMap.get(modelId)!;

      if (tag === 'DownloadCompleted') {
        row.cells[nodeId] = { kind: 'completed', totalBytes: getBytes(payload.total) };
      } else if (tag === 'DownloadOngoing') {
        const prog = (payload.downloadProgress ?? payload.download_progress ?? {}) as Record<string, unknown>;
        const total = getBytes(prog.total ?? payload.total);
        const downloaded = getBytes(prog.downloaded);
        row.cells[nodeId] = {
          kind: 'downloading',
          totalBytes: total,
          downloadedBytes: downloaded,
          percentage: total > 0 ? (downloaded / total) * 100 : 0,
          speed: (prog.speed as number) ?? 0,
        };
      } else if (tag === 'DownloadPending') {
        row.cells[nodeId] = {
          kind: 'pending',
          totalBytes: getBytes(payload.total),
          downloadedBytes: getBytes(payload.downloaded),
          percentage: 0,
        };
      } else if (tag === 'DownloadFailed') {
        row.cells[nodeId] = { kind: 'failed', totalBytes: 0 };
      }
    }
  }

  return { rows: Array.from(rowMap.values()), columns };
}

function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return '0B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const val = bytes / Math.pow(1024, i);
  return `${val.toFixed(val >= 10 ? 0 : 1)}${units[i]}`;
}

const TrashIcon = () => <FiTrash2 size={14} />;
const SearchIcon = () => <FiSearch size={14} />;

function formatSpeed(bps: number): string {
  if (!bps || bps <= 0) return '--';
  const units = ['B/s', 'KB/s', 'MB/s', 'GB/s'];
  const i = Math.min(Math.floor(Math.log(bps) / Math.log(1024)), units.length - 1);
  const val = bps / Math.pow(1024, i);
  return `${val.toFixed(val >= 10 ? 0 : 1)} ${units[i]}`;
}

/* ── Component ────────────────────────────────────────── */

export function ModelStorePage({ topology, downloads, nodeDisk, instances, runners, onChat }: ModelStorePageProps) {
  const [storeEntries, setStoreEntries] = useState<StoreRegistryEntry[]>([]);
  const [storeDownloads, setStoreDownloads] = useState<StoreDownloadProgress[]>([]);
  const [storeLoading, setStoreLoading] = useState(false);
  const [placementModelId, setPlacementModelId] = useState<string | null>(null);
  const [apiModelTags, setApiModelTags] = useState<Record<string, string[]>>({});
  const pollRef = useRef<ReturnType<typeof setInterval>>(undefined);

  // Fetch model tags from /models API
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch('/models');
        if (!res.ok) return;
        const data = await res.json();
        const tags: Record<string, string[]> = {};
        for (const m of data.data ?? []) {
          if (m.id && m.tags && m.tags.length > 0) tags[m.id] = m.tags;
        }
        setApiModelTags(tags);
      } catch { /* ignore */ }
    })();
  }, []);

  // Extract model card info from download data for the store table tooltips
  const modelCards = useMemo<Record<string, ModelCardInfo>>(() => {
    const cards: Record<string, ModelCardInfo> = {};
    for (const entries of Object.values(downloads)) {
      const list = Array.isArray(entries) ? entries : Object.values(entries);
      for (const entry of list) {
        const tagged = getTag(entry);
        if (!tagged) continue;
        const [, payload] = tagged;
        const shard = (payload.shardMetadata ?? payload.shard_metadata) as Record<string, unknown> | undefined;
        if (!shard) continue;
        for (const val of Object.values(shard)) {
          if (!val || typeof val !== 'object') continue;
          const inner = val as Record<string, unknown>;
          const raw = (inner.modelCard ?? inner.model_card) as Record<string, unknown> | undefined;
          if (!raw) continue;
          const mid = (raw.modelId ?? raw.model_id) as string | undefined;
          if (!mid || cards[mid]) continue;
          const tags: string[] = [];
          if (mid.toLowerCase().includes('optiq') || (raw.quantization as string ?? '').toLowerCase().includes('optiq')) tags.push('optiq');
          if ((raw.capabilities as string[] ?? []).includes('thinking')) tags.push('thinking');
          if ((raw.supportsTensor ?? raw.supports_tensor) as boolean) tags.push('tensor');
          cards[mid] = {
            family: raw.family as string | undefined,
            quantization: raw.quantization as string | undefined,
            baseModel: (raw.baseModel ?? raw.base_model) as string | undefined,
            supportsTensor: (raw.supportsTensor ?? raw.supports_tensor) as boolean | undefined,
            capabilities: raw.capabilities as string[] | undefined,
            tags,
          };
        }
      }
    }
    // Merge tags from /models API for models not in download state
    for (const [mid, tags] of Object.entries(apiModelTags)) {
      if (!cards[mid]) {
        cards[mid] = { tags };
      } else if (!cards[mid].tags || cards[mid].tags!.length === 0) {
        cards[mid].tags = tags;
      }
    }
    return cards;
  }, [downloads, apiModelTags]);

  const fetchDownloads = useCallback(async () => {
    try {
      const res = await fetch('/store/downloads');
      if (!res.ok) return [];
      const data = await res.json();
      return (data.downloads ?? []) as StoreDownloadProgress[];
    } catch { return []; }
  }, []);

  const loadRegistry = useCallback(async () => {
    setStoreLoading(true);
    try {
      const [regRes, dls] = await Promise.all([
        fetch('/store/registry'),
        fetchDownloads(),
      ]);
      if (regRes.ok) {
        const data = await regRes.json();
        setStoreEntries(data.entries ?? []);
      }
      setStoreDownloads(dls);

      // Start polling if active downloads
      if (dls.length > 0 && !pollRef.current) {
        pollRef.current = setInterval(async () => {
          const [newDls, regRefresh] = await Promise.all([
            fetchDownloads(),
            fetch('/store/registry'),
          ]);
          setStoreDownloads(newDls);
          if (regRefresh.ok) {
            const d = await regRefresh.json();
            setStoreEntries(d.entries ?? []);
          }
          // Stop polling when done
          if (newDls.length === 0 && pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = undefined;
          }
        }, 2000);
      }
    } catch { /* ignore */ }
    finally { setStoreLoading(false); }
  }, [fetchDownloads]);

  // Load store registry on mount
  useEffect(() => {
    loadRegistry();
  }, [loadRegistry]);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const [purgeConfirm, setPurgeConfirm] = useState(false);
  const [purging, setPurging] = useState(false);

  const handlePurge = useCallback(async () => {
    setPurging(true);
    try {
      const res = await fetch('/store/purge-staging', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modelId: null }),
      });
      if (res.ok) {
        addToast({ type: 'success', message: 'Purge command sent to all nodes' });
      } else {
        addToast({ type: 'error', message: 'Failed to send purge command' });
      }
    } catch {
      addToast({ type: 'error', message: 'Failed to send purge command' });
    } finally {
      setPurging(false);
      setPurgeConfirm(false);
    }
  }, []);

  const [searchOpen, setSearchOpen] = useState(false);

  const storeModelIds = useMemo(
    () => new Set(storeEntries.map((e) => e.model_id)),
    [storeEntries],
  );

  // Total cluster available (free) RAM
  const totalClusterMemoryBytes = useMemo(() => {
    if (!topology?.nodes) return 0;
    return Object.values(topology.nodes).reduce((sum, node) => {
      const mem = node.macmon_info?.memory;
      if (!mem) return sum;
      const available = mem.ram_total - mem.ram_usage;
      return sum + Math.max(available, 0);
    }, 0);
  }, [topology]);

  // Derive active model IDs and model→instanceId mapping from running instances
  const { activeModelIds, modelToInstanceId } = useMemo(() => {
    const ids: string[] = [];
    const mapping: Record<string, string> = {};
    for (const [iid, inst] of Object.entries(instances)) {
      const inner = inst.MlxRingInstance ?? inst.MlxJacclInstance;
      const modelId = inner?.shardAssignments?.modelId;
      if (modelId) {
        ids.push(modelId);
        mapping[modelId] = (inner as Record<string, unknown>).instanceId as string ?? iid;
      }
    }
    return { activeModelIds: ids, modelToInstanceId: mapping };
  }, [instances]);

  // Build ClusterCard data for active models
  const clusterCards = useMemo<Record<string, Omit<ClusterCardProps, 'onLaunch'>>>(() => {
    const cards: Record<string, Omit<ClusterCardProps, 'onLaunch'>> = {};
    for (const [, inst] of Object.entries(instances)) {
      const isRing = !!inst.MlxRingInstance;
      const inner = inst.MlxRingInstance ?? inst.MlxJacclInstance;
      if (!inner) continue;
      const sa = inner.shardAssignments;
      const modelId = sa?.modelId;
      if (!modelId) continue;

      const nodeToRunner = sa?.nodeToRunner;
      const nodeIds = nodeToRunner ? Object.keys(nodeToRunner) : [];
      const runnerIds = nodeToRunner ? Object.values(nodeToRunner) : [];

      // Derive sharding from shard metadata (tagged union: { "PipelineShardMetadata": {...} } etc.)
      const runnerToShard = sa?.runnerToShard;
      let sharding: 'Pipeline' | 'Tensor' = 'Pipeline';
      if (runnerToShard) {
        const firstShard = Object.values(runnerToShard)[0];
        if (firstShard && 'TensorShardMetadata' in firstShard) {
          sharding = 'Tensor';
        }
      }

      // Gate isReady on all runners being in RunnerReady or RunnerRunning state
      const isReady = runnerIds.length > 0 && runnerIds.every((rid) => {
        const status = runners[rid];
        if (!status) return false;
        return 'RunnerReady' in status || 'RunnerRunning' in status;
      });

      const cardNodes: ClusterCardNode[] = nodeIds.map((nid) => {
        const nodeInfo = topology?.nodes[nid];
        const mem = nodeInfo?.macmon_info?.memory;
        const pct = mem && mem.ram_total > 0
          ? Math.round((mem.ram_usage / mem.ram_total) * 100)
          : 0;
        return {
          nodeId: nid,
          name: nodeInfo?.friendly_name ?? nid.slice(0, 8),
          memoryUsedPercent: pct,
        };
      });

      const storeEntry = storeEntries.find((e) => e.model_id === modelId);

      cards[modelId] = {
        modelId,
        sizeBytes: storeEntry?.total_bytes,
        sharding,
        instanceType: isRing ? 'MlxRing' : 'MlxJaccl',
        nodes: cardNodes,
        isReady,
      };
    }
    return cards;
  }, [instances, runners, topology, storeEntries]);

  const handleLaunchWithParams = useCallback(async (params: { modelId: string; sharding: string; instanceMeta: string; minNodes: number }) => {
    try {
      const res = await fetch('/place_instance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: params.modelId,
          sharding: params.sharding,
          instance_meta: params.instanceMeta,
          min_nodes: params.minNodes,
        }),
      });
      if (res.ok) {
        addToast({ type: 'success', message: `Launching ${params.modelId}` });
      } else {
        const err = await res.json().catch(() => ({}));
        addToast({ type: 'error', message: (err as Record<string, string>).detail ?? `Failed to launch ${params.modelId}` });
      }
    } catch {
      addToast({ type: 'error', message: `Failed to launch model` });
    }
  }, []);

  const handleLaunch = useCallback((modelId: string) => {
    handleLaunchWithParams({ modelId, sharding: 'Pipeline', instanceMeta: 'MlxRing', minNodes: 1 });
  }, [handleLaunchWithParams]);

  const handleStop = useCallback(async (modelId: string) => {
    const instanceId = modelToInstanceId[modelId];
    if (!instanceId) {
      addToast({ type: 'error', message: `No active instance found for ${modelId}` });
      return;
    }
    try {
      const res = await fetch(`/instance/${encodeURIComponent(instanceId)}`, { method: 'DELETE' });
      if (res.ok) {
        addToast({ type: 'success', message: `Stopping ${modelId}` });
      } else {
        addToast({ type: 'error', message: `Failed to stop ${modelId}` });
      }
    } catch {
      addToast({ type: 'error', message: `Failed to stop model` });
    }
  }, [modelToInstanceId]);

  const handleDelete = useCallback(async (entry: { model_id: string }, isActive: boolean) => {
    if (isActive) {
      addToast({ type: 'error', message: 'Stop the model before deleting' });
      return;
    }
    try {
      const res = await fetch(`/store/models/${encodeURIComponent(entry.model_id)}`, { method: 'DELETE' });
      if (res.ok) {
        addToast({ type: 'success', message: `Deleted ${entry.model_id}` });
        loadRegistry();
      } else {
        addToast({ type: 'error', message: `Failed to delete ${entry.model_id}` });
      }
    } catch {
      addToast({ type: 'error', message: 'Failed to delete model' });
    }
  }, [loadRegistry]);

  const handleOptimize = useCallback(async (modelId: string) => {
    try {
      const res = await fetch(`/store/models/${encodeURIComponent(modelId)}/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_bpw: 4.5, candidate_bits: [4, 8] }),
      });
      if (res.ok) {
        addToast({ type: 'success', message: `OptiQ optimization started for ${modelId}` });
        // Poll for completion/failure
        const pollInterval = setInterval(async () => {
          try {
            const statusRes = await fetch(`/store/models/${encodeURIComponent(modelId)}/optimize/status`);
            if (!statusRes.ok) { clearInterval(pollInterval); return; }
            const status = await statusRes.json();
            if (status.status === 'complete') {
              clearInterval(pollInterval);
              addToast({ type: 'success', message: `OptiQ optimization complete: ${status.achievedBpw?.toFixed(2) ?? '?'} BPW` });
              loadRegistry();
            } else if (status.status === 'failed') {
              clearInterval(pollInterval);
              addToast({ type: 'error', message: status.error ?? 'Optimization failed' });
            }
          } catch { clearInterval(pollInterval); }
        }, 5000);
      } else {
        const err = await res.json().catch(() => ({}));
        addToast({ type: 'error', message: (err as Record<string, string>).detail ?? 'Failed to start optimization' });
      }
    } catch {
      addToast({ type: 'error', message: 'Failed to start optimization' });
    }
  }, [loadRegistry]);

  return (
    <Container>
      {purgeConfirm && (
        <PurgeModal>
          <ModalBackdrop onClick={() => setPurgeConfirm(false)} />
          <ModalBox>
            <ModalTitle>Purge all node caches</ModalTitle>
            <ModalText>
              This will remove all staged model files and partial downloads from
              every node in the cluster. Models will need to be re-downloaded
              before they can run again.
            </ModalText>
            <ModalNote>
              Nodes that are currently offline will not receive this command.
            </ModalNote>
            <ModalActions>
              <Button variant="outline" size="md" onClick={() => setPurgeConfirm(false)}>
                Cancel
              </Button>
              <Button variant="danger" size="md" loading={purging} onClick={handlePurge}>
                {purging ? 'Purging...' : 'Purge All'}
              </Button>
            </ModalActions>
          </ModalBox>
        </PurgeModal>
      )}

      <StoreRegistryTable
          entries={storeEntries}
          activeDownloads={storeDownloads}
          loading={storeLoading}
          activeModelIds={activeModelIds}
          modelCards={modelCards}
          actions={
            <>
              <Button variant="danger" size="sm" onClick={() => setPurgeConfirm(true)}>
                <TrashIcon /> Purge Node Caches
              </Button>
              <Button variant="primary" size="sm" onClick={() => setSearchOpen(true)}>
                <SearchIcon /> Find Models
              </Button>
            </>
          }
          onRefresh={loadRegistry}
          onDelete={handleDelete}
          onLaunch={handleLaunch}
          onStop={handleStop}
          onChat={onChat}
          onPlacement={setPlacementModelId}
          clusterCards={clusterCards}
          totalClusterMemoryBytes={totalClusterMemoryBytes}
          onOptimize={handleOptimize}
        />
      <ModelSearchModal
        open={searchOpen}
        onClose={() => setSearchOpen(false)}
        existingModelIds={storeModelIds}
        onDownloadStarted={loadRegistry}
      />
      {placementModelId && topology && (
        <PlacementManager
          modelId={placementModelId}
          modelSizeMb={storeEntries.find((e) => e.model_id === placementModelId)?.total_bytes
            ? Math.round(storeEntries.find((e) => e.model_id === placementModelId)!.total_bytes / 1e6)
            : undefined}
          topology={topology}
          open={!!placementModelId}
          onClose={() => setPlacementModelId(null)}
          onLaunch={handleLaunchWithParams}
        />
      )}
    </Container>
  );
}

function CellContent({ cell }: { cell: CellStatus }) {
  switch (cell.kind) {
    case 'completed':
      return (
        <CellInner>
          <CheckIcon />
          <CellSize>{formatBytes(cell.totalBytes)}</CellSize>
        </CellInner>
      );
    case 'downloading':
      return (
        <CellInner>
          <ProgressText $color="#FFD700">
            {(cell.percentage ?? 0).toFixed(1)}%
          </ProgressText>
          <CellSize>{formatSpeed(cell.speed ?? 0)}</CellSize>
        </CellInner>
      );
    case 'pending':
      return (
        <CellInner>
          <ProgressText $color="#FFD700">
            {cell.downloadedBytes && cell.totalBytes
              ? `${((cell.downloadedBytes / cell.totalBytes) * 100).toFixed(1)}%`
              : '0.0%'}
          </ProgressText>
          <CellSize>--</CellSize>
        </CellInner>
      );
    case 'failed':
      return <FailedText>Failed</FailedText>;
    default:
      return <NotPresent>--</NotPresent>;
  }
}

function CheckIcon() {
  return <FiCheck size={20} color="#4ade80" strokeWidth={2.5} />;
}

/* ── Styles ───────────────────────────────────────────── */

const Container = styled.div`
  padding: 32px;
  height: 100%;
  overflow-y: auto;
`;

const TopBar = styled.div`
  display: flex;
  justify-content: flex-end;
  margin-bottom: 20px;
`;

const PurgeModal = styled.div`
  position: fixed;
  inset: 0;
  z-index: 60;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const ModalBackdrop = styled.div`
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
`;

const ModalBox = styled.div`
  position: relative;
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: 24px;
  width: 420px;
  max-width: 90vw;
`;

const ModalTitle = styled.h3`
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.error};
  margin: 0 0 12px;
`;

const ModalText = styled.p`
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textSecondary};
  line-height: 1.5;
  margin: 0 0 8px;
`;

const ModalNote = styled.p`
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  margin: 0 0 16px;
`;

const ModalActions = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: 12px;
`;

const SegmentedToggle = styled.div`
  display: flex;
  gap: 0;
  margin-bottom: 20px;
`;

const SegmentBtn = styled.button<{ $active: boolean }>`
  all: unset;
  cursor: pointer;
  padding: 8px 20px;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.nav};
  transition: all 0.15s;

  ${({ $active, theme }) =>
    $active
      ? css`
          background: ${theme.colors.gold};
          color: ${theme.colors.surface};
          font-weight: 600;
        `
      : css`
          background: rgba(0, 0, 0, 0.3);
          color: ${theme.colors.textSecondary};
          &:hover {
            color: ${theme.colors.text};
          }
        `}

  &:first-child {
    border-radius: 4px 0 0 4px;
  }
  &:last-child {
    border-radius: 0 4px 4px 0;
  }
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 48px 24px;
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  background: rgba(0, 0, 0, 0.2);
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const TableWrap = styled.div`
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  background: rgba(0, 0, 0, 0.2);
  overflow-x: auto;
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
`;

const ModelHeader = styled.th`
  text-align: left;
  padding: 16px 20px;
  color: ${({ theme }) => theme.colors.gold};
  font-size: ${({ theme }) => theme.fontSizes.tableHead};
  font-weight: 600;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
`;

const NodeHeader = styled.th`
  text-align: center;
  padding: 12px 16px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
`;

const NodeName = styled.div`
  color: ${({ theme }) => theme.colors.text};
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  font-weight: 600;
`;

const DiskFree = styled.div`
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  margin-top: 2px;
`;

const ModelCell = styled.td`
  padding: 16px 20px;
  color: ${({ theme }) => theme.colors.text};
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  white-space: nowrap;
`;

const StatusCell = styled.td`
  text-align: center;
  padding: 12px 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
`;

const CellInner = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
`;

const CellSize = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const ProgressText = styled.div<{ $color: string }>`
  font-size: ${({ theme }) => theme.fontSizes.md};
  font-weight: 600;
  color: ${({ $color }) => $color};
`;

const FailedText = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.error};
`;

const NotPresent = styled.div`
  color: ${({ theme }) => theme.colors.textMuted};
`;
