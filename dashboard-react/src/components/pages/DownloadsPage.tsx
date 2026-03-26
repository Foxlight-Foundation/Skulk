import { useCallback, useEffect, useMemo, useState } from 'react';
import styled, { css } from 'styled-components';
import type { TopologyData } from '../../types/topology';
import type { RawDownloads, NodeDiskInfo } from '../../hooks/useClusterState';
import { StoreRegistryTable, type StoreRegistryEntry, type ModelCardInfo } from '../layout/StoreRegistryTable';
import { Button } from '../common/Button';
import { addToast } from '../../hooks/useToast';

type Tab = 'nodes' | 'store';

interface DownloadsPageProps {
  topology: TopologyData | null;
  downloads: RawDownloads;
  nodeDisk: NodeDiskInfo;
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

const TrashIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
  </svg>
);

const SearchIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

function formatSpeed(bps: number): string {
  if (!bps || bps <= 0) return '--';
  const units = ['B/s', 'KB/s', 'MB/s', 'GB/s'];
  const i = Math.min(Math.floor(Math.log(bps) / Math.log(1024)), units.length - 1);
  const val = bps / Math.pow(1024, i);
  return `${val.toFixed(val >= 10 ? 0 : 1)} ${units[i]}`;
}

/* ── Component ────────────────────────────────────────── */

export function DownloadsPage({ topology, downloads, nodeDisk }: DownloadsPageProps) {
  const [tab, setTab] = useState<Tab | null>(null);
  const [storeAvailable, setStoreAvailable] = useState(false);
  const [storeEntries, setStoreEntries] = useState<StoreRegistryEntry[]>([]);
  const [storeLoading, setStoreLoading] = useState(false);

  const { rows, columns } = useMemo(
    () => buildGrid(downloads, topology, nodeDisk),
    [downloads, topology, nodeDisk],
  );

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
          cards[mid] = {
            family: raw.family as string | undefined,
            quantization: raw.quantization as string | undefined,
            baseModel: (raw.baseModel ?? raw.base_model) as string | undefined,
            supportsTensor: (raw.supportsTensor ?? raw.supports_tensor) as boolean | undefined,
            capabilities: raw.capabilities as string[] | undefined,
          };
        }
      }
    }
    return cards;
  }, [downloads]);

  const loadRegistry = useCallback(async () => {
    setStoreLoading(true);
    try {
      const res = await fetch('/store/registry');
      if (!res.ok) return;
      const data = await res.json();
      setStoreEntries(data.entries ?? []);
    } catch { /* ignore */ }
    finally { setStoreLoading(false); }
  }, []);

  // Detect store availability and set default tab
  useEffect(() => {
    if (tab !== null) return;
    (async () => {
      try {
        const res = await fetch('/store/health');
        if (res.ok) {
          setStoreAvailable(true);
          setTab('store');
          return;
        }
      } catch { /* ignore */ }
      setTab('nodes');
    })();
  }, [tab]);

  // Load store registry when switching to store tab
  useEffect(() => {
    if (tab === 'store') loadRegistry();
  }, [tab, loadRegistry]);

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

  const handleFindModels = useCallback(() => {
    // TODO: wire up model picker modal
    addToast({ type: 'info', message: 'Model search coming soon' });
  }, []);

  const activeTab = tab ?? 'nodes';

  return (
    <Container>
      <TopBar>
        {storeAvailable && (
          <SegmentedToggle>
            <SegmentBtn $active={activeTab === 'nodes'} onClick={() => setTab('nodes')}>
              Node Downloads
            </SegmentBtn>
            <SegmentBtn $active={activeTab === 'store'} onClick={() => setTab('store')}>
              Store Registry
            </SegmentBtn>
          </SegmentedToggle>
        )}
        <ActionButtons>
          {storeAvailable && (
            <Button variant="danger" size="sm" onClick={() => setPurgeConfirm(true)}>
              <TrashIcon /> Purge All Node Caches
            </Button>
          )}
          {storeAvailable && activeTab === 'store' && (
            <Button variant="primary" size="md" onClick={handleFindModels}>
              <SearchIcon /> Find Models
            </Button>
          )}
        </ActionButtons>
      </TopBar>

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

      {activeTab === 'nodes' ? (
        rows.length === 0 ? (
          <EmptyState>
            No downloads found. Start a model download to see progress here.
          </EmptyState>
        ) : (
          <TableWrap>
            <Table>
              <thead>
                <tr>
                  <ModelHeader>MODEL</ModelHeader>
                  {columns.map((col) => (
                    <NodeHeader key={col.nodeId}>
                      <NodeName>{col.label.toUpperCase()}</NodeName>
                      {col.diskFreeBytes != null && (
                        <DiskFree>{formatBytes(col.diskFreeBytes)} free</DiskFree>
                      )}
                    </NodeHeader>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.modelId}>
                    <ModelCell>{row.modelId}</ModelCell>
                    {columns.map((col) => {
                      const cell = row.cells[col.nodeId];
                      return (
                        <StatusCell key={col.nodeId}>
                          {cell ? <CellContent cell={cell} /> : <NotPresent>--</NotPresent>}
                        </StatusCell>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </Table>
          </TableWrap>
        )
      ) : (
        <StoreRegistryTable
          entries={storeEntries}
          loading={storeLoading}
          modelCards={modelCards}
          onRefresh={loadRegistry}
          onDelete={() => {}}
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
      return <FailedText>FAILED</FailedText>;
    default:
      return <NotPresent>--</NotPresent>;
  }
}

function CheckIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4ade80" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

/* ── Styles ───────────────────────────────────────────── */

const Container = styled.div`
  padding: 32px;
  height: 100%;
  overflow-y: auto;
`;

const TopBar = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
  gap: 12px;
`;

const ActionButtons = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
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
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  text-transform: uppercase;
  letter-spacing: 1px;
  color: ${({ theme }) => theme.colors.error};
  margin: 0 0 12px;
`;

const ModalText = styled.p`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textSecondary};
  line-height: 1.5;
  margin: 0 0 8px;
`;

const ModalNote = styled.p`
  font-family: ${({ theme }) => theme.fonts.mono};
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
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.nav};
  text-transform: uppercase;
  letter-spacing: 1px;
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
  font-family: ${({ theme }) => theme.fonts.mono};
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
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
`;

const ModelHeader = styled.th`
  text-align: left;
  padding: 16px 20px;
  color: ${({ theme }) => theme.colors.gold};
  font-size: ${({ theme }) => theme.fontSizes.tableHead};
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
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
  letter-spacing: 0.5px;
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
  text-transform: uppercase;
`;

const NotPresent = styled.div`
  color: ${({ theme }) => theme.colors.textMuted};
`;
