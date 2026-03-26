import { useCallback, useEffect, useMemo, useState } from 'react';
import styled, { css } from 'styled-components';
import type { TopologyData } from '../../types/topology';
import type { RawDownloads, NodeDiskInfo } from '../../hooks/useClusterState';
import { StoreRegistryTable, type StoreRegistryEntry } from '../layout/StoreRegistryTable';

type Tab = 'nodes' | 'store';

interface DownloadsPageProps {
  topology: TopologyData | null;
  downloads: RawDownloads;
  nodeDisk: NodeDiskInfo;
  lastUpdate: number | null;
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

function formatSpeed(bps: number): string {
  if (!bps || bps <= 0) return '--';
  const units = ['B/s', 'KB/s', 'MB/s', 'GB/s'];
  const i = Math.min(Math.floor(Math.log(bps) / Math.log(1024)), units.length - 1);
  const val = bps / Math.pow(1024, i);
  return `${val.toFixed(val >= 10 ? 0 : 1)} ${units[i]}`;
}

/* ── Component ────────────────────────────────────────── */

export function DownloadsPage({ topology, downloads, nodeDisk, lastUpdate }: DownloadsPageProps) {
  const [tab, setTab] = useState<Tab | null>(null);
  const [storeAvailable, setStoreAvailable] = useState(false);
  const [storeEntries, setStoreEntries] = useState<StoreRegistryEntry[]>([]);
  const [storeLoading, setStoreLoading] = useState(false);

  const { rows, columns } = useMemo(
    () => buildGrid(downloads, topology, nodeDisk),
    [downloads, topology, nodeDisk],
  );

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

  const activeTab = tab ?? 'nodes';

  return (
    <Container>
      <Header>
        <div>
          <Title>DOWNLOADS</Title>
          <Subtitle>Overview of models on each node</Subtitle>
        </div>
        <LastUpdate>
          Last update: {lastUpdate ? new Date(lastUpdate).toLocaleTimeString() : '--'}
        </LastUpdate>
      </Header>

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
          onRefresh={loadRegistry}
          onInfo={() => {}}
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

const Header = styled.div`
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 24px;
  position: relative;
`;

const Title = styled.h1`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 28px;
  font-weight: 700;
  color: #FFD700;
  margin: 0;
  letter-spacing: 2px;
`;

const Subtitle = styled.p`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  color: ${({ theme }) => theme.colors.textMuted};
  margin: 4px 0 0;
`;

const LastUpdate = styled.div`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 11px;
  color: ${({ theme }) => theme.colors.textMuted};
  white-space: nowrap;
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
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1px;
  transition: all 0.15s;

  ${({ $active }) =>
    $active
      ? css`
          background: #FFD700;
          color: #111;
          font-weight: 600;
        `
      : css`
          background: rgba(0, 0, 0, 0.3);
          color: rgba(179, 179, 179, 0.8);
          &:hover {
            color: #fff;
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
  border: 1px solid rgba(179, 179, 179, 0.2);
  border-radius: ${({ theme }) => theme.radii.md};
  background: rgba(0, 0, 0, 0.2);
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  color: ${({ theme }) => theme.colors.textMuted};
`;

const TableWrap = styled.div`
  border: 1px solid rgba(179, 179, 179, 0.2);
  border-radius: ${({ theme }) => theme.radii.md};
  background: rgba(0, 0, 0, 0.2);
  overflow-x: auto;
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
`;

const ModelHeader = styled.th`
  text-align: left;
  padding: 16px 20px;
  color: #FFD700;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  border-bottom: 1px solid rgba(179, 179, 179, 0.15);
`;

const NodeHeader = styled.th`
  text-align: center;
  padding: 12px 16px;
  border-bottom: 1px solid rgba(179, 179, 179, 0.15);
`;

const NodeName = styled.div`
  color: ${({ theme }) => theme.colors.text};
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.5px;
`;

const DiskFree = styled.div`
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: 10px;
  margin-top: 2px;
`;

const ModelCell = styled.td`
  padding: 16px 20px;
  color: ${({ theme }) => theme.colors.text};
  font-size: 12px;
  border-bottom: 1px solid rgba(179, 179, 179, 0.08);
  white-space: nowrap;
`;

const StatusCell = styled.td`
  text-align: center;
  padding: 12px 16px;
  border-bottom: 1px solid rgba(179, 179, 179, 0.08);
`;

const CellInner = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
`;

const CellSize = styled.div`
  font-size: 11px;
  color: ${({ theme }) => theme.colors.textMuted};
`;

const ProgressText = styled.div<{ $color: string }>`
  font-size: 13px;
  font-weight: 600;
  color: ${({ $color }) => $color};
`;

const FailedText = styled.div`
  font-size: 11px;
  font-weight: 600;
  color: ${({ theme }) => theme.colors.error};
  text-transform: uppercase;
`;

const NotPresent = styled.div`
  color: ${({ theme }) => theme.colors.textMuted};
`;
