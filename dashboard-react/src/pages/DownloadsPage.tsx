/**
 * DownloadsPage  —  "/downloads"
 *
 * Model download management grid: rows = models, columns = nodes.
 * Mirrors the Svelte downloads/+page.svelte implementation.
 */
import React, { useMemo } from 'react';
import styled from 'styled-components';
import { useTopologyStore } from '../stores/topologyStore';
import {
  buildDownloadGrid,
  formatBytes,
  formatEta,
  formatSpeed,
} from '../utils/downloads';
import type { CellStatus, ModelRow, NodeColumn } from '../utils/downloads';
import { startDownloadForNode, deleteDownload } from '../api/client';

// ─── Styled components ────────────────────────────────────────────────────────

const Page = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 24px;
  overflow-y: auto;
`;

const PageTitle = styled.h1`
  margin: 0 0 24px;
  font-size: 13px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.yellow};
`;

const TableWrapper = styled.div`
  overflow-x: auto;
`;

const Grid = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 11px;
`;

const Th = styled.th`
  padding: 8px 12px;
  text-align: left;
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.lightGray};
  border-bottom: 1px solid ${({ theme }) => `${theme.colors.mediumGray}80`};
  white-space: nowrap;
`;

const Td = styled.td`
  padding: 10px 12px;
  border-bottom: 1px solid ${({ theme }) => `${theme.colors.mediumGray}30`};
  vertical-align: top;
`;

const ModelName = styled.div`
  font-size: 12px;
  color: ${({ theme }) => theme.colors.foreground};
  white-space: nowrap;
  max-width: 280px;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const ModelId = styled.div`
  font-size: 10px;
  color: ${({ theme }) => `${theme.colors.lightGray}80`};
  margin-top: 2px;
  white-space: nowrap;
  max-width: 280px;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const NodeHeader = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2px;
`;

const NodeName = styled.div`
  font-size: 11px;
  color: ${({ theme }) => theme.colors.foreground};
`;

const DiskInfo = styled.div`
  font-size: 10px;
  color: ${({ theme }) => `${theme.colors.lightGray}80`};
`;

// Cell status styling
const CellWrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 120px;
`;

const CellBadge = styled.div<{ $kind: CellStatus['kind'] }>`
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 10px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: ${({ theme, $kind }) => {
    switch ($kind) {
      case 'completed': return '#4ade80';
      case 'downloading': return theme.colors.yellow;
      case 'pending': return theme.colors.lightGray;
      case 'failed': return theme.colors.destructive;
      default: return `${theme.colors.lightGray}50`;
    }
  }};
`;

const ProgressBar = styled.div`
  height: 3px;
  border-radius: 2px;
  background: ${({ theme }) => `${theme.colors.mediumGray}60`};
  overflow: hidden;
`;

const ProgressFill = styled.div<{ $pct: number }>`
  height: 100%;
  border-radius: 2px;
  background: ${({ theme }) => theme.colors.yellow};
  width: ${({ $pct }) => `${Math.min($pct, 100)}%`};
  transition: width 300ms ease;
`;

const CellStats = styled.div`
  font-size: 10px;
  color: ${({ theme }) => `${theme.colors.lightGray}80`};
`;

const CellAction = styled.button`
  background: transparent;
  border: 1px solid ${({ theme }) => `${theme.colors.mediumGray}60`};
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 3px 8px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 10px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  cursor: pointer;
  transition: ${({ theme }) => theme.transitions.fast};
  color: ${({ theme }) => theme.colors.lightGray};

  &:hover {
    border-color: ${({ theme }) => theme.colors.yellow};
    color: ${({ theme }) => theme.colors.yellow};
  }
`;

const DeleteAction = styled(CellAction)`
  color: ${({ theme }) => theme.colors.destructive};
  border-color: ${({ theme }) => `${theme.colors.destructive}50`};

  &:hover {
    border-color: ${({ theme }) => theme.colors.destructive};
    color: ${({ theme }) => theme.colors.destructive};
  }
`;

const EmptyState = styled.div`
  border: 1px solid ${({ theme }) => `${theme.colors.mediumGray}50`};
  border-radius: ${({ theme }) => theme.radius.sm};
  background: ${({ theme }) => `${theme.colors.black}80`};
  padding: 48px 24px;
  text-align: center;
`;

const EmptyTitle = styled.div`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  color: ${({ theme }) => theme.colors.lightGray};
  margin-bottom: 8px;
`;

const EmptyHint = styled.div`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 11px;
  color: ${({ theme }) => `${theme.colors.lightGray}60`};
`;

// ─── Cell component ───────────────────────────────────────────────────────────

interface CellProps {
  cell: CellStatus;
  modelRow: ModelRow;
  nodeId: string;
  onRefresh: () => void;
}

const Cell: React.FC<CellProps> = ({ cell, modelRow, nodeId, onRefresh }) => {
  const handleDelete = async () => {
    if (!confirm(`Delete ${modelRow.modelId} from this node?`)) return;
    try {
      await deleteDownload(nodeId, modelRow.modelId);
      onRefresh();
    } catch (err) {
      console.error('Failed to delete download:', err);
    }
  };

  const handleDownload = async () => {
    if (!modelRow.shardMetadata) return;
    try {
      await startDownloadForNode(nodeId, modelRow.shardMetadata);
    } catch (err) {
      console.error('Failed to start download:', err);
    }
  };

  switch (cell.kind) {
    case 'completed':
      return (
        <CellWrapper>
          <CellBadge $kind="completed">✓ Complete</CellBadge>
          {cell.totalBytes > 0 && (
            <CellStats>{formatBytes(cell.totalBytes)}</CellStats>
          )}
          <DeleteAction type="button" onClick={() => void handleDelete()}>
            Delete
          </DeleteAction>
        </CellWrapper>
      );

    case 'downloading': {
      const pct = Math.round(cell.percentage ?? 0);
      return (
        <CellWrapper>
          <CellBadge $kind="downloading">⬇ {pct}%</CellBadge>
          <ProgressBar>
            <ProgressFill $pct={pct} />
          </ProgressBar>
          <CellStats>
            {formatBytes(cell.downloadedBytes)} / {formatBytes(cell.totalBytes)}
            {cell.speed > 0 && <> · {formatSpeed(cell.speed)}</>}
            {cell.etaMs > 0 && <> · {formatEta(cell.etaMs)}</>}
          </CellStats>
        </CellWrapper>
      );
    }

    case 'pending':
      return (
        <CellWrapper>
          <CellBadge $kind="pending">⏳ Pending</CellBadge>
        </CellWrapper>
      );

    case 'failed':
      return (
        <CellWrapper>
          <CellBadge $kind="failed">✗ Failed</CellBadge>
          {modelRow.shardMetadata && (
            <CellAction type="button" onClick={() => void handleDownload()}>
              Retry
            </CellAction>
          )}
        </CellWrapper>
      );

    case 'not_present':
    default:
      return (
        <CellWrapper>
          <CellBadge $kind="not_present">—</CellBadge>
          {modelRow.shardMetadata && (
            <CellAction type="button" onClick={() => void handleDownload()}>
              Download
            </CellAction>
          )}
        </CellWrapper>
      );
  }
};

// ─── Page component ───────────────────────────────────────────────────────────

const DownloadsPage: React.FC = () => {
  const { downloads, nodeDisk, topology, nodeIdentities } = useTopologyStore();

  const getNodeLabel = (nodeId: string): string => {
    const name = nodeIdentities[nodeId]?.friendlyName;
    if (name) return name;
    const node = topology?.nodes[nodeId];
    return node?.friendly_name ?? node?.system_info?.model_id ?? nodeId.slice(0, 8);
  };

  const { modelRows, nodeColumns } = useMemo(
    () => buildDownloadGrid(downloads, nodeDisk, getNodeLabel),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [downloads, nodeDisk, topology, nodeIdentities],
  );

  // Dummy refresh — topology store already polls every 2s
  const handleRefresh = () => {
    // No-op; data refreshes automatically via topologyStore polling
  };

  return (
    <Page>
      <PageTitle>Model Downloads</PageTitle>

      {modelRows.length === 0 ? (
        <EmptyState>
          <EmptyTitle>No downloads</EmptyTitle>
          <EmptyHint>Launch a model from the chat page to start downloading.</EmptyHint>
        </EmptyState>
      ) : (
        <TableWrapper>
          <Grid>
            <thead>
              <tr>
                <Th>Model</Th>
                {nodeColumns.map((col: NodeColumn) => (
                  <Th key={col.nodeId}>
                    <NodeHeader>
                      <NodeName>{col.label}</NodeName>
                      {col.diskAvailable !== undefined && col.diskTotal !== undefined && (
                        <DiskInfo>
                          {formatBytes(col.diskTotal - col.diskAvailable)} /{' '}
                          {formatBytes(col.diskTotal)} used
                        </DiskInfo>
                      )}
                    </NodeHeader>
                  </Th>
                ))}
              </tr>
            </thead>
            <tbody>
              {modelRows.map((row: ModelRow) => (
                <tr key={row.modelId}>
                  <Td>
                    <ModelName>{row.prettyName ?? row.modelId}</ModelName>
                    {row.prettyName && <ModelId>{row.modelId}</ModelId>}
                    {row.modelCard && (
                      <ModelId>
                        {row.modelCard.family}
                        {row.modelCard.quantization
                          ? ` · ${row.modelCard.quantization}`
                          : ''}
                        {row.modelCard.storageSize > 0
                          ? ` · ${formatBytes(row.modelCard.storageSize)}`
                          : ''}
                      </ModelId>
                    )}
                  </Td>
                  {nodeColumns.map((col: NodeColumn) => {
                    const cell: CellStatus = row.cells[col.nodeId] ?? { kind: 'not_present' };
                    return (
                      <Td key={col.nodeId}>
                        <Cell
                          cell={cell}
                          modelRow={row}
                          nodeId={col.nodeId}
                          onRefresh={handleRefresh}
                        />
                      </Td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </Grid>
        </TableWrapper>
      )}
    </Page>
  );
};

export default DownloadsPage;
