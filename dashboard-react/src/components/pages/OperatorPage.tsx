import { useState, useCallback } from 'react';
import styled from 'styled-components';
import { useGetRawStateQuery, useGetLocalNodeIdQuery, useRestartNodeMutation } from '../../store/endpoints/cluster';
import { addToast } from '../../hooks/useToast';
import { useRemoteAccess } from '../../hooks/useRemoteAccess';
import { copyToClipboard } from '../../utils/clipboard';
import { QrCode } from '../common/QrCode';

/* ── Types ─────────────────────────────────────────────────── */

interface NodeSummary {
  nodeId: string;
  name: string;
  memTotalBytes: number;
  memUsedBytes: number;
  gpuUsage: number | null;
  temp: number | null;
}

/* ── Styled components ─────────────────────────────────────── */

const Page = styled.div`
  padding: 16px;
  max-width: 600px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const SectionTitle = styled.h2`
  margin: 0 0 4px;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: ${({ theme }) => theme.colors.textMuted};
`;

const SummaryRow = styled.div`
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
`;

const Stat = styled.div`
  flex: 1;
  min-width: 100px;
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 4px;
`;

const StatLabel = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  letter-spacing: 1px;
  color: ${({ theme }) => theme.colors.textMuted};
`;

const StatValue = styled.span<{ $ok?: boolean; $warn?: boolean }>`
  font-size: ${({ theme }) => theme.fontSizes.xl};
  font-family: ${({ theme }) => theme.fonts.mono};
  font-weight: 700;
  color: ${({ $ok, $warn, theme }) =>
    $ok ? theme.colors.healthy : $warn ? theme.colors.warning : theme.colors.text};
`;

const NodeCard = styled.div`
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg};
  overflow: hidden;
`;

const NodeCardHeader = styled.div`
  padding: 14px 16px 10px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
`;

const NodeName = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.md};
  font-family: ${({ theme }) => theme.fonts.body};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text};
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const NodeCardBody = styled.div`
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const MetricRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
`;

const MetricLabel = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  letter-spacing: 1px;
  color: ${({ theme }) => theme.colors.textMuted};
`;

const MetricValue = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.text};
`;

const MemBar = styled.div`
  height: 4px;
  background: ${({ theme }) => theme.colors.borderStrong};
  border-radius: 2px;
  overflow: hidden;
`;

const MemFill = styled.div<{ $pct: number }>`
  height: 100%;
  width: ${({ $pct }) => $pct}%;
  background: ${({ $pct, theme }) =>
    $pct > 85 ? theme.colors.error : $pct > 65 ? theme.colors.warning : theme.colors.healthy};
  border-radius: 2px;
  transition: width 0.3s ease;
`;

const RestartButton = styled.button<{ $confirming: boolean; $disabled: boolean }>`
  all: unset;
  cursor: ${({ $disabled }) => ($disabled ? 'not-allowed' : 'pointer')};
  flex-shrink: 0;
  padding: 8px 16px;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid ${({ $confirming, theme }) =>
    $confirming ? theme.colors.error : theme.colors.border};
  background: ${({ $confirming, theme }) =>
    $confirming ? theme.colors.errorBg : 'transparent'};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ $confirming, $disabled, theme }) =>
    $disabled ? theme.colors.textMuted : $confirming ? theme.colors.error : theme.colors.textSecondary};
  transition: all 0.15s;
  white-space: nowrap;
  min-width: 90px;
  text-align: center;

  &:hover:not([disabled]) {
    border-color: ${({ theme }) => theme.colors.error};
    color: ${({ theme }) => theme.colors.error};
  }

  &:active:not([disabled]) {
    opacity: 0.8;
  }
`;

const EmptyState = styled.div`
  padding: 48px 16px;
  text-align: center;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.textMuted};
  text-transform: uppercase;
  letter-spacing: 2px;
`;

const AccessCard = styled.div`
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg};
  overflow: hidden;
`;

const AccessCardBody = styled.div`
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const AccessRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
`;

const AccessLabel = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  letter-spacing: 1px;
  color: ${({ theme }) => theme.colors.textMuted};
  flex-shrink: 0;
`;

const AccessUrl = styled.a`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.text};
  text-decoration: none;
  word-break: break-all;
  &:hover { text-decoration: underline; }
`;

const QRWrap = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 8px 0;
`;

const QRLabel = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.textMuted};
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const CopyButton = styled.button`
  all: unset;
  cursor: pointer;
  padding: 4px 10px;
  border-radius: ${({ theme }) => theme.radii.sm};
  border: 1px solid ${({ theme }) => theme.colors.border};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.textSecondary};
  white-space: nowrap;
  flex-shrink: 0;
  &:hover { border-color: ${({ theme }) => theme.colors.textMuted}; }
`;

/* ── Helpers ───────────────────────────────────────────────── */

function fmtBytes(bytes: number): string {
  const gb = bytes / (1024 ** 3);
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(bytes / (1024 ** 2)).toFixed(0)} MB`;
}

/* ── RemoteAccessCard component ───────────────────────────── */

function RemoteAccessCard() {
  const access = useRemoteAccess();
  const [copied, setCopied] = useState(false);

  const operatorUrl =
    access.status === 'ok' ? access.data.operatorUrl : null;
  const localUrl =
    access.status === 'ok' ? access.data.local.url : null;
  const tailscaleUrl =
    access.status === 'ok' ? access.data.tailscale.url : null;
  const tailscaleRunning =
    access.status === 'ok' ? access.data.tailscale.running : false;

  const handleCopy = useCallback(() => {
    if (!operatorUrl) return;
    copyToClipboard(operatorUrl).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }).catch(() => {
      addToast({ type: 'error', message: 'Failed to copy URL' });
    });
  }, [operatorUrl]);

  if (access.status === 'loading') return null;
  if (access.status === 'error') return null;

  return (
    <AccessCard>
      <AccessCardBody>
        {tailscaleUrl && (
          <AccessRow>
            <AccessLabel>Tailscale</AccessLabel>
            <AccessUrl href={tailscaleUrl} target="_blank" rel="noopener noreferrer">
              {tailscaleUrl}
            </AccessUrl>
          </AccessRow>
        )}
        {localUrl && (
          <AccessRow>
            <AccessLabel>Local</AccessLabel>
            <AccessUrl href={localUrl} target="_blank" rel="noopener noreferrer">
              {localUrl}
            </AccessUrl>
          </AccessRow>
        )}
        {!tailscaleRunning && (
          <AccessRow>
            <AccessLabel>Tailscale</AccessLabel>
            <MetricValue>not running</MetricValue>
          </AccessRow>
        )}
        {operatorUrl && (
          <QRWrap>
            <QrCode value={operatorUrl} alt="Operator panel QR code" size={180} />
            <QRLabel>Scan to open operator panel</QRLabel>
            <CopyButton onClick={handleCopy}>
              {copied ? 'Copied!' : 'Copy URL'}
            </CopyButton>
          </QRWrap>
        )}
      </AccessCardBody>
    </AccessCard>
  );
}

/* ── NodeCard component ────────────────────────────────────── */

interface NodeCardProps {
  node: NodeSummary;
  localNodeId: string | undefined;
}

function NodeRestartCard({ node, localNodeId }: NodeCardProps) {
  const [confirming, setConfirming] = useState(false);
  const [restartNode, { isLoading }] = useRestartNodeMutation();

  const handlePress = useCallback(async () => {
    if (isLoading) return;
    if (!confirming) {
      setConfirming(true);
      // Auto-cancel confirmation after 3 s if user doesn't tap again
      setTimeout(() => setConfirming(false), 3000);
      return;
    }
    setConfirming(false);
    try {
      await restartNode({ nodeId: node.nodeId }).unwrap();
      addToast({ type: 'success', message: `Restart sent to ${node.name}` });
    } catch {
      addToast({ type: 'error', message: `Failed to restart ${node.name}` });
    }
  }, [confirming, isLoading, node.nodeId, node.name, restartNode]);

  const memPct = node.memTotalBytes > 0
    ? Math.round((node.memUsedBytes / node.memTotalBytes) * 100)
    : 0;

  const isLocal = node.nodeId === localNodeId;

  return (
    <NodeCard>
      <NodeCardHeader>
        <NodeName title={node.nodeId}>{node.name}{isLocal ? ' (this node)' : ''}</NodeName>
        <RestartButton
          $confirming={confirming}
          $disabled={isLoading}
          disabled={isLoading}
          onClick={handlePress}
          aria-label={confirming ? 'Tap again to confirm restart' : `Restart ${node.name}`}
          title={confirming ? 'Tap again to confirm' : 'Restart node'}
        >
          {isLoading ? '…' : confirming ? 'Confirm?' : 'Restart'}
        </RestartButton>
      </NodeCardHeader>
      <NodeCardBody>
        {node.memTotalBytes > 0 && (
          <>
            <MetricRow>
              <MetricLabel>Memory</MetricLabel>
              <MetricValue>
                {fmtBytes(node.memUsedBytes)} / {fmtBytes(node.memTotalBytes)} ({memPct}%)
              </MetricValue>
            </MetricRow>
            <MemBar>
              <MemFill $pct={memPct} />
            </MemBar>
          </>
        )}
        {node.gpuUsage !== null && (
          <MetricRow>
            <MetricLabel>GPU</MetricLabel>
            <MetricValue>{node.gpuUsage.toFixed(0)}%</MetricValue>
          </MetricRow>
        )}
        {node.temp !== null && (
          <MetricRow>
            <MetricLabel>Temp</MetricLabel>
            <MetricValue>{node.temp.toFixed(0)} °C</MetricValue>
          </MetricRow>
        )}
      </NodeCardBody>
    </NodeCard>
  );
}

/* ── Page component ────────────────────────────────────────── */

/**
 * Mobile-first operator panel. Shows cluster health at a glance and exposes
 * per-node restart buttons for headless / remote operation over Tailscale.
 */
export function OperatorPage() {
  const { data, isLoading } = useGetRawStateQuery(undefined, {
    pollingInterval: 5000,
  });
  const { data: localNodeId } = useGetLocalNodeIdQuery();

  // Build node summaries from raw state
  const nodes: NodeSummary[] = [];
  if (data?.topology?.nodes) {
    for (const nodeId of data.topology.nodes) {
      if (!nodeId) continue;
      const identity = data.nodeIdentities?.[nodeId];
      const name = identity?.friendlyName ?? nodeId.slice(0, 12);
      const mem = data.nodeMemory?.[nodeId];
      const sys = data.nodeSystem?.[nodeId];

      const memTotalBytes = mem?.ramTotal?.inBytes ?? 0;
      const memAvailBytes = mem?.ramAvailable?.inBytes ?? 0;
      const memUsedBytes = Math.max(memTotalBytes - memAvailBytes, 0);

      nodes.push({
        nodeId,
        name,
        memTotalBytes,
        memUsedBytes,
        gpuUsage: sys?.gpuUsage ?? null,
        temp: sys?.temp ?? null,
      });
    }
  }

  // Cluster-level summary stats
  const instanceCount = data?.instances ? Object.keys(data.instances).length : 0;
  const runnerCount = data?.runners ? Object.keys(data.runners).length : 0;
  const nodeCount = nodes.length;

  if (isLoading && nodes.length === 0) {
    return (
      <Page>
        <EmptyState>Loading cluster state…</EmptyState>
      </Page>
    );
  }

  return (
    <Page>
      <SectionTitle>Cluster</SectionTitle>
      <SummaryRow>
        <Stat>
          <StatLabel>Nodes</StatLabel>
          <StatValue $ok={nodeCount > 0}>{nodeCount}</StatValue>
        </Stat>
        <Stat>
          <StatLabel>Instances</StatLabel>
          <StatValue $ok={instanceCount > 0}>{instanceCount}</StatValue>
        </Stat>
        <Stat>
          <StatLabel>Runners</StatLabel>
          <StatValue>{runnerCount}</StatValue>
        </Stat>
      </SummaryRow>

      <SectionTitle>Remote Access</SectionTitle>
      <RemoteAccessCard />

      <SectionTitle>Nodes</SectionTitle>

      {nodes.length === 0 ? (
        <EmptyState>No nodes visible</EmptyState>
      ) : (
        nodes.map((node) => (
          <NodeRestartCard
            key={node.nodeId}
            node={node}
            localNodeId={localNodeId}
          />
        ))
      )}
    </Page>
  );
}
