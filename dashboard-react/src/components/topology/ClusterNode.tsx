import { useState } from 'react';
import { useTheme } from 'styled-components';
import type { NodeInfo, TopologyEdge } from '../../types/topology';
import { detectDeviceModel, type DeviceModel } from '../../types/topology';
import { formatBytes } from '../../utils/format';
import type { Theme } from '../../theme';
import { useSkulkTranslation, type SkulkTranslate } from '../../i18n/tolgee';
import { HardwareBadge } from './HardwareBadge';
import { TopologyNodeActions } from './TopologyNodeActions';
import { clampTelemetryRatio } from './topologyLayout';

/** Props for one scalable SVG topology node. */
export interface ClusterNodeProps {
  nodeId: string;
  nodeInfo: NodeInfo;
  /** Center x of the entire node group. */
  x: number;
  /** Center y of the entire node group. */
  y: number;
  /** Overall scale factor for the node and its operator controls. */
  scale?: number;
  /** All observed edges, used by the node info tooltip. */
  edges?: TopologyEdge[];
  /** All nodes, used to resolve friendly connection names. */
  allNodes?: Record<string, NodeInfo>;
  /** Whether this node is currently selected. */
  selected?: boolean;
  /** Called when the node itself is selected or deselected. */
  onSelect?: () => void;
  /** Called when the user confirms a node restart. */
  onRestart?: () => void;
  /** Called when the user opens live diagnostics for this node. */
  onInspect?: () => void;
}

const NODE_RADIUS = 31;
const UTILIZATION_RADIUS = 39;
const UTILIZATION_CIRCUMFERENCE = 2 * Math.PI * UTILIZATION_RADIUS;
const INTERACTION_SURFACE = {
  height: 206,
  width: 224,
  x: -112,
  y: -52,
} as const;

function hardwareLabel(model: DeviceModel, nodeInfo: NodeInfo, t: SkulkTranslate): string {
  const explicitModel = nodeInfo.system_info?.model_id?.trim();
  if (explicitModel) return explicitModel;
  if (model === 'amd-strix') return 'AMD Ryzen AI Max';
  if (model === 'nvidia-gpu') return nodeInfo.system_info?.accelerator_name ?? 'NVIDIA GPU';
  return t('topology.clusterNode.unknownHardware', 'Unknown hardware');
}

function healthColor(nodeInfo: NodeInfo, theme: Theme): string {
  if (nodeInfo.syncing) return theme.colors.topologyNodeSyncing;
  if (nodeInfo.node_health?.level === 'error') return theme.colors.topologyNodeDanger;
  if (nodeInfo.node_health?.level === 'warn') return theme.colors.topologyNodeWarning;
  return theme.colors.topologyNodeHealthy;
}

function healthLabel(nodeInfo: NodeInfo, t: SkulkTranslate): string {
  if (nodeInfo.syncing) return t('topology.clusterNode.syncingClusterState', 'Syncing cluster state');
  if (nodeInfo.node_health?.level === 'error') return t('topology.clusterNode.healthError', 'Node problem');
  if (nodeInfo.node_health?.level === 'warn') return t('topology.clusterNode.healthWarning', 'Node warning');
  return t('topology.clusterNode.healthHealthy', 'Healthy node');
}

function buildInfoContent(
  nodeId: string,
  nodeInfo: NodeInfo,
  edges: TopologyEdge[],
  allNodes: Record<string, NodeInfo>,
  theme: Theme,
  t: SkulkTranslate,
): React.ReactNode {
  if (nodeInfo.syncing) {
    return (
      <div style={{ lineHeight: 1.6 }}>
        <div style={{ color: theme.colors.info, fontWeight: 600, marginBottom: 4 }}>
          {t('topology.clusterNode.joiningCluster', 'Joining cluster')}
        </div>
        <div style={{ color: theme.colors.textSecondary }}>
          {t(
            'topology.clusterNode.replayingEventLog',
            'This dashboard is still replaying the cluster event log for the current master session.',
          )}
        </div>
        <div style={{ color: theme.colors.textSecondary }}>
          {t(
            'topology.clusterNode.liveTelemetryAfterReplay',
            'The node will switch to live telemetry once its join events have been applied locally.',
          )}
        </div>
      </div>
    );
  }

  const chip = nodeInfo.system_info?.chip ?? '';
  const modelId = nodeInfo.system_info?.model_id ?? t('common.unknown', 'Unknown');
  const osBuild = nodeInfo.os_build_version ? ` (${nodeInfo.os_build_version})` : '';
  const os = nodeInfo.os_version
    ? /^\d/.test(nodeInfo.os_version)
      ? `macOS ${nodeInfo.os_version}${osBuild}`
      : `${nodeInfo.os_version}${osBuild}`
    : '';
  const connectionsByTarget = new Map<string, string[]>();
  for (const edge of edges) {
    if (edge.source !== nodeId) continue;
    const targetName = allNodes[edge.target]?.friendly_name ?? edge.target.slice(-8);
    const connections = connectionsByTarget.get(targetName) ?? [];
    if (edge.sourceRdmaIface && edge.sinkRdmaIface) {
      const isRdma = allNodes[edge.source]?.rdma_enabled && allNodes[edge.target]?.rdma_enabled;
      connections.push(
        isRdma
          ? `RDMA ${edge.sourceRdmaIface} → ${edge.sinkRdmaIface}`
          : `TB ${edge.sourceRdmaIface} → ${edge.sinkRdmaIface}`,
      );
    } else if (edge.sendBackIp) {
      const networkInterface =
        edge.sendBackInterface ??
        allNodes[edge.source]?.ip_to_interface?.[edge.sendBackIp] ??
        allNodes[edge.target]?.ip_to_interface?.[edge.sendBackIp];
      connections.push(`${edge.sendBackIp}${networkInterface ? ` ${networkInterface}` : ''}`);
    }
    connectionsByTarget.set(targetName, connections);
  }

  const rdmaStatus = nodeInfo.rdma_enabled
    ? nodeInfo.rdma_interfaces_present === false
      ? t('topology.clusterNode.rdmaEnabledNoHardware', 'Enabled (no HW support)')
      : t('topology.clusterNode.enabled', 'Enabled')
    : t('topology.clusterNode.disabled', 'Disabled');
  const rdmaColor = nodeInfo.rdma_enabled
    ? nodeInfo.rdma_interfaces_present === false
      ? theme.colors.warning
      : theme.colors.healthy
    : theme.colors.textMuted;
  const version =
    nodeInfo.skulk_version && nodeInfo.skulk_version !== 'Unknown'
      ? `v${nodeInfo.skulk_version}${
          nodeInfo.skulk_commit && nodeInfo.skulk_commit !== 'Unknown'
            ? ` (${nodeInfo.skulk_commit})`
            : ''
        }`
      : '';

  return (
    <div style={{ lineHeight: 1.6 }}>
      <div style={{ color: theme.colors.live, fontWeight: 600, marginBottom: 4 }}>
        {modelId}
        {chip ? ` · ${chip}` : ''}
      </div>
      {os ? <div style={{ color: theme.colors.textSecondary }}>{os}</div> : null}
      {version ? <div style={{ color: theme.colors.textSecondary }}>{version}</div> : null}
      <div style={{ color: rdmaColor, marginBottom: 6 }}>
        {t('topology.clusterNode.rdmaStatus', 'RDMA: {status}', { status: rdmaStatus })}
      </div>
      {nodeInfo.node_health && nodeInfo.node_health.reasons.length > 0 ? (
        <div style={{ marginBottom: 8 }}>
          {nodeInfo.node_health.reasons.map((reason) => (
            <div key={reason.code} style={{ marginBottom: 6 }}>
              <div style={{ color: healthColor(nodeInfo, theme) }}>{reason.message}</div>
              {reason.remediation ? (
                <div style={{ color: theme.colors.textSecondary }}>
                  {t('topology.nodeHealth.fixPrefix', 'Fix: {remediation}', {
                    remediation: reason.remediation,
                  })}
                </div>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
      {connectionsByTarget.size > 0 ? (
        <>
          <div
            style={{
              color: theme.colors.textMuted,
              letterSpacing: 1,
              marginBottom: 4,
              textTransform: 'uppercase',
            }}
          >
            {t('topology.clusterNode.connections', 'Connections')}
          </div>
          {[...connectionsByTarget.entries()].map(([target, connections]) => (
            <div key={target} style={{ marginBottom: 4 }}>
              <div style={{ color: theme.colors.textSecondary, fontWeight: 500 }}>→ {target}</div>
              {connections.map((connection) => (
                <div
                  key={connection}
                  style={{
                    color: connection.startsWith('RDMA')
                      ? theme.colors.live
                      : connection.startsWith('TB ')
                        ? theme.colors.info
                        : theme.colors.textSecondary,
                    paddingLeft: 12,
                  }}
                >
                  {connection}
                </div>
              ))}
            </div>
          ))}
        </>
      ) : null}
    </div>
  );
}

/**
 * Native-style topology node: memory inside, compute around it, and health on
 * the inner wire. Hardware remains identifiable through a compact SVG badge;
 * hover, focus, or selection reveals the desktop operator action rail.
 */
export function ClusterNode({
  nodeId,
  nodeInfo,
  x,
  y,
  scale = 1,
  edges = [],
  allNodes = {},
  selected = false,
  onSelect,
  onRestart,
  onInspect,
}: ClusterNodeProps) {
  const { t } = useSkulkTranslation();
  const theme = useTheme() as Theme;
  const [interacting, setInteracting] = useState(false);
  const model = detectDeviceModel(
    nodeInfo.system_info?.model_id,
    nodeInfo.system_info?.chip,
    nodeInfo.system_info?.accelerator_vendor,
  );
  const memoryUsed = nodeInfo.mactop_info?.memory?.ram_usage ?? 0;
  const memoryTotal = nodeInfo.mactop_info?.memory?.ram_total ?? 0;
  const memoryRatio = clampTelemetryRatio(memoryTotal > 0 ? memoryUsed / memoryTotal : 0);
  const memoryPercent = Math.round(memoryRatio * 100);
  const memoryFillHeight = memoryRatio * NODE_RADIUS * 2;
  const memoryIsVram = nodeInfo.mactop_info?.memory?.is_vram ?? false;
  const computeRatio = clampTelemetryRatio(nodeInfo.mactop_info?.gpu_usage?.[1] ?? 0);
  const gpuTemperature = nodeInfo.mactop_info?.temp?.gpu_temp_avg;
  const systemPower = nodeInfo.mactop_info?.sys_power;
  const displayName = nodeInfo.friendly_name ?? nodeId.slice(-8);
  const deviceLabel = hardwareLabel(model, nodeInfo, t);
  const statusLabel = healthLabel(nodeInfo, t);
  const clipId = `topology-memory-${nodeId.replace(/[^a-zA-Z0-9_-]/g, '-')}`;
  const showActions = interacting || selected;
  const actionLabel = t(
    'topology.clusterNode.selectAria',
    '{name}, {hardware}, {memoryPercent}% memory, {computePercent}% compute, {status}',
    {
      name: displayName,
      hardware: deviceLabel,
      memoryPercent,
      computePercent: Math.round(computeRatio * 100),
      status: statusLabel,
    },
  );
  const infoContent = buildInfoContent(nodeId, nodeInfo, edges, allNodes, theme, t);

  return (
    <g
      className="topology-node"
      data-node-id={nodeId}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setInteracting(false);
      }}
      onFocus={() => setInteracting(true)}
      onMouseEnter={() => setInteracting(true)}
      onMouseLeave={() => setInteracting(false)}
      transform={`translate(${x}, ${y}) scale(${scale})`}
    >
      <defs>
        <clipPath id={clipId}>
          <circle cx={0} cy={0} r={NODE_RADIUS - 1} />
        </clipPath>
      </defs>

      {/* SVG groups have no painted box of their own. This surface keeps the
          node, labels, whitespace, and action rail inside one hover region. */}
      <rect
        aria-hidden="true"
        data-node-interaction-surface="true"
        fill="transparent"
        height={INTERACTION_SURFACE.height}
        pointerEvents="all"
        width={INTERACTION_SURFACE.width}
        x={INTERACTION_SURFACE.x}
        y={INTERACTION_SURFACE.y}
      />

      <circle
        cx={0}
        cy={0}
        fill="none"
        opacity={selected ? 0.92 : interacting ? 0.52 : 0}
        r={48}
        stroke={theme.colors.topologyNodeSelection}
        strokeDasharray={selected ? '2 5' : undefined}
        strokeWidth={1.5}
      />
      <circle
        cx={0}
        cy={0}
        fill="none"
        r={UTILIZATION_RADIUS}
        stroke={theme.colors.topologyNodeComputeTrack}
        strokeWidth={4}
      />
      {computeRatio > 0 ? (
        <circle
          cx={0}
          cy={0}
          fill="none"
          r={UTILIZATION_RADIUS}
          stroke={theme.colors.topologyNodeCompute}
          strokeDasharray={`${UTILIZATION_CIRCUMFERENCE * computeRatio} ${UTILIZATION_CIRCUMFERENCE}`}
          strokeLinecap="round"
          strokeWidth={4}
          transform="rotate(-90)"
        />
      ) : null}
      <circle cx={0} cy={0} fill={theme.colors.topologyNodeSurface} r={NODE_RADIUS} />
      {memoryFillHeight > 0 ? (
        <rect
          clipPath={`url(#${clipId})`}
          fill={theme.colors.topologyNodeMemory}
          height={memoryFillHeight}
          opacity={0.42}
          width={NODE_RADIUS * 2}
          x={-NODE_RADIUS}
          y={NODE_RADIUS - memoryFillHeight}
        />
      ) : null}
      <circle
        cx={0}
        cy={0}
        fill="none"
        r={NODE_RADIUS}
        stroke={healthColor(nodeInfo, theme)}
        strokeWidth={2.5}
      />
      <text
        dominantBaseline="middle"
        fill={theme.colors.topologyNodeText}
        fontFamily={theme.fonts.body}
        fontSize={13}
        fontWeight={600}
        style={{ fontVariantNumeric: 'tabular-nums' }}
        textAnchor="middle"
        x={0}
        y={1}
      >
        {memoryTotal > 0 ? `${memoryPercent}%` : '—'}
      </text>
      <circle
        cx={25}
        cy={-25}
        fill={healthColor(nodeInfo, theme)}
        r={5.5}
        stroke={theme.colors.topologyNodeDotBorder}
        strokeWidth={2}
      />

      <g aria-label={deviceLabel} role="img" transform="translate(-104, -17)">
        <HardwareBadge model={model} />
      </g>

      <foreignObject height={92} width={92} x={-46} y={-46}>
        <button
          aria-label={actionLabel}
          aria-pressed={selected}
          onClick={onSelect}
          style={{
            appearance: 'none',
            background: 'transparent',
            border: 0,
            borderRadius: '50%',
            cursor: 'pointer',
            height: '92px',
            margin: 0,
            outline: 'none',
            padding: 0,
            width: '92px',
          }}
          type="button"
        />
      </foreignObject>

      <text
        dominantBaseline="middle"
        fill={theme.colors.topologyNodeText}
        fontFamily={theme.fonts.body}
        fontSize={15}
        fontWeight={600}
        letterSpacing={0.15}
        textAnchor="middle"
        x={0}
        y={61}
      >
        {displayName}
      </text>
      <text
        dominantBaseline="middle"
        fill={theme.colors.topologyNodeLabel}
        fontFamily={theme.fonts.body}
        fontSize={12}
        style={{ fontVariantNumeric: 'tabular-nums' }}
        textAnchor="middle"
        x={0}
        y={82}
      >
        {nodeInfo.syncing
          ? t('topology.clusterNode.syncingClusterState', 'Syncing cluster state')
          : `${memoryIsVram ? `${t('nodeLabel.vramPrefix', 'VRAM')} ` : ''}${formatBytes(memoryUsed)} / ${formatBytes(memoryTotal)}`}
      </text>
      {!nodeInfo.syncing && (gpuTemperature !== undefined || systemPower != null) ? (
        <text
          dominantBaseline="middle"
          fill={theme.colors.topologyNodeDetail}
          fontFamily={theme.fonts.body}
          fontSize={11}
          style={{ fontVariantNumeric: 'tabular-nums' }}
          textAnchor="middle"
          x={0}
          y={101}
        >
          {gpuTemperature !== undefined && Number.isFinite(gpuTemperature)
            ? `${Math.round(gpuTemperature)}°C`
            : '—'}
          {' · '}
          {systemPower != null ? `${Math.round(systemPower)}W` : '—'}
        </text>
      ) : null}

      {showActions ? (
        <foreignObject height={40} width={110} x={-55} y={112}>
          <div onClick={(event) => event.stopPropagation()}>
            <TopologyNodeActions
              infoContent={infoContent}
              onInspect={onInspect}
              onRestart={onRestart}
            />
          </div>
        </foreignObject>
      ) : null}
    </g>
  );
}
