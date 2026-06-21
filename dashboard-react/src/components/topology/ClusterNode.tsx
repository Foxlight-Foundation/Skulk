import { useTheme } from 'styled-components';
import type { NodeInfo, TopologyEdge } from '../../types/topology';
import { detectDeviceModel } from '../../types/topology';
import { DeviceIcon } from './DeviceIcon';
import { GpuStatsBar } from './GpuStatsBar';
import { NodeLabel } from './NodeLabel';
import type { Theme } from '../../theme';
import { useSkulkTranslation, type SkulkTranslate } from '../../i18n/tolgee';


export interface ClusterNodeProps {
  nodeId: string;
  nodeInfo: NodeInfo;
  /** Center x of the entire node group */
  x: number;
  /** Center y of the entire node group */
  y: number;
  /** Overall scale factor — controls the icon size; defaults to 1 */
  scale?: number;
  /** All edges in the topology (needed for node info tooltip). */
  edges?: TopologyEdge[];
  /** All nodes (needed for interface name resolution). */
  allNodes?: Record<string, NodeInfo>;
  /** Called when the user confirms a node restart. */
  onRestart?: () => void;
  /** Called when the user opens live diagnostics for this node. */
  onInspect?: () => void;
}

function buildDebugContent(
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
        <div style={{ color: theme.colors.gold, fontWeight: 600, marginBottom: 4 }}>
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
  // macOS reports a bare version ("15.3"), so prefix "macOS"; Linux reports a
  // self-naming string ("Ubuntu 26.04 LTS"), so show it as-is rather than
  // mislabeling a non-Mac node "macOS".
  const osBuild = nodeInfo.os_build_version ? ` (${nodeInfo.os_build_version})` : '';
  const os = nodeInfo.os_version
    ? /^\d/.test(nodeInfo.os_version)
      ? `macOS ${nodeInfo.os_version}${osBuild}`
      : `${nodeInfo.os_version}${osBuild}`
    : '';
  // Group outbound connections by target node
  const byTarget = new Map<string, string[]>();
  for (const e of edges) {
    if (e.source !== nodeId) continue;
    const targetName = allNodes[e.target]?.friendly_name ?? e.target.slice(-8);
    const list = byTarget.get(targetName) ?? [];

    if (e.sourceRdmaIface && e.sinkRdmaIface) {
      // These are Thunderbolt interfaces that macOS labels as rdma_en*
      // Actual RDMA requires TB5 + rdma_ctl enabled
      const srcRdma = allNodes[e.source]?.rdma_enabled;
      const sinkRdma = allNodes[e.target]?.rdma_enabled;
      const isRealRdma = srcRdma && sinkRdma;
      const label = isRealRdma
        ? `RDMA ${e.sourceRdmaIface} → ${e.sinkRdmaIface}`
        : `TB ${e.sourceRdmaIface} → ${e.sinkRdmaIface}`;
      list.push(label);
    } else if (e.sendBackIp) {
      const iface =
        e.sendBackInterface ??
        allNodes[e.source]?.ip_to_interface?.[e.sendBackIp] ??
        allNodes[e.target]?.ip_to_interface?.[e.sendBackIp];
      list.push(`${e.sendBackIp}${iface ? ` ${iface}` : ''}`);
    }
    byTarget.set(targetName, list);
  }

  const rdmaStatus = nodeInfo.rdma_enabled
    ? (nodeInfo.rdma_interfaces_present === false
        ? t('topology.clusterNode.rdmaEnabledNoHardware', 'Enabled (no HW support)')
        : t('topology.clusterNode.enabled', 'Enabled'))
    : t('topology.clusterNode.disabled', 'Disabled');
  const rdmaColor = nodeInfo.rdma_enabled
    ? (nodeInfo.rdma_interfaces_present === false ? theme.colors.warning : theme.colors.healthy)
    : theme.colors.textMuted;
  const version = nodeInfo.skulk_version && nodeInfo.skulk_version !== 'Unknown'
    ? `v${nodeInfo.skulk_version}${nodeInfo.skulk_commit && nodeInfo.skulk_commit !== 'Unknown' ? ` (${nodeInfo.skulk_commit})` : ''}`
    : '';

  return (
    <div style={{ lineHeight: 1.6 }}>
      <div style={{ color: theme.colors.gold, fontWeight: 600, marginBottom: 4 }}>
        {modelId}{chip ? ` · ${chip}` : ''}
      </div>
      {os && <div style={{ color: theme.colors.textSecondary }}>{os}</div>}
      {version && <div style={{ color: theme.colors.textSecondary }}>{version}</div>}
      <div style={{ color: rdmaColor, marginBottom: 6 }}>
        {t('topology.clusterNode.rdmaStatus', 'RDMA: {status}', { status: rdmaStatus })}
      </div>
      {byTarget.size > 0 && (
        <>
          <div style={{ color: theme.colors.textMuted, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
            {t('topology.clusterNode.connections', 'Connections')}
          </div>
          {Array.from(byTarget.entries()).map(([target, conns]) => (
            <div key={target} style={{ marginBottom: 4 }}>
              <div style={{ color: theme.colors.textSecondary, fontWeight: 500 }}>→ {target}</div>
              {conns.map((c, i) => (
                <div key={i} style={{ paddingLeft: 12, color: c.startsWith('RDMA') ? theme.colors.gold : c.startsWith('TB ') ? theme.colors.info : theme.colors.textSecondary }}>
                  {c}
                </div>
              ))}
            </div>
          ))}
        </>
      )}
    </div>
  );
}

export function ClusterNode({
  nodeId,
  nodeInfo,
  x,
  y,
  scale = 1,
  edges = [],
  allNodes = {},
  onRestart,
  onInspect,
}: ClusterNodeProps) {
  const { t } = useSkulkTranslation();
  const theme = useTheme() as Theme;
  const model = detectDeviceModel(nodeInfo.system_info?.model_id);

  // Icon dimensions (unscaled)
  const iconW = 150;
  const iconH = model === 'macbook-pro' ? 140 : 120;

  // The actual rendered device body is smaller than the icon canvas
  const deviceBodyH = model === 'macbook-pro' ? iconH * 0.88 : iconH * 0.7;

  // GPU stats bar dimensions — match the device body height
  const barW = 36;
  const barH = deviceBodyH;
  const barGap = 6;

  // RAM metrics
  const ramUsed = nodeInfo.mactop_info?.memory?.ram_usage ?? 0;
  const ramTotal = nodeInfo.mactop_info?.memory?.ram_total ?? 0;
  const ramPercent = ramTotal > 0 ? (ramUsed / ramTotal) * 100 : 0;

  // GPU metrics
  const gpuPercent = (nodeInfo.mactop_info?.gpu_usage?.[1] ?? 0) * 100;
  const gpuTemp = nodeInfo.mactop_info?.temp?.gpu_temp_avg ?? NaN;
  const sysPower = nodeInfo.mactop_info?.sys_power ?? null;

  // Display name
  const name = nodeInfo.friendly_name ?? nodeId.slice(-8);
  const statusText = nodeInfo.syncing ? t('topology.clusterNode.syncingClusterState', 'Syncing cluster state') : undefined;

  // Label sizing
  const labelFontSize = 15;
  const nameOffset = 6;
  const memoryOffset = 6;

  // Layout: icon is centered at (0,0), bar hangs off the right, labels center on icon
  const iconLeft = -iconW / 2;
  const iconTop = -iconH / 2;

  const debugContent = buildDebugContent(nodeId, nodeInfo, edges, allNodes, theme, t);

  return (
    <g transform={`translate(${x}, ${y}) scale(${scale})`}>
      {/* Name & memory labels — centered on the icon */}
      <NodeLabel
        name={name}
        ramUsed={ramUsed}
        ramTotal={ramTotal}
        statusText={statusText}
        cx={0}
        fontSize={labelFontSize}
        nameY={iconTop - nameOffset}
        memoryY={iconTop + iconH + memoryOffset}
        debugContent={debugContent}
        onRestart={onRestart}
        onInspect={onInspect}
      />

      {/* Device icon — centered at origin */}
      <g transform={`translate(${iconLeft}, ${iconTop})`}>
        <DeviceIcon
          model={model}
          ramPercent={ramPercent}
          width={iconW}
          height={iconH}
          clipId={`node-${nodeId}`}
        />
      </g>

      {/* GPU stats bar — to the right of the icon */}
      <g transform={`translate(${iconW / 2 + barGap}, ${-(barH / 2)})`}>
        <GpuStatsBar
          gpuPercent={gpuPercent}
          gpuTemp={gpuTemp}
          sysPower={sysPower}
          width={barW}
          height={barH}
        />
      </g>

    </g>
  );
}
