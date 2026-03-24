import type { NodeInfo } from '../../types/topology';
import { detectDeviceModel } from '../../types/topology';
import { DeviceIcon } from './DeviceIcon';
import { GpuStatsBar } from './GpuStatsBar';
import { NodeLabel } from './NodeLabel';

export interface ClusterNodeProps {
  nodeId: string;
  nodeInfo: NodeInfo;
  /** Center x of the entire node group */
  x: number;
  /** Center y of the entire node group */
  y: number;
  /** Overall scale factor — controls the icon size; defaults to 1 */
  scale?: number;
}

/**
 * Composite node component: NodeLabel (above) + DeviceIcon (center) + GpuStatsBar (right) + memory stats (below).
 *
 * Layout (all values before `scale` is applied):
 *
 *           [ name ]              ← NodeLabel name
 *    ┌─────────────────┐ ┌───┐
 *    │   DeviceIcon    │ │GPU│   ← icon + stats bar
 *    └─────────────────┘ └───┘
 *      15.4GB/24GB (64%)         ← NodeLabel memory
 */
export function ClusterNode({
  nodeId,
  nodeInfo,
  x,
  y,
  scale = 1,
}: ClusterNodeProps) {
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
  const ramUsed = nodeInfo.macmon_info?.memory?.ram_usage ?? 0;
  const ramTotal = nodeInfo.macmon_info?.memory?.ram_total ?? 0;
  const ramPercent = ramTotal > 0 ? (ramUsed / ramTotal) * 100 : 0;

  // GPU metrics
  const gpuPercent = (nodeInfo.macmon_info?.gpu_usage?.[1] ?? 0) * 100;
  const gpuTemp = nodeInfo.macmon_info?.temp?.gpu_temp_avg ?? NaN;
  const sysPower = nodeInfo.macmon_info?.sys_power ?? null;

  // Display name
  const name = nodeInfo.friendly_name ?? nodeId.slice(-8);

  // Label sizing
  const labelFontSize = 15;
  const nameOffset = 6;
  const memoryOffset = 6;

  // Layout: icon is centered at (0,0), bar hangs off the right, labels center on icon
  const iconLeft = -iconW / 2;
  const iconTop = -iconH / 2;

  return (
    <g transform={`translate(${x}, ${y}) scale(${scale})`}>
      {/* Name & memory labels — centered on the icon */}
      <NodeLabel
        name={name}
        ramUsed={ramUsed}
        ramTotal={ramTotal}
        cx={0}
        fontSize={labelFontSize}
        nameY={iconTop - nameOffset}
        memoryY={iconTop + iconH + memoryOffset}
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
