import { useCallback, useId, useMemo, useState } from 'react';
import styled, { useTheme } from 'styled-components';
import type { TopologyData } from '../../types/topology';
import { useResizeObserver } from '../../hooks/useResizeObserver';
import type { Theme } from '../../theme';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { ClusterNode } from './ClusterNode';
import {
  buildCompleteEdgePairs,
  computeTopologyPositions,
  hardwareBadgeSideForPosition,
  orderTopologyPositionsForPainting,
  type TopologyNodePosition,
} from './topologyLayout';

/** Props for the responsive cluster topology canvas. */
export interface TopologyGraphProps {
  data: TopologyData;
  /** Called when a node diagnostics inspection is requested. */
  onInspectNode?: (nodeId: string) => void;
}

const Container = styled.div`
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 420px;
`;

function nodeScaleForCanvas(nodeCount: number, width: number, height: number): number {
  const densityScale = nodeCount <= 4 ? 1 : Math.max(0.62, 1 - (nodeCount - 4) * 0.075);
  // Native topology nodes stay finger-legible on phones. Density may still
  // reduce large fabrics, but viewport scaling alone never shrinks a node
  // below 86% of the desktop mark.
  const widthScale = Math.max(0.86, Math.min(1.12, width / 620));
  const heightScale = Math.max(0.86, Math.min(1.12, height / 560));
  return densityScale * Math.min(widthScale, heightScale);
}

/**
 * Scalable topology projection shared across desktop and phone-sized dashboard
 * canvases. Nodes follow skulk-app's stable orbit while a complete animated
 * mesh preserves the dashboard's bidirectional fabric view.
 */
export function TopologyGraph({ data, onInspectNode }: TopologyGraphProps) {
  const { t } = useSkulkTranslation();
  const theme = useTheme() as Theme;
  const graphInstanceId = useId().replace(/:/g, '');
  const arrowheadId = `topology-arrowhead-${graphInstanceId}`;
  const [svgRef, { width, height }] = useResizeObserver<SVGSVGElement>();
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [interactingNodeId, setInteractingNodeId] = useState<string | null>(null);

  const handleRestart = useCallback((nodeId: string) => {
    fetch(`/admin/restart?node_id=${encodeURIComponent(nodeId)}`, { method: 'POST' }).catch(
      (error: unknown) => {
        console.error('Failed to restart node', nodeId, error);
      },
    );
  }, []);

  const nodeCount = Object.keys(data.nodes).length;
  const nodeScale = useMemo(
    () => nodeScaleForCanvas(nodeCount, width, height),
    [height, nodeCount, width],
  );
  const positions = useMemo(
    () => computeTopologyPositions(data.nodes, width, height, nodeScale),
    [data.nodes, height, nodeScale, width],
  );
  const positionsById = useMemo(() => {
    const positionsMap = new Map<string, TopologyNodePosition>();
    for (const position of positions) positionsMap.set(position.id, position);
    return positionsMap;
  }, [positions]);
  const completeEdges = useMemo(
    () => buildCompleteEdgePairs(Object.keys(data.nodes)),
    [data.nodes],
  );
  const effectiveSelectedNodeId =
    selectedNodeId && data.nodes[selectedNodeId] ? selectedNodeId : null;
  const topNodeId =
    interactingNodeId && data.nodes[interactingNodeId]
      ? interactingNodeId
      : effectiveSelectedNodeId;
  const paintOrderedPositions = useMemo(
    () => orderTopologyPositionsForPainting(positions, topNodeId),
    [positions, topNodeId],
  );

  return (
    <Container>
      <svg
        aria-label={t(
          'topology.graphAria.completeMesh',
          'Cluster topology with {nodeCount} nodes and {routeCount} bidirectional routes. Node fill shows memory pressure, the outer arc shows compute utilization, and the inner ring and dot show health.',
          { nodeCount, routeCount: completeEdges.length },
        )}
        onClick={(event) => {
          if (event.target === event.currentTarget) setSelectedNodeId(null);
        }}
        ref={svgRef}
        role="group"
        style={{ background: 'transparent', height: '100%', width: '100%' }}
      >
        <defs>
          <marker
            id={arrowheadId}
            markerHeight="11"
            markerWidth="11"
            orient="auto-start-reverse"
            refX="10"
            refY="5"
            viewBox="0 0 10 10"
          >
            <path
              d="M 0 0 L 10 5 L 0 10"
              fill="none"
              stroke={theme.colors.topologyConnectionLine}
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="1.6"
              vectorEffect="non-scaling-stroke"
            />
          </marker>
          <style>{`
            .topology-link {
              animation: topologyFlow 0.75s linear infinite;
              opacity: 0.77;
              stroke-dasharray: 4 4;
              stroke-width: 1.35px;
            }
            @keyframes topologyFlow {
              from { stroke-dashoffset: 0; }
              to { stroke-dashoffset: -10; }
            }
            @media (prefers-reduced-motion: reduce) {
              .topology-link { animation: none; }
            }
          `}</style>
        </defs>

        <g aria-hidden>
          {completeEdges.map((edge) => {
            const source = positionsById.get(edge.source);
            const target = positionsById.get(edge.target);
            if (!source || !target) return null;
            const deltaX = target.x - source.x;
            const deltaY = target.y - source.y;
            const length = Math.hypot(deltaX, deltaY) || 1;
            const unitX = deltaX / length;
            const unitY = deltaY / length;
            const midpointX = (source.x + target.x) / 2;
            const midpointY = (source.y + target.y) / 2;
            const arrowOffset = 16;
            const carrierLength = 2;
            return (
              <g key={`${edge.source}-${edge.target}`}>
                <line
                  className="topology-link"
                  stroke={theme.colors.topologyConnectionLine}
                  strokeLinecap="round"
                  vectorEffect="non-scaling-stroke"
                  x1={source.x}
                  x2={target.x}
                  y1={source.y}
                  y2={target.y}
                />
                <line
                  markerEnd={`url(#${arrowheadId})`}
                  stroke="none"
                  x1={midpointX - unitX * (arrowOffset + carrierLength)}
                  x2={midpointX - unitX * arrowOffset}
                  y1={midpointY - unitY * (arrowOffset + carrierLength)}
                  y2={midpointY - unitY * arrowOffset}
                />
                <line
                  markerEnd={`url(#${arrowheadId})`}
                  stroke="none"
                  x1={midpointX + unitX * (arrowOffset + carrierLength)}
                  x2={midpointX + unitX * arrowOffset}
                  y1={midpointY + unitY * (arrowOffset + carrierLength)}
                  y2={midpointY + unitY * arrowOffset}
                />
              </g>
            );
          })}
        </g>

        {paintOrderedPositions.map((position) => {
          const nodeInfo = data.nodes[position.id];
          if (!nodeInfo) return null;
          return (
            <ClusterNode
              allNodes={data.nodes}
              edges={data.edges}
              hardwareBadgeSide={hardwareBadgeSideForPosition(position.x, width)}
              key={position.id}
              nodeId={position.id}
              nodeInfo={nodeInfo}
              onInteractionChange={(interacting) =>
                setInteractingNodeId((current) =>
                  interacting ? position.id : current === position.id ? null : current,
                )
              }
              onInspect={() => onInspectNode?.(position.id)}
              onRestart={() => handleRestart(position.id)}
              onSelect={() =>
                setSelectedNodeId((current) => (current === position.id ? null : position.id))
              }
              scale={nodeScale}
              selected={effectiveSelectedNodeId === position.id}
              x={position.x}
              y={position.y}
            />
          );
        })}
      </svg>
    </Container>
  );
}
