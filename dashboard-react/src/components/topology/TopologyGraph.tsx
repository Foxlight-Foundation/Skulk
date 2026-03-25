import { useMemo, useState } from 'react';
import styled, { css } from 'styled-components';
import type { TopologyData, TopologyEdge, NodeInfo } from '../../types/topology';
import { useResizeObserver } from '../../hooks/useResizeObserver';
import { ClusterNode } from './ClusterNode';

export interface TopologyGraphProps {
  data: TopologyData;
}

interface NodePosition {
  id: string;
  x: number;
  y: number;
}

/* ---- edge pair helpers ---- */

function edgePairKey(a: string, b: string): string {
  return a < b ? `${a}::${b}` : `${b}::${a}`;
}

interface EdgePair {
  a: string;
  b: string;
  aToB: boolean;
  bToA: boolean;
}

function buildEdgePairs(edges: TopologyEdge[]): EdgePair[] {
  const map = new Map<string, EdgePair>();
  for (const e of edges) {
    const key = edgePairKey(e.source, e.target);
    const existing = map.get(key);
    if (existing) {
      if (e.source === existing.a) existing.aToB = true;
      else existing.bToA = true;
    } else {
      const aIsSource = e.source < e.target;
      map.set(key, {
        a: aIsSource ? e.source : e.target,
        b: aIsSource ? e.target : e.source,
        aToB: aIsSource,
        bToA: !aIsSource,
      });
    }
  }
  return Array.from(map.values());
}

/* ---- layout ---- */

function computePositions(
  nodeIds: string[],
  width: number,
  height: number,
): NodePosition[] {
  const n = nodeIds.length;
  if (n === 0 || width === 0 || height === 0) return [];

  const cx = width / 2;
  const topPad = 70;
  const bottomPad = 70;
  const cy = topPad + (height - topPad - bottomPad) / 2;

  if (n === 1) {
    return [{ id: nodeIds[0], x: cx, y: cy }];
  }

  const minDim = Math.min(width, height);
  const orbitRadius = Math.min(
    minDim * 0.32,
    Math.max(minDim * 0.18, minDim * (0.2 + n * 0.02)),
  );

  return nodeIds.map((id, i) => {
    const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
    return {
      id,
      x: cx + orbitRadius * Math.cos(angle),
      y: cy + orbitRadius * Math.sin(angle),
    };
  });
}



/* ---- styles ---- */

const Container = styled.div`
  position: relative;
  width: 100%;
  height: 100%;
`;

const DebugToggle = styled.button<{ $active: boolean }>`
  all: unset;
  cursor: pointer;
  position: absolute;
  top: 12px;
  right: 12px;
  z-index: 5;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: 6px;
  border: 1px solid ${({ $active }) => ($active ? 'rgba(255,215,0,0.5)' : 'rgba(80,80,80,0.4)')};
  color: ${({ $active }) => ($active ? '#FFD700' : 'rgba(179,179,179,0.6)')};
  background: ${({ $active }) => ($active ? 'rgba(255,215,0,0.08)' : 'rgba(17,17,17,0.6)')};
  transition: all 0.15s;

  &:hover {
    border-color: rgba(255, 215, 0, 0.5);
    color: #FFD700;
  }
`;

const BugIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M8 2l1.88 1.88M14.12 3.88L16 2M9 7.13v-1a3.003 3.003 0 116 0v1" />
    <path d="M12 20c-3.3 0-6-2.7-6-6v-3a4 4 0 014-4h4a4 4 0 014 4v3c0 3.3-2.7 6-6 6z" />
    <path d="M12 20v-9M6.53 9C4.6 8.8 3 7.1 3 5M6 13H2M3 21c0-2.1 1.7-3.9 3.8-4M20.97 5c0 2.1-1.6 3.8-3.5 4M22 13h-4M17.2 17c2.1.1 3.8 1.9 3.8 4" />
  </svg>
);

/* ---- component ---- */

export function TopologyGraph({ data }: TopologyGraphProps) {
  const [svgRef, { width, height }] = useResizeObserver<SVGSVGElement>();
  const [debug, setDebug] = useState(false);

  const nodeIds = useMemo(() => Object.keys(data.nodes), [data.nodes]);

  const positions = useMemo(
    () => computePositions(nodeIds, width, height),
    [nodeIds, width, height],
  );

  const posById = useMemo(() => {
    const m = new Map<string, NodePosition>();
    for (const p of positions) m.set(p.id, p);
    return m;
  }, [positions]);

  const edgePairs = useMemo(() => buildEdgePairs(data.edges), [data.edges]);

  const nodeScale = useMemo(() => {
    const n = nodeIds.length;
    if (n <= 1) return 1;
    return Math.max(0.6, 1 - (n - 1) * 0.08);
  }, [nodeIds.length]);


  if (width === 0 || height === 0) {
    return (
      <Container>
        <DebugToggle $active={debug} onClick={() => setDebug(!debug)} aria-label="Toggle debug info" title="Debug overlay">
          <BugIcon />
        </DebugToggle>
        <svg ref={svgRef} style={{ width: '100%', height: '100%', background: 'transparent' }} />
      </Container>
    );
  }

  return (
    <Container>
      <DebugToggle $active={debug} onClick={() => setDebug(!debug)} aria-label="Toggle debug info" title="Debug overlay">
        <BugIcon />
      </DebugToggle>
      <svg
        ref={svgRef}
        style={{ width: '100%', height: '100%', background: 'transparent' }}
      >
        <defs>
          <marker
            id="arrowhead"
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth="11"
            markerHeight="11"
            orient="auto-start-reverse"
          >
            <path
              d="M 0 0 L 10 5 L 0 10"
              fill="none"
              stroke="#B3B3B3"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </marker>
          <style>{`
            .topo-link {
              stroke: #b3b3b3;
              stroke-width: 1px;
              stroke-dasharray: 4 4;
              opacity: 0.8;
              animation: topoFlow 0.75s linear infinite;
            }
            @keyframes topoFlow {
              from { stroke-dashoffset: 0; }
              to   { stroke-dashoffset: -10; }
            }
          `}</style>
        </defs>

        {/* Edges */}
        <g>
          {edgePairs.map((pair) => {
            const pA = posById.get(pair.a);
            const pB = posById.get(pair.b);
            if (!pA || !pB) return null;

            const dx = pB.x - pA.x;
            const dy = pB.y - pA.y;
            const len = Math.hypot(dx, dy) || 1;
            const ux = dx / len;
            const uy = dy / len;
            const mx = (pA.x + pB.x) / 2;
            const my = (pA.y + pB.y) / 2;
            const tipOffset = 16;
            const carrier = 2;

            return (
              <g key={`${pair.a}-${pair.b}`}>
                <line
                  x1={pA.x} y1={pA.y}
                  x2={pB.x} y2={pB.y}
                  className="topo-link"
                />
                {pair.aToB && (
                  <line
                    x1={mx - ux * (tipOffset + carrier)}
                    y1={my - uy * (tipOffset + carrier)}
                    x2={mx - ux * tipOffset}
                    y2={my - uy * tipOffset}
                    stroke="none"
                    markerEnd="url(#arrowhead)"
                  />
                )}
                {pair.bToA && (
                  <line
                    x1={mx + ux * (tipOffset + carrier)}
                    y1={my + uy * (tipOffset + carrier)}
                    x2={mx + ux * tipOffset}
                    y2={my + uy * tipOffset}
                    stroke="none"
                    markerEnd="url(#arrowhead)"
                  />
                )}
              </g>
            );
          })}
        </g>

        {/* Nodes */}
        <g>
          {positions.map((pos) => (
            <ClusterNode
              key={pos.id}
              nodeId={pos.id}
              nodeInfo={data.nodes[pos.id]}
              x={pos.x}
              y={pos.y}
              scale={nodeScale}
            />
          ))}
        </g>

        {/* Debug: node metadata below each node */}
        {debug && positions.map((pos) => {
          const info = data.nodes[pos.id];
          if (!info) return null;
          const rdma = info.rdma_enabled ? 'ON' : 'OFF';
          const os = info.os_version ? `macOS ${info.os_version}${info.os_build_version ? ` (${info.os_build_version})` : ''}` : '';
          const baseY = pos.y + 95 * nodeScale;

          return (
            <g key={`debug-${pos.id}`}>
              <text x={pos.x} y={baseY} textAnchor="middle" fontSize="12" fontFamily="SF Mono, Monaco, monospace" fill="rgba(179,179,179,0.8)">
                RDMA:{rdma}
              </text>
              {os && (
                <text x={pos.x} y={baseY + 14} textAnchor="middle" fontSize="12" fontFamily="SF Mono, Monaco, monospace" fill="rgba(179,179,179,0.7)">
                  {os}
                </text>
              )}
            </g>
          );
        })}

        {/* Debug: connection details in viewport quadrants (matching EXO layout) */}
        {debug && (() => {
          const gcx = width / 2;
          const topPad = 70;
          const bottomPad = 70;
          const gcy = topPad + (height - topPad - bottomPad) / 2;
          const fontSize = 12;
          const lineHeight = 16;

          // Compute cluster bounding box from node positions + margin
          const margin = 170;
          const xs = positions.map(p => p.x);
          const ys = positions.map(p => p.y);
          const clusterLeft = Math.min(...xs) - margin;
          const clusterRight = Math.max(...xs) + margin;
          const clusterTop = Math.min(...ys) - margin;
          const clusterBottom = Math.max(...ys) + margin;

          // Directional arrow based on node positions
          function getArrow(fromId: string, toId: string): string {
            const fp = posById.get(fromId);
            const tp = posById.get(toId);
            if (!fp || !tp) return '→';
            const dx = tp.x - fp.x;
            const dy = tp.y - fp.y;
            const ax = Math.abs(dx);
            const ay = Math.abs(dy);
            if (ax > ay * 2) return dx > 0 ? '→' : '←';
            if (ay > ax * 2) return dy > 0 ? '↓' : '↑';
            if (dx > 0 && dy > 0) return '↘';
            if (dx > 0 && dy < 0) return '↗';
            if (dx < 0 && dy > 0) return '↙';
            return '↖';
          }

          // Group edges by pair, collect connections
          type ConnLine = { arrow: string; label: string; isRdma: boolean };
          const quadrants: { topLeft: ConnLine[]; topRight: ConnLine[]; bottomLeft: ConnLine[]; bottomRight: ConnLine[] } = {
            topLeft: [], topRight: [], bottomLeft: [], bottomRight: [],
          };

          const pairSeen = new Map<string, { mx: number; my: number; lines: ConnLine[] }>();
          for (const e of data.edges) {
            const a = e.source < e.target ? e.source : e.target;
            const b = e.source < e.target ? e.target : e.source;
            const key = `${a}|${b}`;

            let entry = pairSeen.get(key);
            if (!entry) {
              const pA = posById.get(a);
              const pB = posById.get(b);
              if (!pA || !pB) continue;
              entry = { mx: (pA.x + pB.x) / 2, my: (pA.y + pB.y) / 2, lines: [] };
              pairSeen.set(key, entry);
            }

            const arrow = getArrow(e.source, e.target);
            if (e.sourceRdmaIface && e.sinkRdmaIface) {
              entry.lines.push({ arrow, label: `RDMA ${e.sourceRdmaIface} → ${e.sinkRdmaIface}`, isRdma: true });
            } else if (e.sendBackIp) {
              const iface = e.sendBackInterface ?? data.nodes[e.source]?.ip_to_interface?.[e.sendBackIp] ?? data.nodes[e.target]?.ip_to_interface?.[e.sendBackIp];
              entry.lines.push({ arrow, label: `${e.sendBackIp}${iface ? ` ${iface}` : ''}`, isRdma: false });
            }
          }

          // Assign each pair to a quadrant based on midpoint
          for (const entry of pairSeen.values()) {
            const isLeft = entry.mx < gcx;
            const isTop = entry.my < gcy;
            const q = isTop ? (isLeft ? 'topLeft' : 'topRight') : (isLeft ? 'bottomLeft' : 'bottomRight');
            quadrants[q].push(...entry.lines);
          }

          return (
            <g>
              {/* Top Left */}
              {quadrants.topLeft.map((line, i) => (
                <text key={`tl-${i}`} x={clusterLeft} y={clusterTop + i * lineHeight} textAnchor="start" dominantBaseline="hanging"
                  fontSize={fontSize} fontFamily="SF Mono, Monaco, monospace"
                  fill={line.isRdma ? 'rgba(255,215,0,0.9)' : 'rgba(255,255,255,0.85)'}>
                  {line.arrow} {line.label}
                </text>
              ))}
              {/* Top Right */}
              {quadrants.topRight.map((line, i) => (
                <text key={`tr-${i}`} x={clusterRight} y={clusterTop + i * lineHeight} textAnchor="end" dominantBaseline="hanging"
                  fontSize={fontSize} fontFamily="SF Mono, Monaco, monospace"
                  fill={line.isRdma ? 'rgba(255,215,0,0.9)' : 'rgba(255,255,255,0.85)'}>
                  {line.arrow} {line.label}
                </text>
              ))}
              {/* Bottom Left */}
              {quadrants.bottomLeft.map((line, i) => (
                <text key={`bl-${i}`} x={clusterLeft} y={clusterBottom - (quadrants.bottomLeft.length - 1 - i) * lineHeight} textAnchor="start" dominantBaseline="auto"
                  fontSize={fontSize} fontFamily="SF Mono, Monaco, monospace"
                  fill={line.isRdma ? 'rgba(255,215,0,0.9)' : 'rgba(255,255,255,0.85)'}>
                  {line.arrow} {line.label}
                </text>
              ))}
              {/* Bottom Right */}
              {quadrants.bottomRight.map((line, i) => (
                <text key={`br-${i}`} x={clusterRight} y={clusterBottom - (quadrants.bottomRight.length - 1 - i) * lineHeight} textAnchor="end" dominantBaseline="auto"
                  fontSize={fontSize} fontFamily="SF Mono, Monaco, monospace"
                  fill={line.isRdma ? 'rgba(255,215,0,0.9)' : 'rgba(255,255,255,0.85)'}>
                  {line.arrow} {line.label}
                </text>
              ))}
            </g>
          );
        })()}
      </svg>
    </Container>
  );
}
