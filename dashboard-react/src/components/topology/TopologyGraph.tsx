import { useMemo } from 'react';
import styled from 'styled-components';
import type { TopologyData, TopologyEdge } from '../../types/topology';
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

/* ---- component ---- */

export function TopologyGraph({ data }: TopologyGraphProps) {
  const [svgRef, { width, height }] = useResizeObserver<SVGSVGElement>();

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
        <svg ref={svgRef} style={{ width: '100%', height: '100%', background: 'transparent' }} />
      </Container>
    );
  }

  return (
    <Container>
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
              edges={data.edges}
              allNodes={data.nodes}
            />
          ))}
        </g>
      </svg>
    </Container>
  );
}
