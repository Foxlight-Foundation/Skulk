import { describe, expect, it } from 'vitest';
import type { NodeInfo } from '../../types/topology';
import {
  buildCompleteEdgePairs,
  clampTelemetryRatio,
  computeTopologyPositions,
  hardwareBadgeSideForPosition,
  orderTopologyPositionsForPainting,
} from './topologyLayout';

function node(name: string): NodeInfo {
  return { friendly_name: name, last_mactop_update: 0 };
}

describe('topology layout', () => {
  it('sorts nodes by friendly name before placing them on the orbit', () => {
    const positions = computeTopologyPositions(
      { 'node-z': node('kite5'), 'node-a': node('kite1'), 'node-m': node('kite2') },
      900,
      700,
      1,
    );

    expect(positions.map(({ id }) => id)).toEqual(['node-a', 'node-m', 'node-z']);
    expect(positions[0]?.x).toBeCloseTo(450);
    expect(positions[0]?.y).toBeLessThan(342);
  });

  it('places every node at the same distance from the circular orbit center', () => {
    const width = 900;
    const height = 700;
    const nodeScale = 1;
    const centerX = width / 2;
    const centerY = height / 2 - 4 * nodeScale;
    const positions = computeTopologyPositions(
      {
        'node-a': node('kite1'),
        'node-b': node('kite2'),
        'node-c': node('kite3'),
        'node-d': node('kite4'),
      },
      width,
      height,
      nodeScale,
    );
    const radii = positions.map(({ x, y }) => Math.hypot(x - centerX, y - centerY));

    expect(radii).toHaveLength(4);
    for (const radius of radii) expect(radius).toBeCloseTo(radii[0] ?? 0);
  });

  it('expands responsively while keeping the selected-node footprint within 70%', () => {
    const nodeScale = 1.12;
    const nodes = {
      'node-a': node('kite1'),
      'node-b': node('kite2'),
    };
    const compactPositions = computeTopologyPositions(nodes, 940, 650, nodeScale);
    const expandedPositions = computeTopologyPositions(nodes, 1600, 1200, nodeScale);
    const compactRadius = Math.hypot(
      (compactPositions[0]?.x ?? 470) - 470,
      (compactPositions[0]?.y ?? 320.52) - 320.52,
    );
    const expandedRadius = Math.hypot(
      (expandedPositions[0]?.x ?? 800) - 800,
      (expandedPositions[0]?.y ?? 595.52) - 595.52,
    );

    expect(compactRadius).toBeCloseTo(114.94);
    expect(expandedRadius).toBeCloseTo(307.44);
    expect(expandedRadius).toBeGreaterThan(compactRadius);
    expect(compactRadius * 2 + 201 * nodeScale).toBeCloseTo(650 * 0.7);
    expect(expandedRadius * 2 + 201 * nodeScale).toBeCloseTo(1200 * 0.7);
  });

  it('places three-node badges outward while keeping the centered node on the left', () => {
    const width = 390;
    const positions = computeTopologyPositions(
      {
        'node-top': node('kite3'),
        'node-right': node('kite5'),
        'node-left': node('kite6'),
      },
      width,
      844,
      0.86,
    );

    expect(
      Object.fromEntries(
        positions.map(({ id, x }) => [id, hardwareBadgeSideForPosition(x, width)]),
      ),
    ).toEqual({
      'node-top': 'left',
      'node-right': 'right',
      'node-left': 'left',
    });
  });

  it('reserves the full outward-facing badge width on a phone canvas', () => {
    const nodeScale = 0.86;
    const positions = computeTopologyPositions(
      {
        'node-a': node('kite1'),
        'node-b': node('kite2'),
        'node-c': node('kite3'),
        'node-d': node('kite4'),
      },
      390,
      844,
      nodeScale,
    );
    const leftmostNode = positions.reduce((leftmost, position) =>
      position.x < leftmost.x ? position : leftmost,
    );
    const rightmostNode = positions.reduce((rightmost, position) =>
      position.x > rightmost.x ? position : rightmost,
    );

    expect(leftmostNode.x - 104 * nodeScale).toBeGreaterThanOrEqual(0);
    expect(rightmostNode.x + 104 * nodeScale).toBeLessThanOrEqual(390);
  });

  it('keeps a single node centered', () => {
    expect(computeTopologyPositions({ only: node('kite1') }, 800, 600, 1)).toEqual([
      { id: 'only', x: 400, y: 296 },
    ]);
  });

  it('connects every unordered node pair exactly once', () => {
    expect(buildCompleteEdgePairs(['c', 'a', 'd', 'b'])).toEqual([
      { source: 'a', target: 'b' },
      { source: 'a', target: 'c' },
      { source: 'a', target: 'd' },
      { source: 'b', target: 'c' },
      { source: 'b', target: 'd' },
      { source: 'c', target: 'd' },
    ]);
  });

  it('paints the interacting node last without moving its orbit position', () => {
    const positions = [
      { id: 'node-a', x: 10, y: 20 },
      { id: 'node-b', x: 30, y: 40 },
      { id: 'node-c', x: 50, y: 60 },
    ];

    expect(orderTopologyPositionsForPainting(positions, 'node-a')).toEqual([
      positions[1],
      positions[2],
      positions[0],
    ]);
    expect(positions.map(({ id }) => id)).toEqual(['node-a', 'node-b', 'node-c']);
  });

  it('does not create routes for fewer than two nodes', () => {
    expect(buildCompleteEdgePairs([])).toEqual([]);
    expect(buildCompleteEdgePairs(['only'])).toEqual([]);
  });

  it('clamps invalid telemetry ratios', () => {
    expect(clampTelemetryRatio(-0.2)).toBe(0);
    expect(clampTelemetryRatio(0.42)).toBe(0.42);
    expect(clampTelemetryRatio(1.4)).toBe(1);
    expect(clampTelemetryRatio(Number.NaN)).toBe(0);
  });
});
