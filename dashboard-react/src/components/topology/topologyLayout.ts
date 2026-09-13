import type { NodeInfo } from '../../types/topology';

const MAXIMUM_TOPOLOGY_HEIGHT_RATIO = 0.7;
const CENTER_ALIGNMENT_TOLERANCE = 1;
// Selected nodes extend from the selection ring at -49 through the action
// rail at 152. Budgeting the interactive state prevents the rail from
// pushing the visible topology beyond the same responsive height envelope.
const SELECTED_NODE_VERTICAL_FOOTPRINT = 201;

/** Stable center point for one node in the topology canvas. */
export interface TopologyNodePosition {
  id: string;
  x: number;
  y: number;
}

/** One bidirectional visual route in the complete topology mesh. */
export interface CompleteEdgePair {
  source: string;
  target: string;
}

/** Side of a node used to place its hardware identity badge. */
export type HardwareBadgeSide = 'left' | 'right';

/**
 * Places a hardware badge on the outside edge of the topology orbit.
 * Nodes aligned with the canvas center retain the established left placement;
 * the tolerance prevents tiny trigonometric rounding errors from flipping them.
 */
export function hardwareBadgeSideForPosition(
  positionX: number,
  canvasWidth: number,
): HardwareBadgeSide {
  return positionX > canvasWidth / 2 + CENTER_ALIGNMENT_TOLERANCE ? 'right' : 'left';
}

/**
 * Orders node groups for SVG painting while preserving their orbit positions.
 * The active node is painted last so its hover surface and action rail remain
 * above neighboring transparent hit regions in dense layouts.
 */
export function orderTopologyPositionsForPainting(
  positions: readonly TopologyNodePosition[],
  activeNodeId: string | null,
): TopologyNodePosition[] {
  if (!activeNodeId) return [...positions];
  const activePosition = positions.find(({ id }) => id === activeNodeId);
  if (!activePosition) return [...positions];
  return [...positions.filter(({ id }) => id !== activeNodeId), activePosition];
}

/**
 * Builds the complete undirected mesh used by the operator topology view.
 * Every unordered node pair appears exactly once. The canvas presents the
 * cluster as one fabric rather than mirroring transient transport adjacency,
 * and the renderer adds arrows in both directions for every visual route.
 */
export function buildCompleteEdgePairs(nodeIds: readonly string[]): CompleteEdgePair[] {
  const orderedNodeIds = [...nodeIds].sort((left, right) => left.localeCompare(right));
  const pairs: CompleteEdgePair[] = [];
  for (let sourceIndex = 0; sourceIndex < orderedNodeIds.length; sourceIndex += 1) {
    const source = orderedNodeIds[sourceIndex];
    if (!source) continue;
    for (let targetIndex = sourceIndex + 1; targetIndex < orderedNodeIds.length; targetIndex += 1) {
      const target = orderedNodeIds[targetIndex];
      if (target) pairs.push({ source, target });
    }
  }
  return pairs;
}

/**
 * Places current nodes on the same stable symmetric orbit used by skulk-app.
 * Friendly-name sorting prevents telemetry object order from rotating the
 * fabric between refreshes.
 */
export function computeTopologyPositions(
  nodes: Readonly<Record<string, NodeInfo>>,
  width: number,
  height: number,
  nodeScale: number,
): TopologyNodePosition[] {
  if (width <= 0 || height <= 0) return [];

  const orderedNodeIds = Object.keys(nodes).sort((leftId, rightId) => {
    const leftName = nodes[leftId]?.friendly_name ?? leftId;
    const rightName = nodes[rightId]?.friendly_name ?? rightId;
    return leftName.localeCompare(rightName) || leftId.localeCompare(rightId);
  });
  if (orderedNodeIds.length === 0) return [];

  const centerX = width / 2;
  const centerY = height / 2 - 4 * nodeScale;
  if (orderedNodeIds.length === 1) {
    const onlyNodeId = orderedNodeIds[0];
    return onlyNodeId ? [{ id: onlyNodeId, x: centerX, y: centerY }] : [];
  }

  // skulk-app deliberately uses one radius rather than separate horizontal and
  // vertical radii. The extra bounds account for this dashboard's retained
  // hardware badge and operator actions without changing the circular orbit.
  // The outward-facing vendor badge ends seven units clear of the widest
  // (selected) node ring. Reserve its full reach on both canvas edges so small
  // canvases never clip the badge.
  const horizontalLimit = width / 2 - 108 * nodeScale;
  const verticalLimit = height / 2 - 104 * nodeScale;
  const footprintLimit = Math.max(
    0,
    (height * MAXIMUM_TOPOLOGY_HEIGHT_RATIO - SELECTED_NODE_VERTICAL_FOOTPRINT * nodeScale) /
      2,
  );
  const orbitRadius = Math.max(
    0,
    Math.min(
      width * 0.29,
      height * 0.34,
      horizontalLimit,
      verticalLimit,
      footprintLimit,
    ),
  );
  return orderedNodeIds.map((id, index) => {
    const angle = (index / orderedNodeIds.length) * Math.PI * 2 - Math.PI / 2;
    return {
      id,
      x: centerX + orbitRadius * Math.cos(angle),
      y: centerY + orbitRadius * Math.sin(angle),
    };
  });
}

/** Returns a finite zero-to-one telemetry ratio. */
export function clampTelemetryRatio(value: number): number {
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
}

/** Orbit radius of capability satellites around their host, in node units. */
export const SATELLITE_ORBIT_RADIUS = 72;
/** Satellite disc radius, in node units. */
export const SATELLITE_RADIUS = 10;
/** Satellites drawn before the remainder collapses into one overflow glyph. */
export const MAX_VISIBLE_SATELLITES = 6;

/** Node-local center of one satellite (before the host group's scale). */
export interface SatellitePosition {
  index: number;
  x: number;
  y: number;
}

/** Placement of a host's satellites plus how many did not fit. */
export interface SatelliteLayout {
  positions: SatellitePosition[];
  /** Satellites beyond the visible cap, summarized by the overflow glyph. */
  overflow: number;
  /** Where the overflow glyph sits, when there is one. */
  overflowPosition: SatellitePosition | null;
}

/**
 * Places capability satellites on an arc above their host node. The arc
 * stays in the upper half so it never meets the name, telemetry, or action
 * rail drawn below the node, and it leans away from the hardware badge so
 * the badge and the first satellite do not crowd each other. At most
 * `MAX_VISIBLE_SATELLITES` are placed; the rest share one overflow slot at
 * the end of the arc.
 */
export function computeSatellitePositions(
  count: number,
  hardwareBadgeSide: HardwareBadgeSide,
): SatelliteLayout {
  if (count <= 0) return { positions: [], overflow: 0, overflowPosition: null };
  const visible = Math.min(count, MAX_VISIBLE_SATELLITES);
  const overflow = count - visible;
  const slots = visible + (overflow > 0 ? 1 : 0);
  // Degrees, SVG orientation (negative y is up). The badge sits on the
  // outer side at y = -17..17, so the arc's near end stops well above it
  // while its far end reaches further down on the badge-free side; a lone
  // satellite therefore sits just off the vertical, away from the badge.
  const [startDegrees, endDegrees] = hardwareBadgeSide === 'left' ? [-140, -20] : [-40, -160];
  const angleFor = (slot: number): number => {
    if (slots === 1) return (startDegrees + endDegrees) / 2;
    return startDegrees + ((endDegrees - startDegrees) * slot) / (slots - 1);
  };
  const place = (slot: number): SatellitePosition => {
    const radians = (angleFor(slot) * Math.PI) / 180;
    return {
      index: slot,
      x: SATELLITE_ORBIT_RADIUS * Math.cos(radians),
      y: SATELLITE_ORBIT_RADIUS * Math.sin(radians),
    };
  };
  const positions = Array.from({ length: visible }, (_, slot) => place(slot));
  return {
    positions,
    overflow,
    overflowPosition: overflow > 0 ? place(visible) : null,
  };
}
