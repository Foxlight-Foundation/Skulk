import { useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';
import type { TraceEventLike } from '../../types/observabilityEvents';

/**
 * Inline trace waterfall renderer for the observability panel.
 *
 * Replaces the prior Perfetto popup integration: traces stay in-cluster, the
 * UI is reachable without popups, and the same component renders both saved
 * traces and (later, in #120) live cluster timeline data.
 *
 * Renders a horizontal lane per `laneKey` and one bar per event, positioned
 * by `startUs` and sized by `durationUs`. Bars are colored by `category`.
 * Click a bar to surface event detail via `onSelect`; the host component
 * renders the detail panel itself so this stays a pure renderer.
 *
 * Implementation chooses SVG over canvas: at the trace sizes Skulk produces
 * (50–200 events × ≤7 lanes) the SVG node count is well within the comfort
 * zone, and we get free hit-targets, focus rings, and `title` tooltips. If
 * Phase 3 / live data drives counts past ~2000 events we revisit canvas.
 */
export interface TraceWaterfallProps {
  events: readonly TraceEventLike[];
  /** Currently selected event id; bar gets a highlighted outline. */
  selectedId?: string | null;
  /** Click / keyboard activation hands the chosen event back to the parent. */
  onSelect?: (event: TraceEventLike | null) => void;
}

const Wrap = styled.div`
  position: relative;
  width: 100%;
  background: ${({ theme }) => theme.colors.surfaceSunken};
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: 8px 8px 12px;
  box-sizing: border-box;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
  overflow: hidden;
`;

const EmptyState = styled.div`
  padding: 16px 8px;
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const Legend = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  padding: 6px 4px 8px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const LegendItem = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 6px;
`;

const Swatch = styled.span<{ $color: string }>`
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
  background: ${({ $color }) => $color};
  border: 1px solid rgba(0, 0, 0, 0.2);
`;

const TIME_AXIS_HEIGHT = 18;
const LANE_HEIGHT = 22;
const LANE_LABEL_WIDTH = 64;
const BAR_VERTICAL_PADDING = 4;
/** Pixel floor so even zero-duration events stay clickable. */
const MIN_BAR_PX = 2;

/**
 * Stable color-per-category map. Hand-tuned medium-sat / medium-light HSL hexes
 * so the same swatches read on both light and dark themes; bars carry a thin
 * outline (`barStroke`) drawn from the theme so they don't fade into the
 * surface fill on either palette.
 */
const CATEGORY_COLORS: Record<string, string> = {
  compute: '#e6b34a', // gold
  decode: '#4ec48c', // green
  comms: '#5e8de8', // blue
  sync: '#e89358', // orange
  lifecycle: '#b88de0', // purple
  tooling: '#5cc4c4', // cyan
  async: '#e082b5', // pink
};

const FALLBACK_COLOR = '#9aa0a6';

function colorForCategory(category: string): string {
  return CATEGORY_COLORS[category] ?? FALLBACK_COLOR;
}

function formatDuration(microseconds: number): string {
  if (microseconds < 1_000) return `${microseconds.toFixed(0)}us`;
  if (microseconds < 1_000_000) return `${(microseconds / 1_000).toFixed(2)}ms`;
  return `${(microseconds / 1_000_000).toFixed(2)}s`;
}

interface ResolvedLane {
  key: string;
  label: string;
  events: TraceEventLike[];
}

export function TraceWaterfall({ events, selectedId, onSelect }: TraceWaterfallProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  // Live pixel width of the lane area; recomputed on resize via ResizeObserver
  // so the waterfall stays responsive across panel resizes without re-rendering
  // the whole tree on every pointer move.
  const [containerWidth, setContainerWidth] = useState<number>(0);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      setContainerWidth(entry.contentRect.width);
    });
    observer.observe(node);
    setContainerWidth(node.getBoundingClientRect().width);
    return () => observer.disconnect();
  }, []);

  const { lanes, t0, t1, presentCategories } = useMemo(() => {
    if (events.length === 0) {
      return { lanes: [] as ResolvedLane[], t0: 0, t1: 0, presentCategories: [] as string[] };
    }
    let minStart = Infinity;
    let maxEnd = -Infinity;
    const laneOrder: string[] = [];
    const laneIndex = new Map<string, ResolvedLane>();
    const categories = new Set<string>();
    for (const event of events) {
      if (event.startUs < minStart) minStart = event.startUs;
      const end = event.startUs + Math.max(0, event.durationUs);
      if (end > maxEnd) maxEnd = end;
      categories.add(event.category);
      let lane = laneIndex.get(event.laneKey);
      if (!lane) {
        lane = { key: event.laneKey, label: event.laneLabel, events: [] };
        laneIndex.set(event.laneKey, lane);
        laneOrder.push(event.laneKey);
      }
      lane.events.push(event);
    }
    const orderedLanes = laneOrder.map((key) => laneIndex.get(key)!);
    return {
      lanes: orderedLanes,
      t0: minStart,
      t1: maxEnd,
      presentCategories: [...categories].sort(),
    };
  }, [events]);

  const innerWidth = Math.max(0, containerWidth - LANE_LABEL_WIDTH - 12 /* gutters */);
  const totalDurationUs = Math.max(1, t1 - t0);
  const pxPerUs = innerWidth / totalDurationUs;
  const svgHeight = TIME_AXIS_HEIGHT + lanes.length * LANE_HEIGHT;

  if (events.length === 0) {
    return (
      <Wrap ref={containerRef}>
        <EmptyState>No events in this trace.</EmptyState>
      </Wrap>
    );
  }

  return (
    <Wrap ref={containerRef}>
      <Legend>
        {presentCategories.map((category) => (
          <LegendItem key={category}>
            <Swatch $color={colorForCategory(category)} />
            {category}
          </LegendItem>
        ))}
      </Legend>
      <svg
        role="img"
        aria-label="Trace waterfall"
        width={containerWidth}
        height={svgHeight}
        viewBox={`0 0 ${Math.max(containerWidth, 1)} ${svgHeight}`}
      >
        <TimeAxis
          width={innerWidth}
          x={LANE_LABEL_WIDTH}
          y={0}
          totalDurationUs={totalDurationUs}
        />
        {lanes.map((lane, laneIdx) => {
          const laneTop = TIME_AXIS_HEIGHT + laneIdx * LANE_HEIGHT;
          return (
            <g key={lane.key}>
              <LaneRow
                width={containerWidth}
                top={laneTop}
                label={lane.label}
              />
              {lane.events.map((event) => {
                const x = LANE_LABEL_WIDTH + (event.startUs - t0) * pxPerUs;
                const w = Math.max(MIN_BAR_PX, event.durationUs * pxPerUs);
                const y = laneTop + BAR_VERTICAL_PADDING;
                const h = LANE_HEIGHT - BAR_VERTICAL_PADDING * 2;
                const isSelected = selectedId != null && event.id === selectedId;
                return (
                  <EventBar
                    key={event.id}
                    x={x}
                    y={y}
                    width={w}
                    height={h}
                    color={colorForCategory(event.category)}
                    selected={isSelected}
                    event={event}
                    onSelect={onSelect}
                  />
                );
              })}
            </g>
          );
        })}
      </svg>
    </Wrap>
  );
}

/* ---- internal pieces ---- */

function TimeAxis({
  width,
  x,
  y,
  totalDurationUs,
}: {
  width: number;
  x: number;
  y: number;
  totalDurationUs: number;
}) {
  // Five evenly-spaced ticks across the lane area. The labels are rendered
  // with `text-anchor="middle"` so they stay centered under the tick mark.
  const tickCount = 5;
  const ticks: { px: number; label: string }[] = [];
  for (let i = 0; i <= tickCount; i += 1) {
    const fraction = i / tickCount;
    ticks.push({
      px: x + width * fraction,
      label: formatDuration(totalDurationUs * fraction),
    });
  }
  return (
    <g>
      {ticks.map((tick) => (
        <g key={tick.px}>
          <line
            x1={tick.px}
            x2={tick.px}
            y1={y}
            y2={y + TIME_AXIS_HEIGHT - 4}
            stroke="currentColor"
            strokeOpacity={0.18}
          />
          <text
            x={tick.px}
            y={y + TIME_AXIS_HEIGHT - 6}
            textAnchor="middle"
            fontSize={10}
            fill="currentColor"
            fillOpacity={0.7}
          >
            {tick.label}
          </text>
        </g>
      ))}
    </g>
  );
}

function LaneRow({ width, top, label }: { width: number; top: number; label: string }) {
  return (
    <g>
      <line
        x1={0}
        x2={width}
        y1={top + LANE_HEIGHT - 0.5}
        y2={top + LANE_HEIGHT - 0.5}
        stroke="currentColor"
        strokeOpacity={0.08}
      />
      <text
        x={4}
        y={top + LANE_HEIGHT / 2 + 4}
        fontSize={11}
        fill="currentColor"
        fillOpacity={0.75}
      >
        {label}
      </text>
    </g>
  );
}

function EventBar({
  x,
  y,
  width,
  height,
  color,
  selected,
  event,
  onSelect,
}: {
  x: number;
  y: number;
  width: number;
  height: number;
  color: string;
  selected: boolean;
  event: TraceEventLike;
  onSelect?: (event: TraceEventLike | null) => void;
}) {
  return (
    <g
      style={{ cursor: onSelect ? 'pointer' : undefined }}
      onClick={onSelect ? () => onSelect(event) : undefined}
      tabIndex={onSelect ? 0 : undefined}
      role={onSelect ? 'button' : undefined}
      aria-label={`${event.name} (${formatDuration(event.durationUs)})`}
      onKeyDown={
        onSelect
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onSelect(event);
              }
            }
          : undefined
      }
    >
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        fill={color}
        fillOpacity={selected ? 1 : 0.85}
        stroke={selected ? 'currentColor' : 'rgba(0, 0, 0, 0.35)'}
        strokeWidth={selected ? 1.5 : 0.5}
        rx={2}
        ry={2}
      >
        <title>{`${event.name} · ${event.category} · ${formatDuration(event.durationUs)}`}</title>
      </rect>
    </g>
  );
}
