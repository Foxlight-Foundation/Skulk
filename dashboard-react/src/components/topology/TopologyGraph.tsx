/**
 * TopologyGraph
 *
 * D3-based cluster topology visualization.
 * Ported from TopologyGraph.svelte — the D3 render logic is preserved
 * verbatim; only the lifecycle (onMount/onDestroy/reactive) is converted
 * to React hooks.
 */
import React, { useRef, useEffect, useCallback } from 'react';
import * as d3 from 'd3';
import styled from 'styled-components';
import { useTopologyStore } from '../../stores/topologyStore';
import type { NodeInfo } from '../../api/types';

// ─── Styled wrapper ────────────────────────────────────────────────────────────

const SvgEl = styled.svg`
  width: 100%;
  height: 100%;
  display: block;
`;

// ─── Helpers ───────────────────────────────────────────────────────────────────

const APPLE_LOGO_PATH =
  'M788.1 340.9c-5.8 4.5-108.2 62.2-108.2 190.5 0 148.4 130.3 200.9 134.2 202.2-.6 3.2-20.7 71.9-68.7 141.9-42.8 61.6-87.5 123.1-155.5 123.1s-85.5-39.5-164-39.5c-76.5 0-103.7 40.8-165.9 40.8s-105.6-57-155.5-127C46.7 790.7 0 663 0 541.8c0-194.4 126.4-297.5 250.8-297.5 66.1 0 121.2 43.4 162.7 43.4 39.5 0 101.1-46 176.3-46 28.5 0 130.9 2.6 198.3 99.2zm-234-181.5c31.1-36.9 53.1-88.1 53.1-139.3 0-7.1-.6-14.3-1.9-20.1-50.6 1.9-110.8 33.7-147.1 75.8-28.5 32.4-55.1 83.6-55.1 135.5 0 7.8 1.3 15.6 1.9 18.1 3.2.6 8.4 1.3 13.6 1.3 45.4 0 102.5-30.4 135.5-71.3z';
const LOGO_NATIVE_W = 814;
const LOGO_NATIVE_H = 1000;

function formatBytes(bytes: number, decimals = 1): string {
  if (!bytes || bytes === 0) return '0B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + (sizes[i] ?? 'B');
}

function getTemperatureColor(temp: number): string {
  if (isNaN(temp)) return 'rgba(179,179,179,0.8)';
  const coolTemp = 45, midTemp = 57.5, hotTemp = 75;
  const cool = { r: 93, g: 173, b: 226 };
  const mid  = { r: 255, g: 215, b: 0 };
  const hot  = { r: 244, g: 67,  b: 54 };
  let r: number, g: number, b: number;
  if (temp <= coolTemp) { ({ r, g, b } = cool); }
  else if (temp <= midTemp) {
    const t = (temp - coolTemp) / (midTemp - coolTemp);
    r = Math.round(cool.r * (1 - t) + mid.r * t);
    g = Math.round(cool.g * (1 - t) + mid.g * t);
    b = Math.round(cool.b * (1 - t) + mid.b * t);
  } else if (temp < hotTemp) {
    const t = (temp - midTemp) / (hotTemp - midTemp);
    r = Math.round(mid.r * (1 - t) + hot.r * t);
    g = Math.round(mid.g * (1 - t) + hot.g * t);
    b = Math.round(mid.b * (1 - t) + hot.b * t);
  } else { ({ r, g, b } = hot); }
  return `rgb(${r},${g},${b})`;
}

function wrapLine(text: string, maxLen: number): string[] {
  if (text.length <= maxLen) return [text];
  const words = text.split(' ');
  const lines: string[] = [];
  let current = '';
  for (const word of words) {
    if (word.length > maxLen) {
      if (current) { lines.push(current); current = ''; }
      for (let i = 0; i < word.length; i += maxLen) lines.push(word.slice(i, i + maxLen));
    } else if ((current + ' ' + word).trim().length > maxLen) {
      lines.push(current); current = word;
    } else {
      current = current ? `${current} ${word}` : word;
    }
  }
  if (current) lines.push(current);
  return lines;
}

function getInterfaceLabel(
  nodeId: string,
  ip: string | undefined,
  nodes: Record<string, NodeInfo>,
): { label: string; missing: boolean } {
  if (!ip) return { label: '?', missing: true };
  const cleanIp = ip.includes(':') && !ip.includes('[') ? ip.split(':')[0]! : ip;

  function checkNode(node: NodeInfo | undefined): string | null {
    if (!node) return null;
    const match = node.network_interfaces?.find((iface) =>
      (iface.addresses ?? []).some((a) => a === cleanIp || a === ip),
    );
    if (match?.name) return match.name;
    if (node.ip_to_interface) {
      const mapped = node.ip_to_interface[cleanIp] ?? (ip ? node.ip_to_interface[ip] : undefined);
      if (mapped?.trim()) return mapped;
    }
    return null;
  }

  const r = checkNode(nodes[nodeId]);
  if (r) return { label: r, missing: false };
  for (const other of Object.values(nodes)) {
    const o = checkNode(other);
    if (o) return { label: o, missing: false };
  }
  return { label: '?', missing: true };
}

// ─── Component ─────────────────────────────────────────────────────────────────

export interface TopologyGraphProps {
  highlightedNodes?: Set<string>;
  filteredNodes?: Set<string>;
  onNodeClick?: (nodeId: string) => void;
  className?: string;
}

export const TopologyGraph: React.FC<TopologyGraphProps> = ({
  highlightedNodes = new Set(),
  filteredNodes = new Set(),
  onNodeClick,
  className,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  const topology = useTopologyStore((s) => s.topology);
  const isMinimized = false; // Controlled by parent via TopologyPane
  const debugEnabled = false; // TODO: hook up UIStore
  const tbBridgeData = useTopologyStore((s) => s.nodeThunderboltBridge);
  const rdmaCtlData = useTopologyStore((s) => s.nodeRdmaCtl);
  const identitiesData = useTopologyStore((s) => s.nodeIdentities);

  const renderGraph = useCallback(() => {
    const svgEl = svgRef.current;
    if (!svgEl || !topology) return;

    d3.select(svgEl).selectAll('*').remove();

    const nodes = topology.nodes ?? {};
    const edges = topology.edges ?? [];
    const nodeIds = Object.keys(nodes);

    const rect = svgEl.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    if (width === 0 || height === 0) return;

    const centerX = width / 2;
    const centerY = height / 2;
    const svg = d3.select(svgEl);
    const defs = svg.append('defs');

    // Glow filter
    const glow = defs.append('filter').attr('id', 'glow')
      .attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
    glow.append('feGaussianBlur').attr('stdDeviation', '2').attr('result', 'coloredBlur');
    const glowMerge = glow.append('feMerge');
    glowMerge.append('feMergeNode').attr('in', 'coloredBlur');
    glowMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Arrowhead marker
    const marker = defs.append('marker').attr('id', 'arrowhead')
      .attr('viewBox', '0 0 10 10').attr('refX', '10').attr('refY', '5')
      .attr('markerWidth', '11').attr('markerHeight', '11').attr('orient', 'auto-start-reverse');
    marker.append('path').attr('d', 'M 0 0 L 10 5 L 0 10')
      .attr('fill', 'none').attr('stroke', 'var(--exo-light-gray, #B3B3B3)')
      .attr('stroke-width', '1.6').attr('stroke-linecap', 'round').attr('stroke-linejoin', 'round')
      .style('animation', 'none');

    if (nodeIds.length === 0) {
      svg.append('text')
        .attr('x', centerX).attr('y', centerY)
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .attr('fill', 'rgba(255,215,0,0.4)').attr('font-size', 12)
        .attr('font-family', 'SF Mono, monospace').attr('letter-spacing', '0.1em')
        .text('AWAITING NODES');
      return;
    }

    const numNodes = nodeIds.length;
    const minDim = Math.min(width, height);
    const sizeScale = numNodes === 1 ? 1 : Math.max(0.6, 1 - (numNodes - 1) * 0.1);
    const baseNodeRadius = Math.min(120, minDim * 0.2);
    const nodeRadius = baseNodeRadius * sizeScale;

    const circumference = numNodes * nodeRadius * 4;
    const radiusFromCircumference = circumference / (2 * Math.PI);
    const minOrbit = Math.max(radiusFromCircumference, minDim * 0.18);
    const maxOrbit = minDim * 0.3;
    const orbitRadius = Math.min(maxOrbit, Math.max(minOrbit, minDim * (0.22 + numNodes * 0.02)));

    const showFullLabels = numNodes <= 4;
    const topPadding = 70;
    const bottomPadding = 70;
    const safeCenterY = topPadding + (height - topPadding - bottomPadding) / 2;

    const nodesWithPositions = nodeIds.map((id, index) => {
      if (numNodes === 1) return { id, data: nodes[id]!, x: centerX, y: safeCenterY };
      const angle = (index / numNodes) * 2 * Math.PI - Math.PI / 2;
      return {
        id,
        data: nodes[id]!,
        x: centerX + orbitRadius * Math.cos(angle),
        y: safeCenterY + orbitRadius * Math.sin(angle),
      };
    });

    const posById: Record<string, { x: number; y: number }> = {};
    nodesWithPositions.forEach((n) => { posById[n.id] = { x: n.x, y: n.y }; });

    // ── Edges ──────────────────────────────────────────────────────────────────
    const linksGroup = svg.append('g');
    const arrowsGroup = svg.append('g');

    type PairEntry = { a: string; b: string; aToB: boolean; bToA: boolean };
    const pairMap = new Map<string, PairEntry>();

    edges.forEach((edge) => {
      if (!edge.source || !edge.target || edge.source === edge.target) return;
      if (!posById[edge.source] || !posById[edge.target]) return;
      const a = edge.source < edge.target ? edge.source : edge.target;
      const b = edge.source < edge.target ? edge.target : edge.source;
      const key = `${a}|${b}`;
      const entry = pairMap.get(key) ?? { a, b, aToB: false, bToA: false };
      if (edge.source === a) entry.aToB = true; else entry.bToA = true;
      pairMap.set(key, entry);
    });

    pairMap.forEach((entry) => {
      const posA = posById[entry.a];
      const posB = posById[entry.b];
      if (!posA || !posB) return;

      // Check if this is an RDMA edge
      const rdmaEdge = edges.find(
        (e) => (e.source === entry.a && e.target === entry.b) ||
               (e.source === entry.b && e.target === entry.a),
      );
      const isRdma = !!(rdmaEdge?.sourceRdmaIface || rdmaEdge?.sinkRdmaIface);

      linksGroup.append('line')
        .attr('x1', posA.x).attr('y1', posA.y)
        .attr('x2', posB.x).attr('y2', posB.y)
        .attr('class', isRdma ? 'graph-link graph-link-active' : 'graph-link');

      const dx = posB.x - posA.x;
      const dy = posB.y - posA.y;
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len; const uy = dy / len;
      const mx = (posA.x + posB.x) / 2;
      const my = (posA.y + posB.y) / 2;
      const tipOffset = 16; const carrier = 2;

      if (entry.aToB) {
        const tx = mx - ux * tipOffset; const ty = my - uy * tipOffset;
        arrowsGroup.append('line')
          .attr('x1', tx - ux * carrier).attr('y1', ty - uy * carrier)
          .attr('x2', tx).attr('y2', ty)
          .attr('stroke', 'none').attr('fill', 'none')
          .attr('marker-end', 'url(#arrowhead)');
      }
      if (entry.bToA) {
        const tx = mx + ux * tipOffset; const ty = my + uy * tipOffset;
        arrowsGroup.append('line')
          .attr('x1', tx + ux * carrier).attr('y1', ty + uy * carrier)
          .attr('x2', tx).attr('y2', ty)
          .attr('stroke', 'none').attr('fill', 'none')
          .attr('marker-end', 'url(#arrowhead)');
      }
    });

    // ── Nodes ──────────────────────────────────────────────────────────────────
    nodesWithPositions.forEach(({ id, data, x, y }) => {
      const isFiltered = filteredNodes.size > 0 && !filteredNodes.has(id);
      const isHighlighted = highlightedNodes.has(id);
      const modelLower = (data.system_info?.model_id ?? 'macbook pro').toLowerCase();
      const isMacStudio = modelLower === 'mac studio' || modelLower === 'mac mini';
      const g = svg.append('g')
        .attr('class', 'node-group')
        .style('cursor', 'pointer')
        .style('opacity', isFiltered ? 0.3 : 1);

      if (onNodeClick) g.on('click', () => onNodeClick(id));

      const identity = identitiesData[id];
      const chipId = identity?.chipId ?? data.system_info?.chip ?? '';
      const temp = data.macmon_info?.temp?.gpu_temp_avg ?? NaN;
      const wireColor = getTemperatureColor(temp);

      const ramUsed = data.macmon_info?.memory?.ram_usage ?? 0;
      const ramTotal = data.macmon_info?.memory?.ram_total ?? data.system_info?.memory ?? 0;
      const ramPct = ramTotal > 0 ? (ramUsed / ramTotal) * 100 : 0;

      // ── Device shape ────────────────────────────────────────────────────────
      const R = nodeRadius;
      const stW = R * 1.25; const stH = R * 0.85;
      const stX = x - stW / 2; const stY = y - stH / 2;
      const stTopH = stH * 0.15;
      const stMemTotalH = stH - stTopH;
      const stMemH = (ramPct / 100) * stMemTotalH;
      const clipId = `tg-clip-${id.replace(/[^a-z0-9]/gi, '_')}`;

      if (isMacStudio) {
        defs.append('clipPath').attr('id', clipId)
          .append('rect')
          .attr('x', stX).attr('y', stY + stTopH)
          .attr('width', stW).attr('height', stH - stTopH).attr('rx', 3);

        g.append('rect')
          .attr('x', stX).attr('y', stY).attr('width', stW).attr('height', stH)
          .attr('rx', 4).attr('fill', '#1a1a1a')
          .attr('stroke', wireColor).attr('stroke-width', STROKE_W);

        if (ramPct > 0) {
          g.append('rect')
            .attr('x', stX).attr('y', stY + stTopH + (stMemTotalH - stMemH))
            .attr('width', stW).attr('height', stMemH)
            .attr('fill', 'rgba(255,215,0,0.75)')
            .attr('clip-path', `url(#${clipId})`);
        }
        g.append('line')
          .attr('x1', stX).attr('y1', stY + stTopH)
          .attr('x2', stX + stW).attr('y2', stY + stTopH)
          .attr('stroke', 'rgba(179,179,179,0.3)').attr('stroke-width', 0.5);
      } else {
        // MacBook
        const mbW = (R * 1.6 * 0.85) / 1.15; const mbH = R * 0.85;
        const mbX = x - mbW / 2; const mbY = y - mbH / 2;
        const mbScreenH = mbH * 0.7; const mbBaseH = mbH * 0.3;
        const mbScreenW = mbW * 0.85; const mbScreenX = x - mbScreenW / 2;
        const bezel = 3;
        const memTotalH = mbScreenH - bezel * 2;
        const memH = (ramPct / 100) * memTotalH;

        defs.append('clipPath').attr('id', clipId)
          .append('rect')
          .attr('x', mbScreenX + bezel).attr('y', mbY + bezel)
          .attr('width', mbScreenW - bezel * 2).attr('height', mbScreenH - bezel * 2).attr('rx', 2);

        g.append('rect').attr('x', mbScreenX).attr('y', mbY)
          .attr('width', mbScreenW).attr('height', mbScreenH).attr('rx', 3)
          .attr('fill', '#1a1a1a').attr('stroke', wireColor).attr('stroke-width', STROKE_W);
        g.append('rect').attr('x', mbScreenX + bezel).attr('y', mbY + bezel)
          .attr('width', mbScreenW - bezel * 2).attr('height', mbScreenH - bezel * 2).attr('rx', 2).attr('fill', '#0a0a12');

        if (ramPct > 0) {
          g.append('rect')
            .attr('x', mbScreenX + bezel).attr('y', mbY + bezel + (memTotalH - memH))
            .attr('width', mbScreenW - bezel * 2).attr('height', memH)
            .attr('fill', 'rgba(255,215,0,0.85)').attr('clip-path', `url(#${clipId})`);
        }

        const logoTargetH = mbScreenH * 0.22;
        const logoScale = logoTargetH / LOGO_NATIVE_H;
        const logoX = x - (LOGO_NATIVE_W * logoScale) / 2;
        const logoY = mbY + mbScreenH / 2 - (LOGO_NATIVE_H * logoScale) / 2;
        g.append('path').attr('d', APPLE_LOGO_PATH)
          .attr('transform', `translate(${logoX},${logoY}) scale(${logoScale})`)
          .attr('fill', '#FFFFFF').attr('opacity', 0.9);

        const baseY = mbY + mbScreenH;
        const btX = x - mbScreenW / 2; const bbX = x - mbW / 2;
        g.append('path')
          .attr('d', `M ${btX} ${baseY} L ${btX + mbScreenW} ${baseY} L ${bbX + mbW} ${baseY + mbBaseH} L ${bbX} ${baseY + mbBaseH} Z`)
          .attr('fill', '#2c2c2c').attr('stroke', wireColor).attr('stroke-width', 1);
        g.append('rect').attr('x', btX + 6).attr('y', baseY + 3)
          .attr('width', mbScreenW - 12).attr('height', mbBaseH * 0.55)
          .attr('fill', 'rgba(0,0,0,0.2)').attr('rx', 2);
        const tpW = mbScreenW * 0.4; const tpX = x - tpW / 2;
        g.append('rect').attr('x', tpX).attr('y', baseY + mbBaseH * 0.55 + 8)
          .attr('width', tpW).attr('height', mbBaseH * 0.3)
          .attr('fill', 'rgba(255,255,255,0.08)').attr('rx', 2);
      }

      // ── Node label ──────────────────────────────────────────────────────────
      const labelY = y + R * 0.6;
      const label = data.friendly_name ?? id.slice(0, 8);

      if (showFullLabels) {
        const lines = wrapLine(label, 14);
        lines.forEach((line, i) => {
          g.append('text').attr('x', x).attr('y', labelY + i * 14)
            .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.9)')
            .attr('font-size', 11).attr('font-family', 'SF Mono, monospace')
            .attr('letter-spacing', '0.05em').text(line);
        });

        // Chip label
        if (chipId) {
          g.append('text').attr('x', x).attr('y', labelY + lines.length * 14 + 2)
            .attr('text-anchor', 'middle').attr('fill', 'rgba(255,215,0,0.6)')
            .attr('font-size', 9).attr('font-family', 'SF Mono, monospace')
            .attr('letter-spacing', '0.08em').text(chipId.toUpperCase());
        }

        // Memory stats
        if (ramTotal > 0) {
          const usedStr = formatBytes(ramUsed);
          const totalStr = formatBytes(ramTotal);
          g.append('text').attr('x', x).attr('y', labelY + lines.length * 14 + (chipId ? 16 : 4))
            .attr('text-anchor', 'middle').attr('fill', 'rgba(255,215,0,0.8)')
            .attr('font-size', 9).attr('font-family', 'SF Mono, monospace')
            .text(`${usedStr} / ${totalStr}`);
        }

        // GPU usage
        const gpuUsage = data.macmon_info?.gpu_usage?.[0] ?? null;
        if (gpuUsage !== null) {
          g.append('text').attr('x', x).attr('y', labelY + lines.length * 14 + (chipId ? 28 : 16))
            .attr('text-anchor', 'middle').attr('fill', 'rgba(200,200,200,0.6)')
            .attr('font-size', 9).attr('font-family', 'SF Mono, monospace')
            .text(`GPU ${Math.round(gpuUsage)}%`);
        }
      } else {
        // Compact label
        g.append('text').attr('x', x).attr('y', labelY)
          .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.8)')
          .attr('font-size', 9).attr('font-family', 'SF Mono, monospace')
          .text(label.slice(0, 10));
      }

      // Glow highlight
      if (isHighlighted) {
        g.select('rect').attr('filter', 'url(#glow)');
      }
    });

    // ── Header label ───────────────────────────────────────────────────────────
    svg.append('text').attr('x', centerX).attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,215,0,0.3)').attr('font-size', 10)
      .attr('font-family', 'SF Mono, monospace').attr('letter-spacing', '0.15em')
      .text('NETWORK TOPOLOGY');
  }, [topology, debugEnabled, filteredNodes, highlightedNodes, onNodeClick,
      tbBridgeData, rdmaCtlData, identitiesData]);

  // Re-render on data change
  useEffect(() => {
    renderGraph();
  }, [renderGraph]);

  // Re-render on resize
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => renderGraph());
    observer.observe(el);
    return () => observer.disconnect();
  }, [renderGraph]);

  const STROKE_W = 1.5;

  return <SvgEl ref={svgRef} className={className} />;
};
