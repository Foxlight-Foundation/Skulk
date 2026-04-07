import { useMemo, useState } from 'react';
import styled from 'styled-components';
import { FiAlertTriangle, FiX } from 'react-icons/fi';
import type { TopologyData } from '../../types/topology';

interface ClusterWarningsProps {
  topology: TopologyData | null;
}

interface VersionEntry {
  friendlyName: string;
  version: string;
  commit: string;
}

// Icons now from react-icons

export function ClusterWarnings({ topology }: ClusterWarningsProps) {
  const nodes = topology?.nodes;
  const [versionDismissed, setVersionDismissed] = useState(false);
  const [rdmaDismissed, setRdmaDismissed] = useState(false);

  const versionMismatch = useMemo<VersionEntry[] | null>(() => {
    if (!nodes) return null;
    const entries = Object.values(nodes).filter(
      (n) => n.exo_commit && n.exo_commit !== 'Unknown' && n.exo_commit !== 'unknown',
    );
    if (entries.length < 2) return null;
    const commits = new Set(entries.map((n) => n.exo_commit));
    if (commits.size <= 1) return null;
    return entries.map((n) => ({
      friendlyName: n.friendly_name ?? 'Unknown',
      version: n.exo_version ?? 'Unknown',
      commit: n.exo_commit!,
    }));
  }, [nodes]);

  const rdmaPhantom = useMemo(() => {
    if (!nodes) return false;
    return Object.values(nodes).some(
      (n) => n.rdma_enabled && n.rdma_interfaces_present === false,
    );
  }, [nodes]);

  const showVersion = versionMismatch && !versionDismissed;
  const showRdma = rdmaPhantom && !rdmaDismissed;

  if (!showVersion && !showRdma) return null;

  return (
    <WarningsBar>
      {showVersion && (
        <WarningPill $color="error">
          <WarningIcon $color="error" />
          <WarningLabel $color="error">Version Mismatch</WarningLabel>
          <DismissButton $color="error" onClick={() => setVersionDismissed(true)} aria-label="Dismiss">
            <FiX size={14} />
          </DismissButton>
          <Tooltip className="warning-tooltip" $color="error">
            <TooltipInner $color="error">
              <p>
                Nodes in this cluster are running different versions.
                This will cause inference failures and unexpected behavior.
              </p>
              <NodeList>
                {versionMismatch.map((n) => (
                  <li key={n.friendlyName}>
                    {n.friendlyName} — v{n.version} ({n.commit})
                  </li>
                ))}
              </NodeList>
              <p>
                <Emphasis $color="error">Action required:</Emphasis> Update all
                nodes to the same version with{' '}
                <Code>git pull && uv sync</Code>.
              </p>
            </TooltipInner>
          </Tooltip>
        </WarningPill>
      )}

      {showRdma && (
        <WarningPill $color="warning">
          <WarningIcon $color="warning" />
          <WarningLabel $color="warning">RDMA Not Available</WarningLabel>
          <DismissButton $color="warning" onClick={() => setRdmaDismissed(true)} aria-label="Dismiss">
            <FiX size={14} />
          </DismissButton>
          <Tooltip className="warning-tooltip" $color="warning">
            <TooltipInner $color="warning">
              <p>
                macOS reports RDMA as enabled but no RDMA network interfaces
                exist. This typically means your hardware has Thunderbolt 4
                ports, which do not support RDMA. Thunderbolt 5 (M4 Pro/Max
                or newer) is required.
              </p>
              <p>
                <Emphasis $color="warning">Impact:</Emphasis> Tensor parallel
                (MlxJaccl) is not available. Pipeline parallel (MlxRing) works
                normally over Thunderbolt.
              </p>
            </TooltipInner>
          </Tooltip>
        </WarningPill>
      )}
    </WarningsBar>
  );
}

/* ── Styled Components ──────────────────────────────────── */

type ColorKey = 'error' | 'warning';

const colorMap: Record<ColorKey, { border: string; bg: string; text: string; emphasis: string }> = {
  error: {
    border: 'rgba(239,68,68,0.5)',
    bg: 'rgba(239,68,68,0.1)',
    text: 'rgba(254,202,202,1)',
    emphasis: 'rgb(252,165,165)',
  },
  warning: {
    border: 'rgba(249,115,22,0.5)',
    bg: 'rgba(249,115,22,0.1)',
    text: 'rgba(253,186,116,1)',
    emphasis: 'rgb(251,146,60)',
  },
};

const WarningsBar = styled.div`
  position: relative;
  z-index: 10;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 8px 16px;
`;

const WarningPill = styled.div<{ $color: ColorKey }>`
  position: relative;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-radius: 6px;
  border: 1px solid ${({ $color }) => colorMap[$color].border};
  background: ${({ $color }) => colorMap[$color].bg};
  backdrop-filter: blur(8px);
  cursor: help;

  &:hover > .warning-tooltip,
  & > .warning-tooltip:hover {
    opacity: 1;
    visibility: visible;
  }
`;

const WarningLabel = styled.span<{ $color: ColorKey }>`
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ $color }) => colorMap[$color].text};
`;

function WarningIcon({ $color }: { d?: string; $color: ColorKey }) {
  return (
    <FiAlertTriangle size={18} color={colorMap[$color].emphasis} style={{ flexShrink: 0 }} />
  );
}

const DismissButton = styled.button<{ $color: ColorKey }>`
  display: flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: none;
  padding: 2px;
  margin-left: 4px;
  cursor: pointer;
  color: ${({ $color }) => colorMap[$color].text};
  opacity: 0.6;
  transition: opacity 0.15s;
  flex-shrink: 0;

  &:hover {
    opacity: 1;
  }
`;

const Tooltip = styled.div<{ $color: ColorKey }>`
  position: absolute;
  top: 100%;
  left: 0;
  /* padding-top creates an invisible hover bridge between trigger and content */
  padding-top: 6px;
  width: 320px;
  opacity: 0;
  visibility: hidden;
  transition: opacity 0.2s ease, visibility 0.2s ease;
  z-index: 50;
`;

const TooltipInner = styled.div<{ $color: ColorKey }>`
  padding: 12px;
  border-radius: 8px;
  border: 1px solid ${({ $color }) => colorMap[$color].border};
  background: ${({ theme }) => theme.colors.surfaceElevated};
  backdrop-filter: blur(8px);
  box-shadow: 0 8px 32px ${({ theme }) => theme.colors.shadow};

  p {
    font-size: ${({ theme }) => theme.fontSizes.sm};
    color: ${({ theme }) => theme.colors.textSecondary};
    margin: 0 0 8px;
    line-height: 1.5;
  }

  p:last-child {
    margin-bottom: 0;
  }
`;

const NodeList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0 0 8px;
  font-size: ${({ theme }) => theme.fontSizes.label};
  color: ${({ theme }) => theme.colors.textSecondary};

  li {
    padding-left: 8px;
  }
`;

const Emphasis = styled.span<{ $color: ColorKey }>`
  color: ${({ $color }) => colorMap[$color].emphasis};
`;

const Code = styled.code`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
`;
