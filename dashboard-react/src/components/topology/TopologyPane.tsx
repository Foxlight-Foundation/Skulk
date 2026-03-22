/**
 * TopologyPane
 *
 * The right-hand panel showing the cluster network graph with the
 * CRT-screen aesthetic, minimize toggle, and system warnings.
 */
import React from 'react';
import styled from 'styled-components';
import { useTranslate } from '@tolgee/react';
import { TopologyGraph } from './TopologyGraph';
import { SystemWarningsBanner } from './SystemWarningsBanner';
import { useUIStore } from '../../stores/uiStore';
import { useTopologyStore } from '../../stores/topologyStore';

// ─── Styled components ────────────────────────────────────────────────────────

const Pane = styled.div<{ $minimized: boolean }>`
  display: flex;
  flex-direction: column;
  position: relative;
  flex-shrink: 0;
  transition: width ${({ theme }) => theme.transitions.slow},
              min-height ${({ theme }) => theme.transitions.slow};
  width: ${({ $minimized }) => ($minimized ? '200px' : '380px')};
  border-right: 1px solid ${({ theme }) => theme.colors.border};
  background: ${({ theme }) => theme.colors.darkGray};
  overflow: hidden;

  @media (max-width: 768px) {
    width: 100%;
    height: ${({ $minimized }) => ($minimized ? '120px' : '260px')};
    border-right: none;
    border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  }
`;

const TopBar = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  flex-shrink: 0;
`;

const Title = styled.span`
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.lightGray};
`;

const NodeCount = styled.span`
  font-size: 10px;
  letter-spacing: 0.08em;
  color: ${({ theme }) => theme.colors.yellow};
`;

const MinimizeButton = styled.button`
  background: none;
  border: none;
  color: ${({ theme }) => theme.colors.lightGray};
  cursor: pointer;
  font-size: 12px;
  padding: 2px 6px;
  border-radius: ${({ theme }) => theme.radius.sm};
  transition: color ${({ theme }) => theme.transitions.fast},
              background ${({ theme }) => theme.transitions.fast};
  &:hover {
    color: ${({ theme }) => theme.colors.yellow};
    background: ${({ theme }) => theme.colors.mediumGray};
  }
`;

const GraphArea = styled.div`
  flex: 1;
  position: relative;
  overflow: hidden;
  /* CRT screen radial gradient */
  background: radial-gradient(
    ellipse at center,
    oklch(0.16 0 0) 0%,
    oklch(0.12 0 0) 50%,
    oklch(0.09 0 0) 100%
  );
`;

const LastUpdateLabel = styled.div`
  position: absolute;
  bottom: 6px;
  right: 8px;
  font-size: 9px;
  letter-spacing: 0.07em;
  color: ${({ theme }) => theme.colors.lightGray};
  opacity: 0.5;
  pointer-events: none;
`;

// ─── Component ─────────────────────────────────────────────────────────────────

export const TopologyPane: React.FC = () => {
  const { t } = useTranslate();
  const minimized = useUIStore((s) => s.topologyMinimized);
  const toggleMinimized = useUIStore((s) => s.toggleTopologyMinimized);

  const topology = useTopologyStore((s) => s.topology);
  const lastUpdate = useTopologyStore((s) => s.lastUpdate);

  const nodeCount = Object.keys(topology?.nodes ?? {}).length;

  const lastUpdateLabel = lastUpdate
    ? new Date(lastUpdate).toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })
    : null;

  return (
    <Pane $minimized={minimized}>
      <TopBar>
        <Title>{t('topology.title')}</Title>
        <NodeCount>
          {nodeCount > 0
            ? t('topology.nodes', { count: nodeCount })
            : t('topology.waiting')}
        </NodeCount>
        <MinimizeButton
          onClick={toggleMinimized}
          title={minimized ? t('topology.expand') : t('topology.minimize')}
          aria-label={minimized ? t('topology.expand') : t('topology.minimize')}
        >
          {minimized ? '↔' : '↕'}
        </MinimizeButton>
      </TopBar>

      <SystemWarningsBanner />

      <GraphArea className="scanlines">
        <TopologyGraph />
        {lastUpdateLabel && (
          <LastUpdateLabel>
            {t('topology.last_update', { time: lastUpdateLabel })}
          </LastUpdateLabel>
        )}
      </GraphArea>
    </Pane>
  );
};
