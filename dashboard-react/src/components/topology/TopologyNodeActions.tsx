import { useCallback, useEffect, useRef, useState } from 'react';
import { FiActivity, FiRefreshCw } from 'react-icons/fi';
import styled from 'styled-components';
import { InfoTooltip } from '../common/InfoTooltip';
import { useSkulkTranslation } from '../../i18n/tolgee';

/** Props for the compact action rail attached to an active topology node. */
export interface TopologyNodeActionsProps {
  /** Rich node and connection details shown from the info action. */
  infoContent: React.ReactNode;
  /** Restarts the node after the local two-step confirmation. */
  onRestart?: () => void;
  /** Opens live node diagnostics. */
  onInspect?: () => void;
}

const Rail = styled.div`
  box-sizing: border-box;
  display: grid;
  grid-template-columns: repeat(3, 34px);
  width: 106px;
  height: 36px;
  overflow: hidden;
  background: ${({ theme }) => theme.colors.surfaceElevated};
  border: 1px solid ${({ theme }) => theme.colors.borderStrong};
  border-radius: ${({ theme }) => theme.radii.md};
  box-shadow: 0 8px 24px ${({ theme }) => theme.colors.shadow};
  color: ${({ theme }) => theme.colors.textSecondary};

  > * + * {
    border-left: 1px solid ${({ theme }) => theme.colors.border};
  }

  .topology-info-action {
    width: 34px;
    height: 34px;
    cursor: help;
    color: inherit;
  }

  .topology-action-tooltip {
    width: 34px;
    height: 34px;
    cursor: pointer;
    color: inherit;
  }
`;

const ActionButton = styled.button<{ $confirming?: boolean }>`
  appearance: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  margin: 0;
  padding: 0;
  background: ${({ $confirming, theme }) => ($confirming ? theme.colors.warningBg : 'transparent')};
  border: 0;
  color: ${({ $confirming, theme }) => ($confirming ? theme.colors.warningOnSurface : theme.colors.textSecondary)};
  cursor: pointer;
  font: inherit;
  transition: background 140ms ease, color 140ms ease;

  &:hover,
  &:focus-visible {
    background: ${({ theme }) => theme.colors.goldBg};
    color: ${({ theme }) => theme.colors.text};
    outline: none;
  }
`;

/**
 * SVG-compatible HTML action rail for restart, diagnostics, and node info.
 * Restart deliberately remains a two-click operation so the more compact
 * visual treatment does not make a disruptive action easier to trigger.
 */
export function TopologyNodeActions({
  infoContent,
  onRestart,
  onInspect,
}: TopologyNodeActionsProps) {
  const { t } = useSkulkTranslation();
  const [confirming, setConfirming] = useState(false);
  const confirmTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (confirmTimerRef.current) clearTimeout(confirmTimerRef.current);
    };
  }, []);

  const restart = useCallback(() => {
    if (!confirming) {
      setConfirming(true);
      confirmTimerRef.current = setTimeout(() => setConfirming(false), 3000);
      return;
    }
    if (confirmTimerRef.current) clearTimeout(confirmTimerRef.current);
    confirmTimerRef.current = null;
    setConfirming(false);
    onRestart?.();
  }, [confirming, onRestart]);

  return (
    <Rail aria-label={t('topology.nodeActions.label', 'Node actions')} role="toolbar">
      <InfoTooltip
        className="topology-action-tooltip"
        content={
          confirming
            ? t('nodeLabel.confirmRestartTooltip', 'Click again to confirm restart')
            : t(
                'nodeLabel.restartTooltip',
                'Restart this node - releases GPU memory and rejoins the cluster',
              )
        }
        placement="bottom"
      >
        <ActionButton
          $confirming={confirming}
          aria-label={
            confirming
              ? t('nodeLabel.confirmRestart', 'Confirm restart of this node')
              : t('nodeLabel.restart', 'Restart this node')
          }
          onClick={restart}
          type="button"
        >
          <FiRefreshCw aria-hidden size={17} />
        </ActionButton>
      </InfoTooltip>
      <InfoTooltip
        className="topology-action-tooltip"
        content={t('nodeLabel.inspectDiagnostics', 'Inspect live node diagnostics')}
        placement="bottom"
      >
        <ActionButton
          aria-label={t('nodeLabel.inspectDiagnostics', 'Inspect live node diagnostics')}
          disabled={!onInspect}
          onClick={onInspect}
          type="button"
        >
          <FiActivity aria-hidden size={18} />
        </ActionButton>
      </InfoTooltip>
      <InfoTooltip
        className="topology-info-action"
        content={infoContent}
        filled
        placement="bottom"
        size={18}
        triggerLabel={t('topology.nodeActions.info', 'Node information')}
      />
    </Rail>
  );
}
