import { useTheme } from 'styled-components';
import { FiAlertTriangle } from 'react-icons/fi';
import { InfoTooltip } from '../common/InfoTooltip';
import type { Theme } from '../../theme';
import type { NodeHealth } from '../../types/topology';
import { useSkulkTranslation } from '../../i18n/tolgee';

/** Props for {@link NodeHealthBadge}. */
export interface NodeHealthBadgeProps {
  /** Derived health for the node; the badge renders nothing when healthy. */
  health?: NodeHealth;
  /** SVG x of the badge's top-left within the parent node group. */
  x: number;
  /** SVG y of the badge's top-left within the parent node group. */
  y: number;
}

/**
 * Topology overlay that surfaces a node's derived problems (#388).
 *
 * Renders an amber (warn) or red (error) alert icon on the node; hovering shows
 * each problem's message together with its remediation, so the master's silent
 * recovery of a wedged/failed node is legible to the operator. Renders nothing
 * for a healthy node (level `ok` or no reasons), so the topology stays quiet
 * until something actually needs attention.
 */
export function NodeHealthBadge({ health, x, y }: NodeHealthBadgeProps) {
  const { t } = useSkulkTranslation();
  const theme = useTheme() as Theme;

  if (!health || health.level === 'ok' || health.reasons.length === 0) {
    return null;
  }

  const isError = health.level === 'error';
  const color = isError ? theme.colors.error : theme.colors.warning;
  const heading = isError
    ? t('topology.nodeHealth.errorHeading', 'This node has a problem')
    : t('topology.nodeHealth.warnHeading', 'This node needs attention');
  const ariaLabel = isError
    ? t('topology.nodeHealth.errorAria', 'Node problem')
    : t('topology.nodeHealth.warnAria', 'Node warning');

  const content = (
    <div style={{ lineHeight: 1.5, maxWidth: 300 }}>
      <div style={{ color, fontWeight: 600, marginBottom: 6 }}>{heading}</div>
      {health.reasons.map((reason, index) => (
        <div key={`${reason.code}-${index}`} style={{ marginBottom: 8 }}>
          <div style={{ color: theme.colors.text }}>{reason.message}</div>
          {reason.remediation && (
            <div style={{ color: theme.colors.textSecondary, marginTop: 2 }}>
              {t('topology.nodeHealth.fixPrefix', 'Fix: {remediation}', {
                remediation: reason.remediation,
              })}
            </div>
          )}
        </div>
      ))}
    </div>
  );

  return (
    <foreignObject x={x} y={y} width={22} height={22} style={{ overflow: 'visible' }}>
      <InfoTooltip placement="top" content={content}>
        <span
          role="img"
          aria-label={ariaLabel}
          style={{ display: 'flex', color, cursor: 'help' }}
        >
          <FiAlertTriangle size={18} />
        </span>
      </InfoTooltip>
    </foreignObject>
  );
}
