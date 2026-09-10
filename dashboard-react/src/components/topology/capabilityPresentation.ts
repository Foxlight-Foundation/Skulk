import type { Theme } from '../../theme';
import type { SkulkTranslate } from '../../i18n/tolgee';
import {
  capabilityNodeHealth,
  type CapabilityNodeHealthLevel,
  type CapabilityNodeSummary,
} from '../../types/capabilityNodes';

/** Theme color for a satellite health level. */
export function satelliteColor(level: CapabilityNodeHealthLevel, theme: Theme): string {
  if (level === 'error') return theme.colors.topologyNodeDanger;
  if (level === 'warn') return theme.colors.topologyNodeWarning;
  if (level === 'muted') return theme.colors.textMuted;
  return theme.colors.topologyNodeHealthy;
}

/** Spoken status for a satellite, used in its accessible name. */
export function satelliteStatusLabel(summary: CapabilityNodeSummary, t: SkulkTranslate): string {
  const level = capabilityNodeHealth(summary);
  if (level === 'muted' && summary.status !== 'installed' && summary.status !== 'disabled') {
    return t('topology.capability.status.stale', 'no recent report');
  }
  switch (summary.status) {
    case 'ready':
      return summary.ownerAvailable
        ? t('topology.capability.status.ready', 'ready')
        : t('topology.capability.status.ownerUnavailable', 'owner unavailable');
    case 'starting':
      return t('topology.capability.status.starting', 'starting');
    case 'degraded':
      return t('topology.capability.status.degraded', 'degraded');
    case 'configuration_invalid':
      return t('topology.capability.status.configurationInvalid', 'configuration invalid');
    case 'failed':
      return t('topology.capability.status.failed', 'failed');
    case 'disabled':
      return t('topology.capability.status.disabled', 'disabled');
    default:
      return t('topology.capability.status.installed', 'installed');
  }
}
