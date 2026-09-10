import { useEffect, useRef, useState } from 'react';
import styled, { useTheme } from 'styled-components';
import { FiExternalLink, FiPlay, FiInfo, FiX } from 'react-icons/fi';
import type { Theme } from '../../theme';
import { useSkulkTranslation } from '../../i18n/tolgee';
import {
  capabilityNodeHealth,
  capabilityNodeKey,
  capabilityNodeTitle,
  type CapabilityNodeSummary,
} from '../../types/capabilityNodes';
import { buildCapabilityActions, runDescriptorAction, type CapabilityActionItem } from './capabilityActions';
import { satelliteColor, satelliteStatusLabel } from './capabilityPresentation';

/** Width of the flyout card, used to keep it inside the canvas. */
export const FLYOUT_WIDTH = 252;
/** Graph controls whose presses toggle flyouts themselves. */
const SATELLITE_CONTROL_SELECTOR = '.topology-capability-satellite, .topology-capability-overflow';
const CANVAS_MARGIN = 8;

/** Props for the flyout that opens from a capability satellite. */
export interface CapabilityFlyoutProps {
  /** Every visible capability node on the host, for the switcher. */
  summaries: CapabilityNodeSummary[];
  /** Key of the node the flyout currently shows. */
  selectedKey: string;
  /** Switches the flyout to another node on the same host. */
  onSelectKey: (key: string) => void;
  /** Node id of the host running the capability. */
  hostNodeId: string;
  /** Friendly name of the host. */
  hostName: string;
  /** Node id the dashboard is served by; null while unknown. */
  localNodeId: string | null;
  /** Canvas coordinates of the satellite the flyout hangs from. */
  anchor: { x: number; y: number };
  /** Canvas size, so the card stays visible near the edges. */
  canvasWidth: number;
  canvasHeight: number;
  onClose: () => void;
  /** Opens the capability panel on this node. */
  onOpenPanel: (hostNodeId: string, key: string) => void;
}

const Card = styled.div`
  position: absolute;
  box-sizing: border-box;
  width: ${FLYOUT_WIDTH}px;
  padding: 10px 10px 8px;
  background: ${({ theme }) => theme.colors.surfaceElevated};
  border: 1px solid ${({ theme }) => theme.colors.borderStrong};
  border-radius: ${({ theme }) => theme.radii.md};
  box-shadow: 0 12px 32px ${({ theme }) => theme.colors.shadow};
  color: ${({ theme }) => theme.colors.text};
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  z-index: 5;
`;

const HeaderRow = styled.div`
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
`;

const TitleBlock = styled.div`
  min-width: 0;
  flex: 1;
`;

const Title = styled.div`
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const Meta = styled.div`
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: 11px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const StatusPill = styled.span<{ $color: string }>`
  display: inline-flex;
  align-items: center;
  gap: 5px;
  margin-top: 4px;
  font-size: 11px;
  color: ${({ theme }) => theme.colors.textSecondary};

  &::before {
    content: '';
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: ${({ $color }) => $color};
  }
`;

const CloseButton = styled.button`
  appearance: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  margin: -2px -2px 0 0;
  padding: 0;
  background: transparent;
  border: 0;
  border-radius: ${({ theme }) => theme.radii.sm};
  color: ${({ theme }) => theme.colors.textSecondary};
  cursor: pointer;

  &:hover,
  &:focus-visible {
    background: ${({ theme }) => theme.colors.goldBg};
    color: ${({ theme }) => theme.colors.text};
    outline: none;
  }
`;

const Switcher = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 8px;
`;

const SwitcherChip = styled.button<{ $active: boolean }>`
  appearance: none;
  padding: 2px 8px;
  background: ${({ $active, theme }) => ($active ? theme.colors.goldBg : 'transparent')};
  border: 1px solid ${({ $active, theme }) => ($active ? theme.colors.gold : theme.colors.border)};
  border-radius: 999px;
  color: ${({ $active, theme }) => ($active ? theme.colors.text : theme.colors.textSecondary)};
  cursor: pointer;
  font: inherit;
  font-size: 11px;
`;

const ActionList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid ${({ theme }) => theme.colors.border};
`;

const actionRowStyles = `
  appearance: none;
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  box-sizing: border-box;
  padding: 6px 8px;
  background: transparent;
  border: 0;
  border-radius: 6px;
  color: inherit;
  cursor: pointer;
  font: inherit;
  text-align: left;
  text-decoration: none;
`;

const ActionLink = styled.a<{ $muted?: boolean }>`
  ${actionRowStyles}
  opacity: ${({ $muted }) => ($muted ? 0.6 : 1)};

  &:hover,
  &:focus-visible {
    background: ${({ theme }) => theme.colors.goldBg};
    outline: none;
  }
`;

const ActionButton = styled.button`
  ${actionRowStyles}

  &:disabled {
    cursor: default;
    opacity: 0.55;
  }

  &:not(:disabled):hover,
  &:not(:disabled):focus-visible {
    background: ${({ theme }) => theme.colors.goldBg};
    outline: none;
  }
`;

const ActionText = styled.span`
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const Hint = styled.div`
  margin-top: 6px;
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: 11px;
  line-height: 1.4;
`;

const CallOutcome = styled.div<{ $error: boolean }>`
  margin-top: 6px;
  padding: 6px 8px;
  border-radius: 6px;
  background: ${({ $error, theme }) => ($error ? theme.colors.errorBg : theme.colors.goldBg)};
  color: ${({ $error, theme }) => ($error ? theme.colors.errorOnSurface : theme.colors.textSecondary)};
  font-size: 11px;
  line-height: 1.4;
  word-break: break-word;
`;

/**
 * HTML popover anchored to a satellite inside the topology canvas. Link
 * surfaces open in a new tab with `noopener`, descriptor actions call the
 * local host, and everything else routes to the capability panel. Closes on
 * Escape or a click outside the card.
 */
export function CapabilityFlyout({
  summaries,
  selectedKey,
  onSelectKey,
  hostNodeId,
  hostName,
  localNodeId,
  anchor,
  canvasWidth,
  canvasHeight,
  onClose,
  onOpenPanel,
}: CapabilityFlyoutProps) {
  const { t } = useSkulkTranslation();
  const theme = useTheme() as Theme;
  const cardRef = useRef<HTMLDivElement | null>(null);
  const [callOutcome, setCallOutcome] = useState<{ id: string; text: string; error: boolean } | null>(null);
  const [callInFlight, setCallInFlight] = useState<string | null>(null);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (!cardRef.current || !target || cardRef.current.contains(target)) return;
      // A press on a satellite or the overflow glyph is the graph's own
      // toggle, not an outside click; closing here would let the following
      // click reopen the flyout it just closed.
      const element = target instanceof Element ? target : target.parentElement;
      if (element?.closest(SATELLITE_CONTROL_SELECTOR)) return;
      onClose();
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('pointerdown', onPointerDown);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('pointerdown', onPointerDown);
    };
  }, [onClose]);

  useEffect(() => {
    setCallOutcome(null);
  }, [selectedKey]);

  const summary = summaries.find((candidate) => capabilityNodeKey(candidate) === selectedKey) ?? summaries[0];
  if (!summary) return null;
  const isLocalHost = localNodeId !== null && localNodeId === hostNodeId;
  const actions = buildCapabilityActions(summary, {
    isLocalHost,
    hostName,
    dashboardHostname: window.location.hostname,
    t,
  });
  const level = capabilityNodeHealth(summary);
  const left = Math.max(
    CANVAS_MARGIN,
    Math.min(canvasWidth - FLYOUT_WIDTH - CANVAS_MARGIN, anchor.x - FLYOUT_WIDTH / 2),
  );
  // Prefer hanging below the satellite; flip above when the canvas is short.
  const below = anchor.y + 18;
  const top = canvasHeight > 0 && below > canvasHeight * 0.6 ? undefined : below;
  const bottom = top === undefined ? Math.max(CANVAS_MARGIN, canvasHeight - anchor.y + 18) : undefined;

  const runCall = async (item: Extract<CapabilityActionItem, { kind: 'call' }>) => {
    if (!localNodeId) return;
    setCallInFlight(item.id);
    setCallOutcome(null);
    try {
      const result = await runDescriptorAction(localNodeId, item.capabilityId, item.payload);
      if (result.ok === false) {
        const message = result.error?.message ?? result.error?.code ?? 'error';
        setCallOutcome({ id: item.id, text: message, error: true });
      } else {
        setCallOutcome({
          id: item.id,
          text: t('topology.capability.callSucceeded', '{title} completed', { title: item.title }),
          error: false,
        });
      }
    } catch (error: unknown) {
      setCallOutcome({ id: item.id, text: error instanceof Error ? error.message : String(error), error: true });
    } finally {
      setCallInFlight(null);
    }
  };

  return (
    <Card
      aria-label={t('topology.capability.flyoutAria', '{title} capability node actions', {
        title: capabilityNodeTitle(summary),
      })}
      data-capability-flyout={selectedKey}
      ref={cardRef}
      role="dialog"
      style={{ left, top, bottom }}
    >
      <HeaderRow>
        <TitleBlock>
          <Title>{capabilityNodeTitle(summary)}</Title>
          <Meta>
            {summary.bundleId} · {summary.version} · {hostName}
          </Meta>
          <StatusPill $color={satelliteColor(level, theme)}>
            {satelliteStatusLabel(summary, t)}
            {summary.operationsActive > 0
              ? ` · ${t('topology.capability.operationsActive', '{count} running', {
                  count: summary.operationsActive,
                })}`
              : ''}
          </StatusPill>
        </TitleBlock>
        <CloseButton aria-label={t('topology.capability.closeFlyout', 'Close')} onClick={onClose} type="button">
          <FiX aria-hidden size={14} />
        </CloseButton>
      </HeaderRow>
      {summaries.length > 1 ? (
        <Switcher aria-label={t('topology.capability.switcherAria', 'Capability nodes on this host')} role="group">
          {summaries.map((candidate) => {
            const key = capabilityNodeKey(candidate);
            return (
              <SwitcherChip
                $active={key === selectedKey}
                aria-pressed={key === selectedKey}
                key={key}
                onClick={() => onSelectKey(key)}
                type="button"
              >
                {capabilityNodeTitle(candidate)}
              </SwitcherChip>
            );
          })}
        </Switcher>
      ) : null}
      <ActionList>
        {actions.map((item) => {
          if (item.kind === 'open-link') {
            if (!item.reachable) {
              return (
                <ActionButton
                  disabled
                  key={item.id}
                  title={t(
                    'topology.capability.surfaceOnHostOnly',
                    'Reachable only from a browser on {host}',
                    { host: hostName },
                  )}
                  type="button"
                >
                  <FiExternalLink aria-hidden size={14} />
                  <ActionText>{item.title}</ActionText>
                </ActionButton>
              );
            }
            return (
              <ActionLink
                $muted={!item.ready}
                href={item.url}
                key={item.id}
                rel="noopener noreferrer"
                target="_blank"
                title={item.ready ? item.url : t('topology.capability.surfaceNotReady', 'Not answering yet')}
              >
                <FiExternalLink aria-hidden size={14} />
                <ActionText>{item.title}</ActionText>
              </ActionLink>
            );
          }
          if (item.kind === 'call') {
            return (
              <ActionButton
                disabled={!item.enabled || callInFlight !== null}
                key={item.id}
                onClick={() => void runCall(item)}
                title={
                  item.enabled
                    ? item.capabilityId
                    : t('topology.capability.callNeedsHost', 'Runs only from the dashboard on {host}', {
                        host: hostName,
                      })
                }
                type="button"
              >
                <FiPlay aria-hidden size={14} />
                <ActionText>{item.title}</ActionText>
              </ActionButton>
            );
          }
          if (item.kind === 'details') {
            return (
              <ActionButton key={item.id} onClick={() => onOpenPanel(hostNodeId, selectedKey)} type="button">
                <FiInfo aria-hidden size={14} />
                <ActionText>{item.title}</ActionText>
              </ActionButton>
            );
          }
          return (
            <Hint key={item.id}>
              {t(
                'topology.capability.manageOnHostHint',
                'Settings and actions for this node are managed from the dashboard on {host}.',
                { host: item.hostName },
              )}
            </Hint>
          );
        })}
      </ActionList>
      {callOutcome ? <CallOutcome $error={callOutcome.error}>{callOutcome.text}</CallOutcome> : null}
    </Card>
  );
}
