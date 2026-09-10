import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled, { useTheme } from 'styled-components';
import { FiExternalLink, FiPlay } from 'react-icons/fi';
import type { Theme } from '../../theme';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import {
  uiActions,
  type CapabilityPanelTab,
  OBSERVABILITY_WIDTH_MIN,
  OBSERVABILITY_WIDTH_MAX,
} from '../../store/slices/uiSlice';
import { useClusterState } from '../../hooks/useClusterState';
import { useSkulkTranslation } from '../../i18n/tolgee';
import {
  capabilityNodeHealth,
  capabilityNodeKey,
  capabilityNodeTitle,
  type CapabilityNodeSummary,
} from '../../types/capabilityNodes';
import { RightDrawer } from '../common/RightDrawer';
import { DrawerBody, DrawerTabBar, DrawerTabButton } from '../common/drawerParts';
import {
  buildCapabilityActions,
  resolveSurfaceUrl,
  runDescriptorAction,
  type CapabilityActionItem,
} from '../topology/capabilityActions';
import { satelliteColor, satelliteStatusLabel } from '../topology/capabilityPresentation';

const TAB_ORDER: CapabilityPanelTab[] = ['overview', 'surfaces', 'actions'];

const Scroll = styled.div`
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.text};
`;

const Grid = styled.dl`
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 6px 14px;
  margin: 0;

  dt {
    color: ${({ theme }) => theme.colors.textMuted};
  }

  dd {
    margin: 0;
    overflow-wrap: anywhere;
  }
`;

const Status = styled.span<{ $color: string }>`
  display: inline-flex;
  align-items: center;
  gap: 6px;

  &::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: ${({ $color }) => $color};
  }
`;

const List = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

const rowStyles = `
  appearance: none;
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  box-sizing: border-box;
  padding: 10px 12px;
  border-radius: 8px;
  color: inherit;
  cursor: pointer;
  font: inherit;
  text-align: left;
  text-decoration: none;
`;

const LinkRow = styled.a<{ $muted?: boolean }>`
  ${rowStyles}
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  opacity: ${({ $muted }) => ($muted ? 0.6 : 1)};

  &:hover,
  &:focus-visible {
    border-color: ${({ theme }) => theme.colors.gold};
    outline: none;
  }
`;

const ButtonRow = styled.button`
  ${rowStyles}
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};

  &:disabled {
    cursor: default;
    opacity: 0.55;
  }

  &:not(:disabled):hover,
  &:not(:disabled):focus-visible {
    border-color: ${({ theme }) => theme.colors.gold};
    outline: none;
  }
`;

const RowText = styled.span`
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;

  small {
    color: ${({ theme }) => theme.colors.textMuted};
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
`;

const Empty = styled.p`
  margin: 0;
  color: ${({ theme }) => theme.colors.textMuted};
`;

const Outcome = styled.pre<{ $error: boolean }>`
  margin: 8px 0 0;
  padding: 8px 10px;
  border-radius: 8px;
  background: ${({ $error, theme }) => ($error ? theme.colors.errorBg : theme.colors.goldBg)};
  color: ${({ $error, theme }) => ($error ? theme.colors.errorOnSurface : theme.colors.textSecondary)};
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-word;
`;

function formatAge(observedAt: string, nowMs: number): string {
  const observed = Date.parse(observedAt);
  if (!Number.isFinite(observed)) return observedAt;
  const seconds = Math.max(0, Math.round((nowMs - observed) / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  return `${Math.round(minutes / 60)}h`;
}

/**
 * Right-hand drawer with the details of one capability node: identity and
 * status, its surfaces, and its actions. Settings, credentials, preflight,
 * and setup tabs arrive with the managed-plugin panels; until then the
 * panel is the durable home for what the flyout shows in brief.
 */
export function CapabilityPanel() {
  const { t } = useSkulkTranslation();
  const theme = useTheme() as Theme;
  const dispatch = useAppDispatch();
  const open = useAppSelector((s) => s.ui.capabilityPanelOpen);
  const activeTab = useAppSelector((s) => s.ui.capabilityPanelTab);
  const target = useAppSelector((s) => s.ui.capabilityPanelTarget);
  const width = useAppSelector((s) => s.ui.observabilityPanelWidth);
  const { capabilityNodes, topology, localNodeId } = useClusterState();
  const [outcome, setOutcome] = useState<{ text: string; error: boolean } | null>(null);
  const [inFlight, setInFlight] = useState(false);
  const targetKey = target ? `${target.hostNodeId}/${target.key}` : null;
  // The panel stays mounted across targets, so call state is scoped to the
  // target it was produced for: a target change clears it, and a completion
  // for a previous target is dropped instead of landing under the new one.
  const callTargetRef = useRef<string | null>(null);
  useEffect(() => {
    setOutcome(null);
    setInFlight(false);
    callTargetRef.current = null;
  }, [targetKey]);

  const close = useCallback(() => dispatch(uiActions.closeCapabilityPanel()), [dispatch]);
  const setWidth = useCallback(
    (next: number) => dispatch(uiActions.setObservabilityPanelWidth(next)),
    [dispatch],
  );

  const summary: CapabilityNodeSummary | null = useMemo(() => {
    if (!target) return null;
    return (
      (capabilityNodes[target.hostNodeId] ?? []).find(
        (candidate) => capabilityNodeKey(candidate) === target.key,
      ) ?? null
    );
  }, [capabilityNodes, target]);
  const hostName = target
    ? topology?.nodes[target.hostNodeId]?.friendly_name ?? target.hostNodeId.slice(-8)
    : '';
  const isLocalHost = target !== null && localNodeId !== null && localNodeId === target.hostNodeId;
  const actions = useMemo(
    () =>
      summary
        ? buildCapabilityActions(summary, {
            isLocalHost,
            hostName,
            dashboardHostname: window.location.hostname,
            t,
          })
        : [],
    [summary, isLocalHost, hostName, t],
  );

  const runCall = async (item: Extract<CapabilityActionItem, { kind: 'call' }>) => {
    if (!localNodeId || !targetKey) return;
    const startedFor = targetKey;
    callTargetRef.current = startedFor;
    setInFlight(true);
    setOutcome(null);
    let next: { text: string; error: boolean };
    try {
      const result = await runDescriptorAction(localNodeId, item.capabilityId, item.payload);
      next =
        result.ok === false
          ? { text: result.error?.message ?? result.error?.code ?? 'error', error: true }
          : { text: JSON.stringify(result.result ?? null, null, 2), error: false };
    } catch (error: unknown) {
      next = { text: error instanceof Error ? error.message : String(error), error: true };
    }
    if (callTargetRef.current !== startedFor) return;
    setOutcome(next);
    setInFlight(false);
  };

  const title = summary ? capabilityNodeTitle(summary) : t('capabilityPanel.title', 'Capability node');
  const tabLabel = (tab: CapabilityPanelTab) =>
    tab === 'overview'
      ? t('capabilityPanel.tabs.overview', 'Overview')
      : tab === 'surfaces'
        ? t('capabilityPanel.tabs.surfaces', 'Surfaces')
        : t('capabilityPanel.tabs.actions', 'Actions');
  const nowMs = Date.now();

  return (
    <RightDrawer
      ariaLabel={t('capabilityPanel.panelAria', 'Capability node panel')}
      closeLabel={t('capabilityPanel.closePanel', 'Close capability panel')}
      id="capability-panel"
      maxWidth={OBSERVABILITY_WIDTH_MAX}
      minWidth={OBSERVABILITY_WIDTH_MIN}
      onClose={close}
      onWidthChange={setWidth}
      open={open}
      resizeLabel={t('capabilityPanel.resizePanel', 'Resize capability panel')}
      title={title}
      width={width}
    >
      <DrawerTabBar role="tablist" aria-label={t('capabilityPanel.views', 'Capability node views')}>
        {TAB_ORDER.map((tab) => (
          <DrawerTabButton
            $active={activeTab === tab}
            aria-controls={`capability-panel-${tab}`}
            aria-selected={activeTab === tab}
            id={`capability-tab-${tab}`}
            key={tab}
            onClick={() => dispatch(uiActions.setCapabilityPanelTab(tab))}
            role="tab"
          >
            {tabLabel(tab)}
          </DrawerTabButton>
        ))}
      </DrawerTabBar>
      <DrawerBody aria-labelledby={`capability-tab-${activeTab}`} id={`capability-panel-${activeTab}`} role="tabpanel">
        <Scroll>
          {!summary ? (
            <Empty>
              {t(
                'capabilityPanel.gone',
                'This capability node is no longer reported by its host.',
              )}
            </Empty>
          ) : activeTab === 'overview' ? (
            <Grid>
              <dt>{t('capabilityPanel.field.status', 'Status')}</dt>
              <dd>
                <Status $color={satelliteColor(capabilityNodeHealth(summary), theme)}>
                  {satelliteStatusLabel(summary, t)}
                </Status>
              </dd>
              <dt>{t('capabilityPanel.field.host', 'Host')}</dt>
              <dd>{hostName}</dd>
              <dt>{t('capabilityPanel.field.plugin', 'Plugin')}</dt>
              <dd>{summary.pluginId}</dd>
              <dt>{t('capabilityPanel.field.node', 'Node')}</dt>
              <dd>{summary.nodeId}</dd>
              <dt>{t('capabilityPanel.field.bundle', 'Bundle')}</dt>
              <dd>
                {summary.bundleId} {summary.version}
              </dd>
              <dt>{t('capabilityPanel.field.owner', 'Owner')}</dt>
              <dd>
                {summary.ownerAvailable
                  ? t('capabilityPanel.owner.available', 'Available')
                  : t('capabilityPanel.owner.unavailable', 'Unavailable')}
              </dd>
              <dt>{t('capabilityPanel.field.operations', 'Operations')}</dt>
              <dd>{summary.operationsActive}</dd>
              <dt>{t('capabilityPanel.field.observed', 'Last report')}</dt>
              <dd>{t('capabilityPanel.observedAgo', '{age} ago', { age: formatAge(summary.observedAt, nowMs) })}</dd>
              {!isLocalHost ? (
                <>
                  <dt>{t('capabilityPanel.field.manage', 'Manage')}</dt>
                  <dd>
                    {t(
                      'topology.capability.manageOnHostHint',
                      'Settings and actions for this node are managed from the dashboard on {host}.',
                      { host: hostName },
                    )}
                  </dd>
                </>
              ) : null}
            </Grid>
          ) : activeTab === 'surfaces' ? (
            summary.surfaces.length === 0 ? (
              <Empty>{t('capabilityPanel.noSurfaces', 'This node exposes no surfaces.')}</Empty>
            ) : (
              <List>
                {summary.surfaces.map((surface) => {
                  const resolved = resolveSurfaceUrl(surface.url, {
                    isLocalHost,
                    dashboardHostname: window.location.hostname,
                  });
                  if (!resolved.reachable) {
                    return (
                      <ButtonRow disabled key={surface.surfaceId} type="button">
                        <FiExternalLink aria-hidden size={16} />
                        <RowText>
                          <span>{surface.title}</span>
                          <small>
                            {t('topology.capability.surfaceOnHostOnly', 'Reachable only from a browser running on {host}', {
                              host: hostName,
                            })}
                          </small>
                        </RowText>
                      </ButtonRow>
                    );
                  }
                  return (
                    <LinkRow
                      $muted={!surface.ready}
                      href={resolved.url}
                      key={surface.surfaceId}
                      rel="noopener noreferrer"
                      target="_blank"
                    >
                      <FiExternalLink aria-hidden size={16} />
                      <RowText>
                        <span>{surface.title}</span>
                        <small>
                          {surface.ready
                            ? resolved.url
                            : t('topology.capability.surfaceNotReady', 'Not answering yet')}
                        </small>
                      </RowText>
                    </LinkRow>
                  );
                })}
              </List>
            )
          ) : (
            <>
              {actions.filter((item) => item.kind === 'call' || item.kind === 'open-link').length === 0 ? (
                <Empty>{t('capabilityPanel.noActions', 'This node declares no actions.')}</Empty>
              ) : (
                <List>
                  {actions.map((item) => {
                    if (item.kind === 'open-link') {
                      if (!item.reachable) {
                        return (
                          <ButtonRow disabled key={item.id} type="button">
                            <FiExternalLink aria-hidden size={16} />
                            <RowText>
                              <span>{item.title}</span>
                              <small>
                                {t('topology.capability.surfaceOnHostOnly', 'Reachable only from a browser running on {host}', {
                                  host: hostName,
                                })}
                              </small>
                            </RowText>
                          </ButtonRow>
                        );
                      }
                      return (
                        <LinkRow
                          $muted={!item.ready}
                          href={item.url}
                          key={item.id}
                          rel="noopener noreferrer"
                          target="_blank"
                        >
                          <FiExternalLink aria-hidden size={16} />
                          <RowText>
                            <span>{item.title}</span>
                            <small>{item.url}</small>
                          </RowText>
                        </LinkRow>
                      );
                    }
                    if (item.kind === 'call') {
                      return (
                        <ButtonRow
                          disabled={!item.enabled || inFlight}
                          key={item.id}
                          onClick={() => void runCall(item)}
                          type="button"
                        >
                          <FiPlay aria-hidden size={16} />
                          <RowText>
                            <span>{item.title}</span>
                            <small>
                              {item.enabled
                                ? item.capabilityId
                                : t('topology.capability.callNeedsHost', 'Runs only from the dashboard on {host}', {
                                    host: hostName,
                                  })}
                            </small>
                          </RowText>
                        </ButtonRow>
                      );
                    }
                    return null;
                  })}
                </List>
              )}
              {outcome ? <Outcome $error={outcome.error}>{outcome.text}</Outcome> : null}
            </>
          )}
        </Scroll>
      </DrawerBody>
    </RightDrawer>
  );
}
