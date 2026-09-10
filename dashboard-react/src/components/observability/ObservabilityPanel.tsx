import { useCallback } from 'react';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import {
  uiActions,
  type ObservabilityTab,
  OBSERVABILITY_WIDTH_MIN,
  OBSERVABILITY_WIDTH_MAX,
} from '../../store/slices/uiSlice';
import { RightDrawer } from '../common/RightDrawer';
import { DrawerBody, DrawerTabBar, DrawerTabButton } from '../common/drawerParts';
import { LiveTab } from './LiveTab';
import { NodeTab } from './NodeTab';
import { TracesTab } from './TracesTab';
import { PerformanceTab } from './PerformanceTab';
import { useSkulkTranslation } from '../../i18n/tolgee';

/**
 * Right-side resizable panel that hosts every observability surface: live
 * cluster health, per-node deep dive, saved trace browsing, and performance
 * envelopes, under one nav entry.
 *
 * The drawer chrome (backdrop, slide-in, drag-resize, Escape) lives in
 * `RightDrawer` and is shared with the capability panel. Width is
 * operator-controlled and persisted to localStorage outside the
 * sessionStorage UI state, so operators settle on a width once. All panel
 * state lives on the Redux UI slice so any component can open the panel to
 * a specific tab or node: the toolbar calls `openObservability()` without
 * args; per-node actions call `openObservability('node', nodeId)`.
 */

const TAB_ORDER: { key: ObservabilityTab }[] = [
  { key: 'live' },
  { key: 'node' },
  { key: 'traces' },
  { key: 'performance' },
];

export function ObservabilityPanel() {
  const { t } = useSkulkTranslation();
  const dispatch = useAppDispatch();
  const open = useAppSelector((s) => s.ui.observabilityPanelOpen);
  const activeTab = useAppSelector((s) => s.ui.observabilityActiveTab);
  const width = useAppSelector((s) => s.ui.observabilityPanelWidth);
  const selectedNodeId = useAppSelector((s) => s.ui.observabilitySelectedNodeId);
  const setTab = (tab: ObservabilityTab) => dispatch(uiActions.setObservabilityTab(tab));
  const setWidth = useCallback(
    (next: number) => dispatch(uiActions.setObservabilityPanelWidth(next)),
    [dispatch],
  );
  const close = useCallback(() => dispatch(uiActions.closeObservability()), [dispatch]);

  return (
    <RightDrawer
      ariaLabel={t('observability.panelAria', 'Observability panel')}
      closeLabel={t('observability.closePanel', 'Close observability panel')}
      id="observability-panel"
      maxWidth={OBSERVABILITY_WIDTH_MAX}
      minWidth={OBSERVABILITY_WIDTH_MIN}
      onClose={close}
      onWidthChange={setWidth}
      open={open}
      resizeLabel={t('observability.resizePanel', 'Resize observability panel')}
      title={t('header.observability', 'Observability')}
      width={width}
    >
      <DrawerTabBar role="tablist" aria-label={t('observability.views', 'Observability views')}>
        {TAB_ORDER.map((tab) => (
          <DrawerTabButton
            key={tab.key}
            $active={activeTab === tab.key}
            role="tab"
            aria-selected={activeTab === tab.key}
            aria-controls={`observability-panel-${tab.key}`}
            id={`observability-tab-${tab.key}`}
            onClick={() => setTab(tab.key)}
          >
            {tab.key === 'live'
              ? t('observability.tabs.live', 'Live')
              : tab.key === 'node'
                ? t('observability.tabs.node', 'Node')
                : tab.key === 'traces'
                  ? t('observability.tabs.traces', 'Traces')
                  : t('observability.tabs.performance', 'Performance')}
          </DrawerTabButton>
        ))}
      </DrawerTabBar>
      <DrawerBody
        role="tabpanel"
        id={`observability-panel-${activeTab}`}
        aria-labelledby={`observability-tab-${activeTab}`}
      >
        {activeTab === 'live' && <LiveTab />}
        {activeTab === 'node' && <NodeTab nodeId={selectedNodeId} />}
        {activeTab === 'traces' && <TracesTab />}
        {activeTab === 'performance' && <PerformanceTab />}
      </DrawerBody>
    </RightDrawer>
  );
}
