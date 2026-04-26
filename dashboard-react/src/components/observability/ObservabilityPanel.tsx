import { useCallback, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { FiX } from 'react-icons/fi';
import { Button } from '../common/Button';
import {
  useUIStore,
  type ObservabilityTab,
  OBSERVABILITY_WIDTH_MIN,
  OBSERVABILITY_WIDTH_MAX,
} from '../../stores/uiStore';
import { LiveTab } from './LiveTab';
import { NodeTab } from './NodeTab';
import { TracesTab } from './TracesTab';

/**
 * Right-side resizable panel that hosts every observability surface — live cluster
 * health, per-node deep dive, saved trace browsing — under one nav entry.
 *
 * Architecture decisions worth knowing:
 *
 * - The panel **overlays** the current route's content rather than replacing it.
 *   Operators in chat / model store / topology can glance at observability without
 *   losing their place.
 * - The panel is **side-docked, no backdrop**. The topology view stays interactive so
 *   the panel and the spatial cluster picture are usable simultaneously.
 * - Width is **operator-controlled** (drag the left edge) and **persisted to
 *   localStorage** outside the sessionStorage UI state — operators settle on a width
 *   and shouldn't redo it on every refresh.
 * - All panel state lives on the global Zustand store so any component can open the
 *   panel to a specific tab/node. The toolbar nav button calls `openObservability()`
 *   without args; per-node bug icons call `openObservability('node', nodeId)`.
 */

const Aside = styled.aside<{ $width: number }>`
  position: fixed;
  top: 0;
  right: 0;
  height: 100vh;
  width: ${({ $width }) => $width}px;
  background: ${({ theme }) => theme.colors.surfaceElevated};
  border-left: 1px solid ${({ theme }) => theme.colors.borderStrong};
  box-shadow: -18px 0 48px ${({ theme }) => theme.colors.shadowStrong};
  display: flex;
  flex-direction: column;
  /*
   * Z-index sits above the topology + main content but below modal dialogs.
   * The DiagnosticsDrawer it replaces used 70; we sit just below at 65 so future
   * modal work can layer on top without arguing with the panel.
   */
  z-index: 65;
`;

/**
 * Drag handle on the left edge of the panel. Sits in front of `Aside`'s left border
 * with a slightly wider hit area than its visible footprint so the cursor catches it
 * reliably.
 */
const ResizeHandle = styled.div`
  position: absolute;
  left: -4px;
  top: 0;
  bottom: 0;
  width: 8px;
  cursor: ew-resize;
  /* Subtle hover hint without being distracting. */
  &:hover {
    background: ${({ theme }) => theme.colors.goldDim};
    opacity: 0.4;
  }
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px 10px;
  gap: 12px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
`;

const Title = styled.h2`
  margin: 0;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.lg};
  color: ${({ theme }) => theme.colors.text};
`;

const TabBar = styled.div`
  display: flex;
  gap: 4px;
  padding: 8px 12px 0;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
`;

const TabButton = styled.button<{ $active: boolean }>`
  all: unset;
  cursor: pointer;
  padding: 6px 14px 8px;
  border-radius: ${({ theme }) => theme.radii.sm} ${({ theme }) => theme.radii.sm} 0 0;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ $active, theme }) => ($active ? theme.colors.gold : theme.colors.textSecondary)};
  border-bottom: 2px solid
    ${({ $active, theme }) => ($active ? theme.colors.gold : 'transparent')};
  transition: color 0.15s, border-color 0.15s;

  &:hover {
    color: ${({ theme }) => theme.colors.text};
  }

  /* Visible focus ring for keyboard users — 'all: unset' strips the default. */
  &:focus-visible {
    outline: 2px solid ${({ theme }) => theme.colors.goldDim};
    outline-offset: 2px;
  }
`;

const Body = styled.div`
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 16px 18px;
`;

const TAB_ORDER: { key: ObservabilityTab; label: string }[] = [
  { key: 'live', label: 'Live' },
  { key: 'node', label: 'Node' },
  { key: 'traces', label: 'Traces' },
];

export function ObservabilityPanel() {
  const open = useUIStore((s) => s.observabilityPanelOpen);
  const activeTab = useUIStore((s) => s.observabilityActiveTab);
  const width = useUIStore((s) => s.observabilityPanelWidth);
  const selectedNodeId = useUIStore((s) => s.observabilitySelectedNodeId);
  const setTab = useUIStore((s) => s.setObservabilityTab);
  const setWidth = useUIStore((s) => s.setObservabilityPanelWidth);
  const close = useUIStore((s) => s.closeObservability);

  // Drag-to-resize: capture pointer at the handle; resizing computes width from
  // the cursor's distance from the right edge of the viewport. We hold an Aside
  // ref instead of looking the element up via getElementById in the pointermove
  // hot path, both to avoid the lookup cost and to keep the contract local
  // (this component owns the element it mutates during drag).
  const asideRef = useRef<HTMLElement | null>(null);
  const draggingRef = useRef(false);
  const dragWidthRef = useRef<number>(width);

  // Begin a resize drag. Store-side commit happens on pointerup; in-flight
  // updates set the panel width via DOM directly to keep things smooth.
  const onResizeStart = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      event.preventDefault();
      draggingRef.current = true;
      dragWidthRef.current = width;
      // Capture the pointer so we keep getting events even if the cursor
      // briefly leaves the handle's hitbox during fast drags.
      (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    },
    [width],
  );

  useEffect(() => {
    if (!open) return;
    const onMove = (event: PointerEvent) => {
      if (!draggingRef.current) return;
      // Clamp during live preview, not just at commit time. Without this the
      // pointer leaving the viewport would set the inline width to negative or
      // wildly large values and the panel would visibly flicker off-screen
      // even though the eventual store commit clamps. Using the same range
      // here keeps the live preview and the persisted state visually identical.
      const raw = window.innerWidth - event.clientX;
      const next = Math.max(
        OBSERVABILITY_WIDTH_MIN,
        Math.min(OBSERVABILITY_WIDTH_MAX, raw),
      );
      dragWidthRef.current = next;
      // Live preview via inline style on the panel; final commit lands on up.
      if (asideRef.current) asideRef.current.style.width = `${next}px`;
    };
    const onUp = () => {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      // Clear the inline style BEFORE committing the store update. Inline
      // styles beat styled-components' generated CSS rules on specificity, so
      // leaving an inline width here would silently override every subsequent
      // state-driven width change (including from `setObservabilityPanelWidth`
      // and from any other component that touches the store). Clearing first
      // hands width control back to the styled-component template literal.
      if (asideRef.current) asideRef.current.style.width = '';
      setWidth(dragWidthRef.current);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
    };
  }, [open, setWidth]);

  // Esc closes the panel — operators expect this for any modal-like surface.
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, close]);

  if (!open) return null;

  return (
    <Aside
      $width={width}
      ref={asideRef}
      id="observability-panel"
      aria-label="Observability panel"
    >
      <ResizeHandle
        onPointerDown={onResizeStart}
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize observability panel"
      />
      <Header>
        <Title>Observability</Title>
        <Button variant="ghost" size="sm" onClick={close} aria-label="Close observability panel">
          <FiX size={16} />
        </Button>
      </Header>
      <TabBar role="tablist" aria-label="Observability views">
        {TAB_ORDER.map((tab) => (
          <TabButton
            key={tab.key}
            $active={activeTab === tab.key}
            role="tab"
            aria-selected={activeTab === tab.key}
            aria-controls={`observability-panel-${tab.key}`}
            id={`observability-tab-${tab.key}`}
            onClick={() => setTab(tab.key)}
          >
            {tab.label}
          </TabButton>
        ))}
      </TabBar>
      <Body
        role="tabpanel"
        id={`observability-panel-${activeTab}`}
        aria-labelledby={`observability-tab-${activeTab}`}
      >
        {activeTab === 'live' && <LiveTab />}
        {activeTab === 'node' && <NodeTab nodeId={selectedNodeId} />}
        {activeTab === 'traces' && <TracesTab />}
      </Body>
    </Aside>
  );
}
