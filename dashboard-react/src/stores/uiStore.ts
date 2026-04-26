import { create } from 'zustand';
import { devtools, persist, createJSONStorage } from 'zustand/middleware';
import type { NavRoute } from '../components/layout/HeaderNav';
import type { ThemeName } from '../theme';

const THEME_STORAGE_KEY = 'skulk-theme';
const OBSERVABILITY_WIDTH_KEY = 'skulk-observability-panel-width';
/** Default width of the right-side observability panel, in pixels. */
const OBSERVABILITY_WIDTH_DEFAULT = 560;
/**
 * Hard floor on the panel width — narrower than this and the content stops being
 * legible. Exported because the resize handler clamps during live drag preview, not
 * just at commit time, so the live width and the persisted width share one range.
 */
export const OBSERVABILITY_WIDTH_MIN = 360;
/**
 * Hard ceiling — any wider and the panel would consume the whole viewport on most
 * screens. Exported for the same reason as the min.
 */
export const OBSERVABILITY_WIDTH_MAX = 1200;

/** Tabs available inside the observability panel. */
export type ObservabilityTab = 'live' | 'node' | 'traces';

/** Read the persisted theme preference from localStorage, falling back to OS preference. */
function loadInitialTheme(): ThemeName {
  if (typeof window === 'undefined') return 'dark';
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
  } catch {
    /* ignore */
  }
  try {
    if (window.matchMedia?.('(prefers-color-scheme: light)').matches) return 'light';
  } catch {
    /* ignore */
  }
  return 'dark';
}

function persistTheme(name: ThemeName): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, name);
  } catch {
    /* ignore */
  }
}

function clampPanelWidth(value: number): number {
  if (Number.isNaN(value)) return OBSERVABILITY_WIDTH_DEFAULT;
  return Math.max(OBSERVABILITY_WIDTH_MIN, Math.min(OBSERVABILITY_WIDTH_MAX, Math.round(value)));
}

/** Read the persisted observability panel width from localStorage, clamped to [min, max]. */
function loadInitialObservabilityWidth(): number {
  if (typeof window === 'undefined') return OBSERVABILITY_WIDTH_DEFAULT;
  try {
    const raw = window.localStorage.getItem(OBSERVABILITY_WIDTH_KEY);
    if (raw === null) return OBSERVABILITY_WIDTH_DEFAULT;
    const parsed = Number.parseInt(raw, 10);
    if (Number.isFinite(parsed)) return clampPanelWidth(parsed);
  } catch {
    /* ignore */
  }
  return OBSERVABILITY_WIDTH_DEFAULT;
}

function persistObservabilityWidth(width: number): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(OBSERVABILITY_WIDTH_KEY, String(clampPanelWidth(width)));
  } catch {
    /* ignore */
  }
}

export interface UIState {
  activeRoute: NavRoute;
  selectedTraceTaskId: string | null;
  panelOpen: boolean;
  historyPanelOpen: boolean;
  chatScrollTop: number;
  /** Message IDs with thinking expanded, keyed by conversation ID */
  expandedThinking: Record<string, string[]>;
  /** Active color theme. Persisted to localStorage outside the sessionStorage `persist` block. */
  theme: ThemeName;

  /**
   * Whether the observability panel (right-side dock) is visible. The panel overlays the
   * current route's content rather than being a route itself, so the user can keep
   * working in topology / chat / model store while glancing at observability data.
   */
  observabilityPanelOpen: boolean;
  /** Active tab inside the observability panel. */
  observabilityActiveTab: ObservabilityTab;
  /**
   * Width of the observability panel in pixels. Persisted to localStorage outside the
   * sessionStorage `persist` block so it survives across sessions — operators settle
   * on a width that suits their displays and shouldn't have to redo it on every refresh.
   */
  observabilityPanelWidth: number;
  /**
   * Node ID currently focused in the panel's "Node" tab. Independent of `panelOpen`
   * so closing the panel and reopening it returns to the same node.
   */
  observabilitySelectedNodeId: string | null;

  setActiveRoute: (route: NavRoute) => void;
  setSelectedTraceTaskId: (taskId: string | null) => void;
  setPanelOpen: (open: boolean) => void;
  togglePanel: () => void;
  toggleHistoryPanel: () => void;
  setChatScrollTop: (pos: number) => void;
  setExpandedThinking: (conversationId: string, messageIds: string[]) => void;
  setTheme: (name: ThemeName) => void;
  toggleTheme: () => void;

  /**
   * Open the observability panel. If a tab is provided the panel switches to it; if a
   * nodeId is provided (typical when the user clicks the per-node bug icon) the panel
   * additionally selects the Node tab and remembers the node selection.
   */
  openObservability: (tab?: ObservabilityTab, nodeId?: string) => void;
  closeObservability: () => void;
  setObservabilityTab: (tab: ObservabilityTab) => void;
  /**
   * Set the observability panel width. Clamped to the safe range and persisted to
   * localStorage outside the sessionStorage `persist` block.
   */
  setObservabilityPanelWidth: (width: number) => void;
  setObservabilitySelectedNodeId: (nodeId: string | null) => void;
}

export const useUIStore = create<UIState>()(
  devtools(
  persist(
    (set) => ({
      activeRoute: 'cluster',
      selectedTraceTaskId: null,
      panelOpen: true,
      historyPanelOpen: true,
      chatScrollTop: 0,
      expandedThinking: {},
      theme: loadInitialTheme(),

      observabilityPanelOpen: false,
      observabilityActiveTab: 'live',
      observabilityPanelWidth: loadInitialObservabilityWidth(),
      observabilitySelectedNodeId: null,

      setActiveRoute: (route) => set({ activeRoute: route }),
      setSelectedTraceTaskId: (taskId) => set({ selectedTraceTaskId: taskId }),
      setPanelOpen: (open) => set({ panelOpen: open }),
      togglePanel: () => set((s) => ({ panelOpen: !s.panelOpen })),
      toggleHistoryPanel: () => set((s) => ({ historyPanelOpen: !s.historyPanelOpen })),
      setChatScrollTop: (pos) => set({ chatScrollTop: pos }),
      setExpandedThinking: (conversationId, messageIds) =>
        set((s) => ({
          expandedThinking: { ...s.expandedThinking, [conversationId]: messageIds },
        })),
      setTheme: (name) => {
        persistTheme(name);
        set({ theme: name });
      },
      toggleTheme: () =>
        set((s) => {
          const next: ThemeName = s.theme === 'dark' ? 'light' : 'dark';
          persistTheme(next);
          return { theme: next };
        }),

      openObservability: (tab, nodeId) =>
        set((s) => ({
          observabilityPanelOpen: true,
          // If the caller named a tab, switch to it; otherwise — and this is the
          // case for the toolbar icon — keep whatever tab the operator last left
          // open. Default lands on 'live' on first ever open via the initial state.
          observabilityActiveTab: tab ?? s.observabilityActiveTab,
          // Remember the node only when one is explicitly given. Reopening the panel
          // without a nodeId should return to the previously focused node, not clear it.
          observabilitySelectedNodeId: nodeId ?? s.observabilitySelectedNodeId,
        })),
      closeObservability: () => set({ observabilityPanelOpen: false }),
      setObservabilityTab: (tab) => set({ observabilityActiveTab: tab }),
      setObservabilityPanelWidth: (width) => {
        const clamped = clampPanelWidth(width);
        persistObservabilityWidth(clamped);
        set({ observabilityPanelWidth: clamped });
      },
      setObservabilitySelectedNodeId: (nodeId) =>
        set({ observabilitySelectedNodeId: nodeId }),
    }),
    {
      name: 'skulk-ui',
      storage: createJSONStorage(() => sessionStorage),
      // Theme + observabilityPanelWidth live in localStorage so they survive
      // across sessions; everything else uses sessionStorage so a fresh tab
      // starts with sensible defaults rather than inheriting stale state.
      partialize: (state) => ({
        activeRoute: state.activeRoute,
        selectedTraceTaskId: state.selectedTraceTaskId,
        panelOpen: state.panelOpen,
        historyPanelOpen: state.historyPanelOpen,
        chatScrollTop: state.chatScrollTop,
        expandedThinking: state.expandedThinking,
        observabilityPanelOpen: state.observabilityPanelOpen,
        observabilityActiveTab: state.observabilityActiveTab,
        observabilitySelectedNodeId: state.observabilitySelectedNodeId,
      }),
    },
  ),
  { name: 'UIStore' },
  ),
);
