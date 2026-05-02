import { createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { NavRoute } from '../../components/layout/HeaderNav';
import type { ThemeName } from '../../theme';

/** Tabs available inside the observability panel. */
export type ObservabilityTab = 'live' | 'node' | 'traces';

/** Default width of the right-side observability panel, in pixels. */
const OBSERVABILITY_WIDTH_DEFAULT = 560;
/**
 * Hard floor on the panel width — narrower than this and the content stops
 * being legible. Exported because the resize handler clamps during live drag
 * preview, not just at commit time, so the live width and the persisted width
 * share one range.
 */
export const OBSERVABILITY_WIDTH_MIN = 360;
/** Hard ceiling — any wider and the panel would consume the whole viewport. */
export const OBSERVABILITY_WIDTH_MAX = 1200;

/**
 * Subset of state persisted to sessionStorage. Theme and observability panel
 * width live in localStorage instead so they survive across sessions —
 * everything else uses sessionStorage so a fresh tab starts with sensible
 * defaults rather than inheriting stale state.
 */
const SESSION_PERSIST_KEY = 'skulk-ui';
const THEME_STORAGE_KEY = 'skulk-theme';
const OBSERVABILITY_WIDTH_KEY = 'skulk-observability-panel-width';

export interface UIState {
  activeRoute: NavRoute;
  panelOpen: boolean;
  historyPanelOpen: boolean;
  chatScrollTop: number;
  /** Message IDs with thinking expanded, keyed by conversation ID. */
  expandedThinking: Record<string, string[]>;
  /** Active color theme. Persisted to localStorage. */
  theme: ThemeName;

  observabilityPanelOpen: boolean;
  observabilityActiveTab: ObservabilityTab;
  /** Panel width in pixels. Persisted to localStorage. */
  observabilityPanelWidth: number;
  observabilitySelectedNodeId: string | null;
}

function clampPanelWidth(value: number): number {
  if (Number.isNaN(value)) return OBSERVABILITY_WIDTH_DEFAULT;
  return Math.max(
    OBSERVABILITY_WIDTH_MIN,
    Math.min(OBSERVABILITY_WIDTH_MAX, Math.round(value)),
  );
}

function loadPersistedTheme(): ThemeName {
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

function loadPersistedObservabilityWidth(): number {
  if (typeof window === 'undefined') return OBSERVABILITY_WIDTH_DEFAULT;
  try {
    const raw = window.localStorage.getItem(OBSERVABILITY_WIDTH_KEY);
    if (raw == null) return OBSERVABILITY_WIDTH_DEFAULT;
    const parsed = Number.parseInt(raw, 10);
    if (Number.isFinite(parsed)) return clampPanelWidth(parsed);
  } catch {
    /* ignore */
  }
  return OBSERVABILITY_WIDTH_DEFAULT;
}

/** Loads the sessionStorage subset, falling back to defaults on any failure. */
function loadSessionState(): Partial<UIState> {
  if (typeof window === 'undefined') return {};
  try {
    const raw = window.sessionStorage.getItem(SESSION_PERSIST_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Partial<UIState>;
    return parsed;
  } catch {
    return {};
  }
}

function defaultState(): UIState {
  return {
    activeRoute: 'cluster',
    panelOpen: true,
    historyPanelOpen: true,
    chatScrollTop: 0,
    expandedThinking: {},
    theme: loadPersistedTheme(),
    observabilityPanelOpen: false,
    observabilityActiveTab: 'live',
    observabilityPanelWidth: loadPersistedObservabilityWidth(),
    observabilitySelectedNodeId: null,
  };
}

/** Initial state merges defaults with the sessionStorage subset. */
function initialState(): UIState {
  const session = loadSessionState();
  return { ...defaultState(), ...session };
}

const slice = createSlice({
  name: 'ui',
  initialState: initialState(),
  reducers: {
    setActiveRoute(state, action: PayloadAction<NavRoute>) {
      state.activeRoute = action.payload;
    },
    setPanelOpen(state, action: PayloadAction<boolean>) {
      state.panelOpen = action.payload;
    },
    togglePanel(state) {
      state.panelOpen = !state.panelOpen;
    },
    toggleHistoryPanel(state) {
      state.historyPanelOpen = !state.historyPanelOpen;
    },
    setChatScrollTop(state, action: PayloadAction<number>) {
      state.chatScrollTop = action.payload;
    },
    setExpandedThinking(
      state,
      action: PayloadAction<{ conversationId: string; messageIds: string[] }>,
    ) {
      state.expandedThinking[action.payload.conversationId] = action.payload.messageIds;
    },
    setTheme(state, action: PayloadAction<ThemeName>) {
      state.theme = action.payload;
    },
    toggleTheme(state) {
      state.theme = state.theme === 'dark' ? 'light' : 'dark';
    },
    openObservability(
      state,
      action: PayloadAction<{ tab?: ObservabilityTab; nodeId?: string } | undefined>,
    ) {
      state.observabilityPanelOpen = true;
      const payload = action.payload;
      if (payload?.tab) state.observabilityActiveTab = payload.tab;
      // Remember the node only when one is explicitly given; reopening the
      // panel without a nodeId should return to the previously focused node.
      if (payload?.nodeId) state.observabilitySelectedNodeId = payload.nodeId;
    },
    closeObservability(state) {
      state.observabilityPanelOpen = false;
    },
    setObservabilityTab(state, action: PayloadAction<ObservabilityTab>) {
      state.observabilityActiveTab = action.payload;
    },
    setObservabilityPanelWidth(state, action: PayloadAction<number>) {
      state.observabilityPanelWidth = clampPanelWidth(action.payload);
    },
    setObservabilitySelectedNodeId(state, action: PayloadAction<string | null>) {
      state.observabilitySelectedNodeId = action.payload;
    },
  },
});

export const uiSliceReducer = slice.reducer;
export const uiActions = slice.actions;

/**
 * Persistence side-effects. Wires up two storage layers:
 *  - Theme + observability panel width → localStorage (cross-session).
 *  - Everything else → sessionStorage (per-tab).
 *
 * Called once from the store entrypoint after the store is created. Returns
 * an unsubscribe function for tests; in app code we never unsubscribe.
 */
export function subscribeUIPersistence(
  store: { subscribe: (listener: () => void) => () => void; getState: () => { ui: UIState } },
): () => void {
  if (typeof window === 'undefined') return () => {};
  let lastTheme: ThemeName | null = null;
  let lastWidth: number | null = null;
  let lastSessionJson: string | null = null;

  return store.subscribe(() => {
    const ui = store.getState().ui;

    // Theme — write only when it changes to avoid hammering localStorage.
    if (ui.theme !== lastTheme) {
      lastTheme = ui.theme;
      try {
        window.localStorage.setItem(THEME_STORAGE_KEY, ui.theme);
      } catch {
        /* ignore */
      }
    }

    // Observability panel width — same idea.
    if (ui.observabilityPanelWidth !== lastWidth) {
      lastWidth = ui.observabilityPanelWidth;
      try {
        window.localStorage.setItem(
          OBSERVABILITY_WIDTH_KEY,
          String(ui.observabilityPanelWidth),
        );
      } catch {
        /* ignore */
      }
    }

    // Session subset — JSON-stringify the partialize set; only write on change.
    const sessionSubset = {
      activeRoute: ui.activeRoute,
      panelOpen: ui.panelOpen,
      historyPanelOpen: ui.historyPanelOpen,
      chatScrollTop: ui.chatScrollTop,
      expandedThinking: ui.expandedThinking,
      observabilityPanelOpen: ui.observabilityPanelOpen,
      observabilityActiveTab: ui.observabilityActiveTab,
      observabilitySelectedNodeId: ui.observabilitySelectedNodeId,
    };
    const json = JSON.stringify(sessionSubset);
    if (json !== lastSessionJson) {
      lastSessionJson = json;
      try {
        window.sessionStorage.setItem(SESSION_PERSIST_KEY, json);
      } catch {
        /* ignore */
      }
    }
  });
}
