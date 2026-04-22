import { create } from 'zustand';
import { devtools, persist, createJSONStorage } from 'zustand/middleware';
import type { NavRoute } from '../components/layout/HeaderNav';
import type { ThemeName } from '../theme';

const THEME_STORAGE_KEY = 'skulk-theme';

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

  setActiveRoute: (route: NavRoute) => void;
  setSelectedTraceTaskId: (taskId: string | null) => void;
  setPanelOpen: (open: boolean) => void;
  togglePanel: () => void;
  toggleHistoryPanel: () => void;
  setChatScrollTop: (pos: number) => void;
  setExpandedThinking: (conversationId: string, messageIds: string[]) => void;
  setTheme: (name: ThemeName) => void;
  toggleTheme: () => void;
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
    }),
    {
      name: 'skulk-ui',
      storage: createJSONStorage(() => sessionStorage),
      // Theme lives in localStorage so it survives across sessions; exclude it from
      // the sessionStorage-backed persist block.
      partialize: (state) => ({
        activeRoute: state.activeRoute,
        selectedTraceTaskId: state.selectedTraceTaskId,
        panelOpen: state.panelOpen,
        historyPanelOpen: state.historyPanelOpen,
        chatScrollTop: state.chatScrollTop,
        expandedThinking: state.expandedThinking,
      }),
    },
  ),
  { name: 'UIStore' },
  ),
);
