import { create } from 'zustand';
import { devtools, persist, createJSONStorage } from 'zustand/middleware';
import type { NavRoute } from '../components/layout/HeaderNav';

export interface UIState {
  activeRoute: NavRoute;
  panelOpen: boolean;
  chatScrollTop: number;
  /** Message IDs with thinking expanded, keyed by conversation ID */
  expandedThinking: Record<string, string[]>;

  setActiveRoute: (route: NavRoute) => void;
  setPanelOpen: (open: boolean) => void;
  togglePanel: () => void;
  setChatScrollTop: (pos: number) => void;
  setExpandedThinking: (conversationId: string, messageIds: string[]) => void;
}

export const useUIStore = create<UIState>()(
  devtools(
  persist(
    (set) => ({
      activeRoute: 'cluster',
      panelOpen: true,
      chatScrollTop: 0,
      expandedThinking: {},

      setActiveRoute: (route) => set({ activeRoute: route }),
      setPanelOpen: (open) => set({ panelOpen: open }),
      togglePanel: () => set((s) => ({ panelOpen: !s.panelOpen })),
      setChatScrollTop: (pos) => set({ chatScrollTop: pos }),
      setExpandedThinking: (conversationId, messageIds) =>
        set((s) => ({
          expandedThinking: { ...s.expandedThinking, [conversationId]: messageIds },
        })),
    }),
    {
      name: 'skulk-ui',
      storage: createJSONStorage(() => sessionStorage),
    },
  ),
  { name: 'UIStore' },
  ),
);
