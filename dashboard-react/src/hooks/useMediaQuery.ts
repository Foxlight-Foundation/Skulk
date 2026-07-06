import { useCallback, useSyncExternalStore } from 'react';

/**
 * React to a CSS media query from component logic.
 *
 * Returns whether `query` currently matches, re-rendering on change (for
 * example when the viewport crosses a breakpoint or the device rotates).
 * Implemented on `useSyncExternalStore` because `matchMedia` is an external
 * store: no effect-time setState, and the subscription follows the query.
 * `matchMedia` is treated as potentially throwing (the uiSlice convention);
 * a missing implementation, an odd runtime, or a malformed query reads as
 * a non-match rather than crashing rendering. The server snapshot is a
 * stable `false`.
 *
 * @param query A media query string, e.g. `'(max-width: 480px)'`.
 */
export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (onStoreChange: () => void) => {
      try {
        const list = window.matchMedia(query);
        list.addEventListener('change', onStoreChange);
        return () => list.removeEventListener('change', onStoreChange);
      } catch {
        return () => {};
      }
    },
    [query],
  );
  const getSnapshot = useCallback(() => {
    try {
      return window.matchMedia(query).matches;
    } catch {
      return false;
    }
  }, [query]);
  const getServerSnapshot = useCallback(() => false, []);
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}

/**
 * The dashboard's phone breakpoint. Below this width the header collapses to
 * logo + hamburger and navigation moves into the mobile menu sheet.
 */
export const MOBILE_BREAKPOINT_PX = 480;

/** Whether the viewport is at or below the phone breakpoint. */
export function useIsMobile(): boolean {
  return useMediaQuery(`(max-width: ${MOBILE_BREAKPOINT_PX}px)`);
}
