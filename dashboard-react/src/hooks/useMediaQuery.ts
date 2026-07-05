import { useCallback, useSyncExternalStore } from 'react';

/**
 * React to a CSS media query from component logic.
 *
 * Returns whether `query` currently matches, re-rendering on change (for
 * example when the viewport crosses a breakpoint or the device rotates).
 * Implemented on `useSyncExternalStore` because `matchMedia` is an external
 * store: no effect-time setState, and the subscription follows the query.
 * Environments without `window.matchMedia` (tests) read as `false`.
 *
 * @param query A media query string, e.g. `'(max-width: 480px)'`.
 */
export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (onStoreChange: () => void) => {
      if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
        return () => {};
      }
      const list = window.matchMedia(query);
      list.addEventListener('change', onStoreChange);
      return () => list.removeEventListener('change', onStoreChange);
    },
    [query],
  );
  const getSnapshot = useCallback(
    () =>
      typeof window !== 'undefined' && typeof window.matchMedia === 'function'
        ? window.matchMedia(query).matches
        : false,
    [query],
  );
  return useSyncExternalStore(subscribe, getSnapshot);
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
