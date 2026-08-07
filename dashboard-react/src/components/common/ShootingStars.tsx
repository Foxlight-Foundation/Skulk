/**
 * ShootingStars - episodic shooting stars drawn over the night sky.
 *
 * Adapted from the foxlight.ai site: a positioned layer that, on a random
 * interval, spawns a short-lived linear-gradient "comet" that sweeps
 * down-right at a random angle with a fixed-length tail. Each star animates
 * with the Web Animations API (linear easing - real shooting stars don't
 * accelerate or decelerate) and removes itself on finish.
 *
 * Design notes:
 * - Stars belong to the night, so the layer renders only when the active
 *   palette declares a scene AND opts in via `sceneMeteors` (the day scene
 *   declines), and
 *   every star must be fully gone before halfway down the viewport: the
 *   layer hard-clips at 50vh and a bottom mask dissolves anything that
 *   approaches the boundary, so no trajectory can end in the content.
 * - Spawn interval is random between `minIntervalMs` and `maxIntervalMs`
 *   (default 2.5-9s) so the rhythm feels organic, never metronomic.
 * - `prefers-reduced-motion` disables the layer entirely.
 * - Cleans up the pending timeout and any in-flight star elements on
 *   unmount so the layer doesn't leak across theme switches.
 */
import { useEffect, useRef } from 'react';
import styled, { useTheme } from 'styled-components';
import type { Theme } from '../../theme';

/** Props for the `ShootingStars` layer. */
export interface ShootingStarsProps {
  /** Minimum interval between shooting stars in ms. Defaults to 2500. */
  minIntervalMs?: number;
  /** Maximum interval between shooting stars in ms. Defaults to 9000. */
  maxIntervalMs?: number;
  /** Initial delay before the first star appears. Defaults to 1800. */
  initialDelayMs?: number;
}

const Layer = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  /* Stars live in the sky's crown only: hard-clip at half the viewport and
     dissolve the last stretch so a star nearing the boundary fades out
     completely instead of hitting an edge. */
  height: 50vh;
  z-index: 0;
  pointer-events: none;
  overflow: hidden;
  mask-image: linear-gradient(180deg, rgba(0, 0, 0, 1) 0%, rgba(0, 0, 0, 1) 70%, transparent 95%);

  @media (prefers-reduced-motion: reduce) {
    display: none;
  }
`;

const prefersReducedMotion = (): boolean => {
  if (typeof window === 'undefined' || !window.matchMedia) return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
};

/**
 * Renders a random shooting star inside `host`. Returns the spawned element
 * so callers can track it for cleanup.
 */
const spawnShootingStar = (host: HTMLDivElement): HTMLDivElement | null => {
  const startY = 3 + Math.random() * 30; // top 3-33% of the 50vh layer
  const startX = 2 + Math.random() * 32; // left 2-34%
  const tailLen = 80 + Math.random() * 220; // 80-300px
  const angle = 15 + Math.random() * 40; // 15-55 degrees
  const travel = 280 + Math.random() * 180; // 280-460px
  const dur = 500 + Math.random() * 700; // 0.5-1.2s
  const bright = 0.55 + Math.random() * 0.45;
  const thick = 0.8 + Math.random() * 0.9; // 0.8-1.7px

  const el = document.createElement('div');
  el.style.cssText = `
    position: absolute;
    top: ${startY}%;
    left: ${startX}%;
    width: ${tailLen}px;
    height: ${thick}px;
    background: linear-gradient(to right,
      rgba(255,255,255,0) 0%,
      rgba(210,228,255,${bright * 0.3}) 25%,
      rgba(240,248,255,${bright * 0.75}) 65%,
      rgba(255,255,255,${bright}) 100%);
    border-radius: 9999px;
    transform: rotate(${angle}deg) translateX(0px);
    transform-origin: 0% 50%;
    pointer-events: none;
  `;
  host.appendChild(el);

  // Web Animations API - may not exist in older test environments. Fall
  // back to a plain removal if the element can't animate.
  if (typeof el.animate !== 'function') {
    setTimeout(() => el.remove(), dur);
    return el;
  }

  const anim = el.animate(
    [
      { opacity: 0, transform: `rotate(${angle}deg) translateX(-${tailLen}px)` },
      { opacity: bright, transform: `rotate(${angle}deg) translateX(0px)`, offset: 0.05 },
      {
        opacity: bright,
        transform: `rotate(${angle}deg) translateX(${travel}px)`,
        offset: 0.82,
      },
      {
        opacity: 0,
        transform: `rotate(${angle}deg) translateX(${travel + tailLen * 0.3}px)`,
      },
    ],
    { duration: dur, easing: 'linear', fill: 'forwards' },
  );
  anim.onfinish = () => el.remove();
  return el;
};

/**
 * Mounts the shooting-star layer when the active palette has a night sky.
 * Spawns stars on a random interval until unmount; respects
 * `prefers-reduced-motion`.
 */
export function ShootingStars({
  minIntervalMs = 2500,
  maxIntervalMs = 9000,
  initialDelayMs = 1800,
}: ShootingStarsProps) {
  const theme = useTheme() as Theme;
  const hostRef = useRef<HTMLDivElement | null>(null);
  const hasScene = theme.colors.scene !== 'none' && theme.colors.sceneMeteors === 'on';

  useEffect(() => {
    if (!hasScene || prefersReducedMotion()) return;
    const host = hostRef.current;
    if (!host) return;

    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;

    const scheduleNext = (): void => {
      const interval = minIntervalMs + Math.random() * (maxIntervalMs - minIntervalMs);
      timeoutId = setTimeout(() => {
        if (cancelled || !hostRef.current) return;
        spawnShootingStar(hostRef.current);
        scheduleNext();
      }, interval);
    };

    timeoutId = setTimeout(scheduleNext, initialDelayMs);

    return () => {
      cancelled = true;
      if (timeoutId !== undefined) clearTimeout(timeoutId);
      // Remove any in-flight star elements so they don't leak across a
      // theme switch.
      while (host.firstChild) host.removeChild(host.firstChild);
    };
  }, [hasScene, minIntervalMs, maxIntervalMs, initialDelayMs]);

  if (!hasScene) return null;
  return <Layer ref={hostRef} aria-hidden="true" />;
}
