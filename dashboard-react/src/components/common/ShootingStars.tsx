import { useEffect, useRef } from 'react';
import styled from 'styled-components';

/**
 * Props for the decorative dashboard shooting-star layer.
 */
export interface ShootingStarsProps {
  /** Minimum interval between stars, in milliseconds. */
  minIntervalMs?: number;
  /** Maximum interval between stars, in milliseconds. */
  maxIntervalMs?: number;
  /** Delay before the first star appears, in milliseconds. */
  initialDelayMs?: number;
  className?: string;
}

const Layer = styled.div`
  position: fixed;
  inset: 0 0 auto 0;
  height: 42%;
  z-index: 0;
  pointer-events: none;
  overflow: hidden;

  @media (prefers-reduced-motion: reduce) {
    display: none;
  }
`;

const prefersReducedMotion = (): boolean => {
  if (typeof window === 'undefined' || !window.matchMedia) return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
};

/**
 * Spawns one animated streak and returns the created element for cleanup.
 */
const spawnShootingStar = (host: HTMLDivElement): HTMLDivElement | null => {
  const startY = 3 + Math.random() * 35;
  const startX = 2 + Math.random() * 32;
  const tailLength = 80 + Math.random() * 220;
  const angle = 15 + Math.random() * 40;
  const travel = 280 + Math.random() * 180;
  const duration = 500 + Math.random() * 700;
  const brightness = 0.55 + Math.random() * 0.45;
  const thickness = 0.8 + Math.random() * 0.9;

  const element = document.createElement('div');
  element.style.cssText = `
    position: absolute;
    top: ${startY}%;
    left: ${startX}%;
    width: ${tailLength}px;
    height: ${thickness}px;
    background: linear-gradient(to right,
      rgba(255,255,255,0) 0%,
      rgba(214,229,255,${brightness * 0.28}) 25%,
      rgba(241,247,255,${brightness * 0.72}) 65%,
      rgba(255,255,255,${brightness}) 100%);
    border-radius: 9999px;
    transform: rotate(${angle}deg) translateX(0px);
    transform-origin: 0% 50%;
    pointer-events: none;
    z-index: 1;
  `;
  host.appendChild(element);

  if (typeof element.animate !== 'function') {
    setTimeout(() => element.remove(), duration);
    return element;
  }

  const animation = element.animate(
    [
      { opacity: 0, transform: `rotate(${angle}deg) translateX(-${tailLength}px)` },
      { opacity: brightness, transform: `rotate(${angle}deg) translateX(0px)`, offset: 0.05 },
      { opacity: brightness, transform: `rotate(${angle}deg) translateX(${travel}px)`, offset: 0.82 },
      { opacity: 0, transform: `rotate(${angle}deg) translateX(${travel + tailLength * 0.3}px)` },
    ],
    { duration, easing: 'linear', fill: 'forwards' },
  );
  animation.onfinish = () => element.remove();
  return element;
};

/**
 * Periodically spawns decorative shooting stars in the upper sky band.
 */
export const ShootingStars: React.FC<ShootingStarsProps> = ({
  minIntervalMs = 2500,
  maxIntervalMs = 9000,
  initialDelayMs = 1800,
  className,
}) => {
  const hostRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (prefersReducedMotion()) return;
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
      while (host.firstChild) host.removeChild(host.firstChild);
    };
  }, [minIntervalMs, maxIntervalMs, initialDelayMs]);

  return <Layer ref={hostRef} className={className} aria-hidden="true" />;
};
