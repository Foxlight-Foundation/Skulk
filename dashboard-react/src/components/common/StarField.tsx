import { useMemo, type CSSProperties } from 'react';
import styled from 'styled-components';

/**
 * Props for the decorative dashboard star field.
 */
export interface StarFieldProps {
  /** Number of small twinkling stars to render. */
  density?: number;
  /** Number of brighter halo stars to render. */
  brightCount?: number;
  /**
   * Deterministic seed used by tests and stories. Production mounts can
   * leave this undefined for fresh randomness on each load.
   */
  seed?: number;
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

const Star = styled.span`
  position: absolute;
  border-radius: 50%;
  opacity: 0;
  background: radial-gradient(
    circle,
    rgba(214, 229, 255, 0.95) 0%,
    rgba(214, 229, 255, 0.42) 35%,
    rgba(214, 229, 255, 0) 70%
  );
  filter: brightness(var(--base-b, 0.82));
  animation:
    skulk-starfield-appear 1.5s var(--delay, 0s) ease-out forwards,
    skulk-starfield-shimmer var(--duration, 4s) var(--shimmer-start, 1.5s) ease-in-out infinite;

  @keyframes skulk-starfield-appear {
    from {
      opacity: 0;
      filter: brightness(0);
      transform: scale(0.6);
    }
    to {
      opacity: 1;
      filter: brightness(var(--base-b, 0.82));
      transform: scale(1);
    }
  }

  @keyframes skulk-starfield-shimmer {
    0% {
      filter: brightness(var(--base-b, 0.82));
      transform: scale(1);
    }
    30% {
      filter: brightness(var(--peak-b, 1.28));
      transform: scale(1.12);
    }
    55% {
      filter: brightness(var(--dim-b, 0.5));
      transform: scale(0.9);
    }
    80% {
      filter: brightness(var(--base-b, 0.82));
      transform: scale(1.04);
    }
    100% {
      filter: brightness(var(--base-b, 0.82));
      transform: scale(1);
    }
  }
`;

/**
 * Tiny seeded PRNG for stable decorative layouts in tests.
 */
const mulberry32 = (seed: number): (() => number) => {
  let value = seed >>> 0;
  return () => {
    value |= 0;
    value = (value + 0x6d2b79f5) | 0;
    let t = Math.imul(value ^ (value >>> 15), 1 | value);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

interface StarSpec {
  id: string;
  left: number;
  top: number;
  size: number;
  duration: number;
  delay: number;
  baseBrightness: number;
  glow?: string;
}

const buildStars = (density: number, brightCount: number, seed?: number): StarSpec[] => {
  const rand = seed !== undefined ? mulberry32(seed) : Math.random;
  const stars: StarSpec[] = [];

  for (let index = 0; index < density; index++) {
    const x = rand() * 100;
    const y = Math.pow(rand(), 1.4) * 95;
    const size = rand() < 0.06 ? 6 + rand() * 3 : 4 + rand() * 3;
    const baseBrightness = y < 50 ? 0.5 + rand() * 0.5 : 0.3 + rand() * 0.3;
    stars.push({
      id: `star-${index}`,
      left: x,
      top: y,
      size,
      duration: 3 + rand() * 7,
      delay: rand() * 6,
      baseBrightness: Number(baseBrightness.toFixed(2)),
    });
  }

  const brightPositions: Array<[number, number]> = [
    [20, 10],
    [41, 7],
    [58, 14],
    [72, 12],
    [81, 20],
    [10, 26],
    [90, 17],
  ];

  for (let index = 0; index < Math.min(brightCount, brightPositions.length); index++) {
    const position = brightPositions[index];
    if (!position) continue;
    const [left, top] = position;
    stars.push({
      id: `bright-${index}`,
      left,
      top,
      size: 5 + rand() * 4,
      duration: 2.5 + rand() * 4,
      delay: rand() * 4,
      baseBrightness: Number((0.82 + rand() * 0.18).toFixed(2)),
      glow: `0 0 ${4 + rand() * 6}px rgba(214, 229, 255, 0.55)`,
    });
  }

  return stars;
};

/**
 * Decorative fixed star layer used behind the dark dashboard.
 */
export const StarField: React.FC<StarFieldProps> = ({
  density = 260,
  brightCount = 7,
  seed,
  className,
}) => {
  const stars = useMemo(() => buildStars(density, brightCount, seed), [density, brightCount, seed]);

  return (
    <Layer className={className} aria-hidden="true">
      {stars.map((star) => {
        const dimBrightness = Math.max(0.5, star.baseBrightness * 0.7);
        const peakBrightness = Math.min(1.35, star.baseBrightness * 1.3);
        return (
          <Star
            key={star.id}
            style={
              {
                left: `${star.left}%`,
                top: `${star.top}%`,
                width: `${star.size}px`,
                height: `${star.size}px`,
                boxShadow: star.glow,
                '--duration': `${star.duration}s`,
                '--delay': `${star.delay}s`,
                '--shimmer-start': `${star.delay + 1.5}s`,
                '--base-b': `${star.baseBrightness}`,
                '--dim-b': `${dimBrightness}`,
                '--peak-b': `${peakBrightness}`,
              } as CSSProperties
            }
          />
        );
      })}
    </Layer>
  );
};
