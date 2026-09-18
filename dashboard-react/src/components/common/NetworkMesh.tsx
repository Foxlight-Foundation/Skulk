import styled, { useTheme } from 'styled-components';

export interface NetworkMeshProps {
  /** @deprecated Static reference mesh has a fixed node count. */
  count?: number;
  /** @deprecated Static reference mesh has fixed connections. */
  linkDistance?: number;
  /** Particle color. Defaults to `theme.colors.bgMeshNode`. */
  color?: string;
  /** Connection color. Defaults to `theme.colors.bgMeshLine`. */
  lineColor?: string;
  /** Particle radius. Default 1.5. */
  radius?: number;
  /** @deprecated The reference mesh does not animate. */
  speed?: number;
  className?: string;
}

const Mesh = styled.svg`
  position: fixed; inset: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none;
`;

/** Static, token-aware backdrop matching the supplied mesh without continuous canvas work. */
export function NetworkMesh({ color, lineColor, radius = 2, className }: NetworkMeshProps) {
  const theme = useTheme();
  return <Mesh viewBox="0 0 1440 900" preserveAspectRatio="xMidYMid slice" className={className} aria-hidden="true">
    <g stroke={lineColor ?? theme.colors.bgMeshLine} strokeWidth="1" fill="none">
      <path d="M0 120L180 40L360 160L520 60L700 180L900 90L1080 200L1260 110L1440 190" />
      <path d="M0 420L140 300L360 160M180 40L360 160L520 380L700 180L860 420L900 90M520 60L520 380L340 560L140 300M700 180L1080 200L1260 480L1440 380M900 90L1260 110M860 420L1260 480M1080 200L860 420" />
      <path d="M0 700L240 620L340 560L520 760L700 620L860 420L1000 700L1260 480L1440 620M240 620L140 300M340 560L520 760L700 900M520 760L700 620L1000 700L1080 900M1000 700L1260 800L1440 780M1260 480L1260 800M0 700L120 900M1260 800L1440 620" />
    </g>
    <g fill={color ?? theme.colors.bgMeshNode}>
      <circle cx="180" cy="40" r={radius} />
      <circle cx="360" cy="160" r={radius} />
      <circle cx="520" cy="60" r={radius} />
      <circle cx="700" cy="180" r={radius} />
      <circle cx="900" cy="90" r={radius} />
      <circle cx="1080" cy="200" r={radius} />
      <circle cx="1260" cy="110" r={radius} />
      <circle cx="140" cy="300" r={radius} />
      <circle cx="520" cy="380" r={radius} />
      <circle cx="860" cy="420" r={radius} />
      <circle cx="1260" cy="480" r={radius} />
      <circle cx="240" cy="620" r={radius} />
      <circle cx="340" cy="560" r={radius} />
      <circle cx="520" cy="760" r={radius} />
      <circle cx="700" cy="620" r={radius} />
      <circle cx="1000" cy="700" r={radius} />
      <circle cx="1260" cy="800" r={radius} />
    </g>
  </Mesh>;
}
