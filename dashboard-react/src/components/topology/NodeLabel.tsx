import { formatBytes } from '../../utils/format';

export interface NodeLabelProps {
  name: string;
  ramUsed: number;
  ramTotal: number;
  /** font size for the name label */
  fontSize?: number;
  /** horizontal center for text alignment */
  cx?: number;
  /** vertical offset from top of this group to the name text */
  nameY?: number;
  /** vertical offset from top of this group to the memory text */
  memoryY?: number;
}

export function NodeLabel({
  name,
  ramUsed,
  ramTotal,
  fontSize = 13,
  cx = 0,
  nameY = 0,
  memoryY = 20,
}: NodeLabelProps) {
  const ramPercent = ramTotal > 0 ? ((ramUsed / ramTotal) * 100).toFixed(0) : '0';
  const usedStr = formatBytes(ramUsed);
  const totalStr = formatBytes(ramTotal);

  return (
    <g>
      {/* Name above */}
      <text x={cx} y={nameY} textAnchor="middle" dominantBaseline="middle"
        fill="#FFD700" fontSize={fontSize} fontWeight={700}
        fontFamily="SF Mono, Monaco, monospace">
        {name}
      </text>
      {/* Memory below: "15.4GB" in grey, "/24GB (64%)" in yellow */}
      <text x={cx} y={memoryY} textAnchor="middle" dominantBaseline="middle"
        fontFamily="SF Mono, Monaco, monospace" fontSize={13}>
        <tspan fill="#999999">{usedStr}</tspan>
        <tspan fill="#FFD700">/{totalStr}</tspan>
        <tspan fill="#999999">{' '}({ramPercent}%)</tspan>
      </text>
    </g>
  );
}
