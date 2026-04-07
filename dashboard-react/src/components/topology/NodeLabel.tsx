import { useState, useCallback, useRef, useEffect } from 'react';
import { useTheme } from 'styled-components';
import { formatBytes } from '../../utils/format';
import { InfoTooltip } from '../common/InfoTooltip';
import type { Theme } from '../../theme';

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
  /** When set, shows an info icon next to the name with this content in a tooltip. */
  debugContent?: React.ReactNode;
  /** When set, shows a restart icon before the name. */
  onRestart?: () => void;
}

export function NodeLabel({
  name,
  ramUsed,
  ramTotal,
  fontSize = 13,
  cx = 0,
  nameY = 0,
  memoryY = 20,
  debugContent,
  onRestart,
}: NodeLabelProps) {
  const theme = useTheme() as Theme;
  const ramPercent = ramTotal > 0 ? ((ramUsed / ramTotal) * 100).toFixed(0) : '0';
  const usedStr = formatBytes(ramUsed);
  const totalStr = formatBytes(ramTotal);

  // Estimate name width for positioning icons around the name text
  const nameWidth = name.length * fontSize * 0.62;

  const [confirming, setConfirming] = useState(false);
  const confirmTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clear the confirmation timeout on unmount to avoid stale state updates
  useEffect(() => {
    return () => {
      if (confirmTimerRef.current) clearTimeout(confirmTimerRef.current);
    };
  }, []);

  const handleRestartClick = useCallback(() => {
    if (!confirming) {
      setConfirming(true);
      confirmTimerRef.current = setTimeout(() => setConfirming(false), 3000);
      return;
    }
    if (confirmTimerRef.current) clearTimeout(confirmTimerRef.current);
    confirmTimerRef.current = null;
    setConfirming(false);
    onRestart?.();
  }, [confirming, onRestart]);

  return (
    <g>
      {/* Restart icon — positioned left of the name text */}
      {onRestart && (
        <foreignObject
          x={cx - nameWidth / 2 - 24}
          y={nameY - 10}
          width={20}
          height={20}
          style={{ overflow: 'visible' }}
        >
          <InfoTooltip
            placement="left"
            content={confirming ? 'Click again to confirm restart' : 'Restart this node — releases GPU memory and rejoins the cluster'}
          >
            <svg
              width="16" height="16" viewBox="0 0 16 16" fill="none"
              style={{ cursor: 'pointer', opacity: confirming ? 1 : 0.5, transition: 'opacity 0.15s' }}
              role="button"
              tabIndex={0}
              aria-label={confirming ? 'Confirm restart of this node' : 'Restart this node'}
              onClick={handleRestartClick}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleRestartClick(); } }}
              onMouseEnter={(e) => { (e.currentTarget as SVGElement).style.opacity = '1'; }}
              onMouseLeave={(e) => { (e.currentTarget as SVGElement).style.opacity = confirming ? '1' : '0.5'; }}
            >
              <path
                d="M13.65 2.35A7.96 7.96 0 0 0 8 0a8 8 0 1 0 8 8h-2a6 6 0 1 1-1.76-4.24L9 7h7V0l-2.35 2.35z"
                fill={confirming ? theme.colors.warning : '#999'}
              />
            </svg>
          </InfoTooltip>
        </foreignObject>
      )}
      {/* Name above */}
      <text x={cx} y={nameY} textAnchor="middle" dominantBaseline="middle"
        fill={theme.colors.gold} fontSize={fontSize} fontWeight={700}
        fontFamily="SF Mono, Monaco, monospace">
        {name}
      </text>
      {/* Debug info icon — positioned right after the name text */}
      {debugContent && (
        <foreignObject
          x={cx + nameWidth / 2 + 4}
          y={nameY - 12}
          width={20}
          height={20}
          style={{ overflow: 'visible' }}
        >
          <InfoTooltip
            filled
            placement="right"
            content={debugContent}
          />
        </foreignObject>
      )}
      {/* Memory below: "15.4GB" in grey, "/24GB (64%)" in yellow */}
      <text x={cx} y={memoryY} textAnchor="middle" dominantBaseline="middle"
        fontFamily="SF Mono, Monaco, monospace" fontSize={13}>
        <tspan fill="#999999">{usedStr}</tspan>
        <tspan fill={theme.colors.gold}>/{totalStr}</tspan>
        <tspan fill="#999999">{' '}({ramPercent}%)</tspan>
      </text>
    </g>
  );
}
