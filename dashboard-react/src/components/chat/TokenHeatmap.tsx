/**
 * TokenHeatmap
 *
 * Interactive token-by-token uncertainty visualization.
 * Tokens coloured blue (high confidence) → red (low confidence).
 * Ported from TokenHeatmap.svelte.
 */
import React, { useState, useCallback } from 'react';
import styled from 'styled-components';

// ─── Types ─────────────────────────────────────────────────────────────────────

export interface HeatmapToken {
  token: string;
  logprob: number;
  topLogprobs?: Array<{ token: string; logprob: number }>;
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

function logprobToConfidence(lp: number): number {
  // logprob is in (-∞, 0]; map to [0, 1]
  return Math.exp(Math.max(lp, -10));
}

function confidenceToColor(conf: number): string {
  // Blue (high) → Yellow (mid) → Red (low)
  const blue   = { r: 93,  g: 173, b: 226 };
  const yellow = { r: 255, g: 215, b: 0 };
  const red    = { r: 244, g: 67,  b: 54 };

  let r: number, g: number, b: number;
  if (conf >= 0.5) {
    const t = (conf - 0.5) * 2;
    r = Math.round(yellow.r * (1 - t) + blue.r * t);
    g = Math.round(yellow.g * (1 - t) + blue.g * t);
    b = Math.round(yellow.b * (1 - t) + blue.b * t);
  } else {
    const t = conf * 2;
    r = Math.round(red.r * (1 - t) + yellow.r * t);
    g = Math.round(red.g * (1 - t) + yellow.g * t);
    b = Math.round(red.b * (1 - t) + yellow.b * t);
  }
  return `rgba(${r},${g},${b},0.7)`;
}

// ─── Styled components ────────────────────────────────────────────────────────

const Container = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 2px;
  position: relative;
  font-size: 12px;
  font-family: ${({ theme }) => theme.fonts.mono};
  line-height: 1.6;
`;

const TokenSpan = styled.span<{ $bg: string; $opacity: number }>`
  background: ${({ $bg }) => $bg};
  opacity: ${({ $opacity }) => $opacity};
  border-radius: 2px;
  padding: 0 2px;
  cursor: pointer;
  white-space: pre;
  position: relative;
  &:hover { opacity: 1; }
`;

const Tooltip = styled.div`
  position: fixed;
  z-index: 9998;
  background: ${({ theme }) => theme.colors.darkGray};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radius.md};
  padding: 8px 10px;
  font-size: 11px;
  min-width: 160px;
  box-shadow: 0 4px 20px oklch(0 0 0 / 0.5);
  pointer-events: none;
`;

const TooltipToken = styled.div`
  font-weight: 700;
  color: ${({ theme }) => theme.colors.yellow};
  margin-bottom: 4px;
`;

const TooltipRow = styled.div`
  color: ${({ theme }) => theme.colors.lightGray};
  display: flex;
  justify-content: space-between;
  gap: 12px;
`;

const TooltipAlt = styled.div<{ $conf: number }>`
  color: ${({ $conf }) => confidenceToColor($conf)};
  font-size: 10px;
  margin-top: 2px;
`;

// ─── Component ─────────────────────────────────────────────────────────────────

export interface TokenHeatmapProps {
  tokens: HeatmapToken[];
}

export const TokenHeatmap: React.FC<TokenHeatmapProps> = ({ tokens }) => {
  const [tooltip, setTooltip] = useState<{
    token: HeatmapToken;
    x: number;
    y: number;
  } | null>(null);

  const handleMouseEnter = useCallback(
    (e: React.MouseEvent, token: HeatmapToken) => {
      setTooltip({ token, x: e.clientX + 12, y: e.clientY - 8 });
    },
    [],
  );

  const handleMouseLeave = useCallback(() => setTooltip(null), []);

  return (
    <Container>
      {tokens.map((tok, i) => {
        const conf = logprobToConfidence(tok.logprob);
        const bg = confidenceToColor(conf);
        return (
          <TokenSpan
            key={i}
            $bg={bg}
            $opacity={0.5 + conf * 0.5}
            onMouseEnter={(e) => handleMouseEnter(e, tok)}
            onMouseLeave={handleMouseLeave}
            role="button"
            tabIndex={0}
            title={`logprob: ${tok.logprob.toFixed(3)}`}
          >
            {tok.token}
          </TokenSpan>
        );
      })}

      {tooltip && (
        <Tooltip style={{ top: tooltip.y, left: tooltip.x }}>
          <TooltipToken>
            &ldquo;{tooltip.token.token}&rdquo;
          </TooltipToken>
          <TooltipRow>
            <span>logprob</span>
            <span>{tooltip.token.logprob.toFixed(4)}</span>
          </TooltipRow>
          <TooltipRow>
            <span>confidence</span>
            <span>{(logprobToConfidence(tooltip.token.logprob) * 100).toFixed(1)}%</span>
          </TooltipRow>
          {tooltip.token.topLogprobs && tooltip.token.topLogprobs.length > 0 && (
            <>
              <TooltipRow style={{ marginTop: 6, opacity: 0.6 }}>
                <span>alternatives</span>
              </TooltipRow>
              {tooltip.token.topLogprobs.slice(0, 4).map((alt, j) => (
                <TooltipAlt key={j} $conf={logprobToConfidence(alt.logprob)}>
                  &ldquo;{alt.token}&rdquo; {alt.logprob.toFixed(3)}
                </TooltipAlt>
              ))}
            </>
          )}
        </Tooltip>
      )}
    </Container>
  );
};
