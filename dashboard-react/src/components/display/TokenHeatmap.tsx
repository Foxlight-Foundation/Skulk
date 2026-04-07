import { useCallback, useRef, useState } from 'react';
import styled, { useTheme } from 'styled-components';
import { Button } from '../common/Button';
import type { Theme } from '../../theme';

export interface TokenData {
  token: string;
  probability: number;
  logprob: number;
  topLogprobs?: Array<{ token: string; logprob: number }>;
}

export interface TokenHeatmapProps {
  tokens: TokenData[];
  isGenerating?: boolean;
  onRegenerateFrom?: (tokenIndex: number) => void;
  className?: string;
}

/* ---- confidence helpers ---- */

function getConfidenceStyle(prob: number): { bg: string; color: string; border: string } {
  if (prob > 0.8) return { bg: 'transparent', color: 'inherit', border: 'transparent' };
  if (prob > 0.5) return { bg: 'rgba(107,114,128,0.1)', color: 'inherit', border: 'rgba(107,114,128,0.2)' };
  if (prob > 0.2) return { bg: 'rgba(245,158,11,0.15)', color: 'rgba(253,230,138,0.9)', border: 'rgba(245,158,11,0.3)' };
  return { bg: 'rgba(239,68,68,0.2)', color: 'rgba(254,202,202,0.9)', border: 'rgba(239,68,68,0.4)' };
}

function probColor(prob: number, theme: Theme): string {
  if (prob > 0.8) return theme.colors.healthy;
  if (prob > 0.5) return theme.colors.textSecondary;
  if (prob > 0.2) return theme.colors.warning;
  return theme.colors.error;
}

/* ---- styles ---- */

const Container = styled.div`
  line-height: 1.6;
  white-space: pre-wrap;
  word-wrap: break-word;
`;

const Token = styled.span<{ $bg: string; $color: string; $border: string }>`
  padding: 2px;
  border-radius: 3px;
  border: 1px solid ${({ $border }) => $border};
  background: ${({ $bg }) => $bg};
  color: ${({ $color }) => $color};
  cursor: pointer;
  transition: opacity 0.15s;

  &:hover {
    opacity: 0.8;
  }
`;

const Tooltip = styled.div<{ $x: number; $y: number }>`
  position: fixed;
  left: ${({ $x }) => $x}px;
  top: ${({ $y }) => $y}px;
  transform: translate(-50%, -100%);
  z-index: 50;
  min-width: 192px;
  background: ${({ theme }) => theme.colors.surfaceElevated};
  backdrop-filter: blur(4px);
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: 12px;
  box-shadow: 0 20px 40px ${({ theme }) => theme.colors.shadow};
  padding: 10px 12px;
  font-size: ${({ theme }) => theme.fontSizes.sm};

  &::after {
    content: '';
    position: absolute;
    bottom: -6px;
    left: 50%;
    transform: translateX(-50%);
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 6px solid ${({ theme }) => theme.colors.surfaceElevated};
  }
`;

const TooltipHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
`;

const TooltipToken = styled.span`
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.text};
`;

const LogprobText = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: rgba(156, 163, 175, 0.8);
  margin-bottom: 8px;
`;

const AltRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.label};
  padding: 2px 0;
  color: rgba(209, 213, 219, 0.8);
`;

const RegenButton = styled(Button)`
  margin-top: 8px;
  color: rgba(156, 163, 175, 0.8);
  &:hover:not(:disabled) {
    color: ${({ theme }) => theme.colors.gold};
  }
`;

/* ---- component ---- */

export function TokenHeatmap({
  tokens,
  isGenerating = false,
  onRegenerateFrom,
  className,
}: TokenHeatmapProps) {
  const [hovered, setHovered] = useState<{
    index: number;
    x: number;
    y: number;
  } | null>(null);
  const [tooltipHovered, setTooltipHovered] = useState(false);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const theme = useTheme() as Theme;

  const handleMouseEnter = useCallback(
    (e: React.MouseEvent, index: number) => {
      clearTimeout(hideTimer.current);
      const rect = (e.target as HTMLElement).getBoundingClientRect();
      setHovered({ index, x: rect.left + rect.width / 2, y: rect.top - 10 });
    },
    [],
  );

  const handleMouseLeave = useCallback(() => {
    const delay = isGenerating ? 300 : 200;
    hideTimer.current = setTimeout(() => {
      if (!tooltipHovered) setHovered(null);
    }, delay);
  }, [isGenerating, tooltipHovered]);

  const handleTooltipEnter = useCallback(() => {
    clearTimeout(hideTimer.current);
    setTooltipHovered(true);
  }, []);

  const handleTooltipLeave = useCallback(() => {
    setTooltipHovered(false);
    setHovered(null);
  }, []);

  const hoveredToken = hovered ? tokens[hovered.index] : null;

  return (
    <Container className={className}>
      {tokens.map((t, i) => {
        const style = getConfidenceStyle(t.probability);
        return (
          <Token
            key={i}
            $bg={style.bg}
            $color={style.color}
            $border={style.border}
            role="button"
            tabIndex={0}
            onMouseEnter={(e) => handleMouseEnter(e, i)}
            onMouseLeave={handleMouseLeave}
          >
            {t.token}
          </Token>
        );
      })}

      {hovered && hoveredToken && (
        <Tooltip
          $x={hovered.x}
          $y={hovered.y}
          onMouseEnter={handleTooltipEnter}
          onMouseLeave={handleTooltipLeave}
        >
          <TooltipHeader>
            <TooltipToken>"{hoveredToken.token}"</TooltipToken>
            <span style={{ color: probColor(hoveredToken.probability, theme), fontWeight: 600 }}>
              {(hoveredToken.probability * 100).toFixed(1)}%
            </span>
          </TooltipHeader>
          <LogprobText>logprob: {hoveredToken.logprob.toFixed(3)}</LogprobText>

          {hoveredToken.topLogprobs && hoveredToken.topLogprobs.length > 0 && (
            <div style={{ borderTop: '1px solid rgba(75,85,99,0.3)', paddingTop: 6, marginBottom: 4 }}>
              {hoveredToken.topLogprobs.slice(0, 5).map((alt, i) => (
                <AltRow key={i}>
                  <span>"{alt.token}"</span>
                  <span>{(Math.exp(alt.logprob) * 100).toFixed(1)}%</span>
                </AltRow>
              ))}
            </div>
          )}

          {onRegenerateFrom && (
            <RegenButton
              variant="ghost"
              size="sm"
              onClick={() => {
                setHovered(null);
                onRegenerateFrom(hovered.index);
              }}
            >
              ↻ Regenerate from here
            </RegenButton>
          )}
        </Tooltip>
      )}
    </Container>
  );
}
