/**
 * PrefillProgressBar
 *
 * Shows prompt processing (prefill) progress with a percentage bar.
 * Ported from PrefillProgressBar.svelte.
 */
import React from 'react';
import styled, { keyframes } from 'styled-components';
import { useTranslate } from '@tolgee/react';

const shimmer = keyframes`
  0%   { background-position: -200% center; }
  100% { background-position:  200% center; }
`;

const Wrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 6px 0;
`;

const Label = styled.div`
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.mutedForeground};
`;

const Track = styled.div`
  height: 3px;
  background: ${({ theme }) => theme.colors.mediumGray};
  border-radius: 2px;
  overflow: hidden;
`;

const Fill = styled.div<{ $pct: number }>`
  height: 100%;
  width: ${({ $pct }) => $pct}%;
  background: linear-gradient(
    90deg,
    ${({ theme }) => theme.colors.yellowDarker} 0%,
    ${({ theme }) => theme.colors.yellow} 50%,
    ${({ theme }) => theme.colors.yellowDarker} 100%
  );
  background-size: 200% auto;
  animation: ${shimmer} 1.5s linear infinite;
  border-radius: 2px;
  transition: width 0.2s ease;
`;

export interface PrefillProgressBarProps {
  processed?: number;
  total?: number;
  percentage?: number;
}

export const PrefillProgressBar: React.FC<PrefillProgressBarProps> = ({
  processed = 0,
  total = 0,
  percentage,
}) => {
  const { t } = useTranslate();
  const pct = percentage ?? (total > 0 ? Math.round((processed / total) * 100) : 0);

  return (
    <Wrapper>
      <Label>{t('chat.prefill')}</Label>
      <Track>
        <Fill $pct={pct} />
      </Track>
    </Wrapper>
  );
};
