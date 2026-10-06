import styled from 'styled-components';
import { useSkulkTranslation, type SkulkTranslate } from '../../i18n/tolgee';

export interface PrefillProgress {
  processed: number;
  total: number;
  /** performance.now() timestamp when processing started. */
  startedAt: number;
}

export interface PrefillProgressBarProps {
  progress: PrefillProgress;
  className?: string;
}

function formatTokenCount(n: number | null | undefined): string {
  if (n == null || n === 0) return '0';
  if (n < 1000) return String(n);
  return `${(n / 1000).toFixed(1)}k`;
}

function computeEta(progress: PrefillProgress, t: SkulkTranslate): string | null {
  const elapsed = performance.now() - progress.startedAt;
  if (elapsed < 200 || progress.processed <= 0) return null;

  const tokensPerMs = progress.processed / elapsed;
  const remaining = progress.total - progress.processed;
  const remainingMs = remaining / tokensPerMs;
  const remainingSec = Math.ceil(remainingMs / 1000);

  if (remainingSec <= 0) return null;
  if (remainingSec < 60) return t('prefillProgress.secondsRemaining', '~{seconds}s remaining', { seconds: remainingSec });
  const m = Math.floor(remainingSec / 60);
  const s = remainingSec % 60;
  return t('prefillProgress.minutesRemaining', '~{minutes}m {seconds}s remaining', { minutes: m, seconds: s });
}

/* ---- styles ---- */

const Container = styled.div`
  width: 100%; padding: 8px 12px; border-radius: 8px;
  border: 1px solid ${({ theme }) => theme.colors.borderLive};
  background: ${({ theme }) => theme.colors.liveBg};
`;

const LabelRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  color: ${({ theme }) => theme.colors.textSecondary};
  margin-bottom: 4px;
`;

const TokenCount = styled.span`
  font-family: ${({ theme }) => theme.fonts.mono};
`;

const Track = styled.div`
  height: 3px;
  background: ${({ theme }) => theme.colors.selected};
  border-radius: 2px;
  overflow: hidden;
`;

const Fill = styled.div<{ $pct: number }>`
  height: 100%;
  width: ${({ $pct }) => $pct}%;
  background: linear-gradient(90deg, ${({ theme }) => theme.colors.liveDeep}, ${({ theme }) => theme.colors.live});
  border-radius: 2px;
  transition: width 150ms ease-out;
`;

const FooterRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.subtleText};
  margin-top: 4px;
`;

/* ---- component ---- */

export function PrefillProgressBar({ progress, className }: PrefillProgressBarProps) {
  const { t } = useSkulkTranslation();
  const percentage = progress.total > 0
    ? Math.round((progress.processed / progress.total) * 100)
    : 0;

  const eta = computeEta(progress, t);

  return (
    <Container className={className}>
      <LabelRow>
        <span>{t('prefillProgress.processingPrompt', 'Processing prompt')}</span>
        <TokenCount>
          {t('prefillProgress.tokenCount', '{processed} / {total} tokens', {
            processed: formatTokenCount(progress.processed),
            total: formatTokenCount(progress.total),
          })}
        </TokenCount>
      </LabelRow>
      <Track>
        <Fill $pct={percentage} />
      </Track>
      <FooterRow>
        <span>{eta ?? ''}</span>
        <span>{percentage}%</span>
      </FooterRow>
    </Container>
  );
}
