import styled from 'styled-components';

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

function computeEta(progress: PrefillProgress): string | null {
  const elapsed = performance.now() - progress.startedAt;
  if (elapsed < 200 || progress.processed <= 0) return null;

  const tokensPerMs = progress.processed / elapsed;
  const remaining = progress.total - progress.processed;
  const remainingMs = remaining / tokensPerMs;
  const remainingSec = Math.ceil(remainingMs / 1000);

  if (remainingSec <= 0) return null;
  if (remainingSec < 60) return `~${remainingSec}s remaining`;
  const m = Math.floor(remainingSec / 60);
  const s = remainingSec % 60;
  return `~${m}m ${s}s remaining`;
}

/* ---- styles ---- */

const Container = styled.div`
  width: 100%;
`;

const LabelRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: ${({ theme }) => theme.fontSizes.label};
  color: ${({ theme }) => theme.colors.textSecondary};
  margin-bottom: 4px;
`;

const TokenCount = styled.span`
  font-family: ${({ theme }) => theme.fonts.body};
`;

const Track = styled.div`
  height: 6px;
  background: ${({ theme }) => theme.colors.overlay};
  border-radius: 3px;
  overflow: hidden;
`;

const Fill = styled.div<{ $pct: number }>`
  height: 100%;
  width: ${({ $pct }) => $pct}%;
  background: ${({ theme }) => theme.colors.gold};
  border-radius: 3px;
  transition: width 150ms ease-out;
`;

const FooterRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-family: ${({ theme }) => theme.fonts.body};
  color: rgba(179, 179, 179, 0.7);
  margin-top: 4px;
`;

/* ---- component ---- */

export function PrefillProgressBar({ progress, className }: PrefillProgressBarProps) {
  const percentage = progress.total > 0
    ? Math.round((progress.processed / progress.total) * 100)
    : 0;

  const eta = computeEta(progress);

  return (
    <Container className={className}>
      <LabelRow>
        <span>Processing prompt</span>
        <TokenCount>
          {formatTokenCount(progress.processed)} / {formatTokenCount(progress.total)} tokens
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
