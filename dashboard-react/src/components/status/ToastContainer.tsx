import styled, { keyframes, useTheme } from 'styled-components';
import { useToast, type Toast } from '../../hooks/useToast';
import { Button } from '../common/Button';
import type { Theme } from '../../theme';
import { useSkulkTranslation } from '../../i18n/tolgee';

/* ---- type config ---- */

interface TypeStyle {
  borderColor: string;
  iconColor: string;
  progressColor: string;
}

/** Per-toast-type display config. Built from the active theme so colors track theme switches. */
function buildTypeStyles(theme: Theme): Record<Toast['type'], TypeStyle> {
  return {
    success: {
      borderColor: theme.colors.accent,
      iconColor: theme.colors.healthy,
      progressColor: theme.colors.accentBg,
    },
    error: {
      borderColor: theme.colors.error,
      iconColor: theme.colors.error,
      progressColor: theme.colors.errorBg,
    },
    warning: {
      borderColor: theme.colors.warning,
      iconColor: theme.colors.warning,
      progressColor: theme.colors.warningBg,
    },
    info: {
      borderColor: theme.colors.info,
      iconColor: theme.colors.info,
      progressColor: theme.colors.infoBg,
    },
  };
}

/* ---- animations ---- */

const slideIn = keyframes`
  from { transform: translateX(80px); opacity: 0; }
  to   { transform: translateX(0); opacity: 1; }
`;

const shrink = keyframes`
  from { width: 100%; }
  to   { width: 0%; }
`;

/* ---- styles ---- */

const Container = styled.div`
  position: fixed;
  bottom: 24px;
  right: 24px;
  max-width: calc(100vw - 32px);
  @media (max-width: 480px) { right: 16px; bottom: 16px; }
  z-index: 9999;
  display: flex;
  flex-direction: column;
  gap: 8px;
  pointer-events: none;
`;

const ToastCard = styled.div<{ $borderColor: string }>`
  pointer-events: auto;
  max-width: 100%;
  width: 320px;
  background: ${({ theme }) => theme.colors.surfaceElevated};
  backdrop-filter: blur(4px);
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-left: 3px solid ${({ $borderColor }) => $borderColor};
  border-radius: ${({ theme }) => theme.radii.md};
  box-shadow: 0 4px 12px ${({ theme }) => theme.colors.shadow};
  animation: ${slideIn} 0.25s ease-out;
`;

const Body = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 16px;
`;

const Message = styled.p`
  flex: 1;
  min-width: 0; overflow-wrap: anywhere;
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.text};
  line-height: 1.4;
  margin: 0;
`;

const DismissBtn = styled(Button)`
  flex-shrink: 0;
  color: ${({ theme }) => theme.colors.subtleText};
  &:hover:not(:disabled) { color: ${({ theme }) => theme.colors.textSecondary}; background: transparent; }
`;

const ProgressTrack = styled.div`
  height: 2px;
  background: ${({ theme }) => theme.colors.surfaceSunken};
  border-radius: 0 0 ${({ theme }) => theme.radii.md} ${({ theme }) => theme.radii.md};
  overflow: hidden;
`;

const ProgressBar = styled.div<{ $color: string; $duration: number }>`
  height: 100%;
  background: ${({ $color }) => $color};
  animation: ${shrink} ${({ $duration }) => $duration}ms linear forwards;
`;

/* ---- component ---- */

/** Single notification presentation, also used for deterministic gallery specimens. */
export function ToastNotification({ toast, onDismiss }: { toast: Toast; onDismiss: () => void }) {
  const { t } = useSkulkTranslation();
  const theme = useTheme() as Theme;
  const style = buildTypeStyles(theme)[toast.type];
  return <ToastCard $borderColor={style.borderColor} role="alert">
    <Body>
      <span aria-hidden="true" style={{ width: 8, height: 8, borderRadius: '50%', flexShrink: 0, marginTop: 6, background: style.iconColor }} />
      <Message>{toast.message}</Message>
      <DismissBtn variant="ghost" size="sm" icon onClick={onDismiss} aria-label={t('toast.dismissNotification', 'Dismiss notification')}>
        <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round"><path d="M6 18L18 6M6 6l12 12" /></svg>
      </DismissBtn>
    </Body>
    {toast.duration > 0 && <ProgressTrack><ProgressBar $color={style.progressColor} $duration={toast.duration} /></ProgressTrack>}
  </ToastCard>;
}

/** Subscribes to the existing notification store and preserves its dismissal lifecycle. */
export function ToastContainer() {
  const { t } = useSkulkTranslation();
  const { toasts, dismissToast } = useToast();
  if (toasts.length === 0) return null;
  return <Container role="log" aria-live="polite" aria-label={t('toast.notifications', 'Notifications')}>
    {toasts.map(toast => <ToastNotification key={toast.id} toast={toast} onDismiss={() => dismissToast(toast.id)} />)}
  </Container>;
}
