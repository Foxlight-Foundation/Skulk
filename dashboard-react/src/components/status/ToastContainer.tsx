import styled, { keyframes, useTheme } from 'styled-components';
import { useToast, type Toast } from '../../hooks/useToast';
import { Button } from '../common/Button';
import type { Theme } from '../../theme';

/* ---- type config ---- */

interface TypeStyle {
  borderColor: string;
  iconColor: string;
  iconPath: string;
  progressColor: string;
}

/** Per-toast-type display config. Built from the active theme so colors track theme switches. */
function buildTypeStyles(theme: Theme): Record<Toast['type'], TypeStyle> {
  return {
    success: {
      borderColor: theme.colors.accent,
      iconColor: theme.colors.healthy,
      iconPath: 'M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
      progressColor: theme.colors.accentBg,
    },
    error: {
      borderColor: theme.colors.error,
      iconColor: theme.colors.error,
      iconPath: 'M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z',
      progressColor: theme.colors.errorBg,
    },
    warning: {
      borderColor: theme.colors.warning,
      iconColor: theme.colors.warning,
      iconPath: 'M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126z',
      progressColor: theme.colors.warningBg,
    },
    info: {
      borderColor: theme.colors.info,
      iconColor: theme.colors.info,
      iconPath: 'M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z',
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
  z-index: 9999;
  display: flex;
  flex-direction: column;
  gap: 8px;
  pointer-events: none;
`;

const ToastCard = styled.div<{ $borderColor: string }>`
  pointer-events: auto;
  max-width: 360px;
  width: 320px;
  background: ${({ theme }) => theme.colors.surface};
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
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.text};
  line-height: 1.4;
  margin: 0;
`;

const DismissBtn = styled(Button)`
  flex-shrink: 0;
  color: ${({ theme }) => theme.colors.textMuted};
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

export function ToastContainer() {
  const { toasts, dismissToast } = useToast();
  const theme = useTheme() as Theme;
  const TYPE_STYLES = buildTypeStyles(theme);

  if (toasts.length === 0) return null;

  return (
    <Container role="log" aria-live="polite" aria-label="Notifications">
      {toasts.map((toast) => {
        const style = TYPE_STYLES[toast.type];
        return (
          <ToastCard key={toast.id} $borderColor={style.borderColor} role="alert">
            <Body>
              <svg
                width={20} height={20} viewBox="0 0 24 24"
                fill="none" stroke={style.iconColor}
                strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"
                style={{ flexShrink: 0, marginTop: 2 }}
              >
                <path d={style.iconPath} />
              </svg>
              <Message>{toast.message}</Message>
              <DismissBtn variant="ghost" size="sm" icon onClick={() => dismissToast(toast.id)} aria-label="Dismiss notification">
                <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                  <path d="M6 18L18 6M6 6l12 12" />
                </svg>
              </DismissBtn>
            </Body>
            {toast.duration > 0 && (
              <ProgressTrack>
                <ProgressBar $color={style.progressColor} $duration={toast.duration} />
              </ProgressTrack>
            )}
          </ToastCard>
        );
      })}
    </Container>
  );
}
