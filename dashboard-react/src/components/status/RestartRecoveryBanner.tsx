import styled from 'styled-components';
import { useSkulkTranslation } from '../../i18n/tolgee';

/** Props for {@link RestartRecoveryBanner}. */
export interface RestartRecoveryBannerProps {
  /** Whether a cluster that started cold is still restoring from its record. */
  pending: boolean;
  /**
   * How many models the master is bringing back, once it has chosen which
   * record to restore from; null before that.
   */
  placements: number | null;
}

const Banner = styled.div`
  position: relative;
  z-index: 50;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px 16px;
  background: ${({ theme }) => theme.colors.infoBg};
  border-bottom: 1px solid ${({ theme }) => theme.colors.info};
`;

const Text = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.text};
`;

/**
 * Say, while a restart restore runs, that models are coming back on their own,
 * so an operator who opens the dashboard right after a restart does not start
 * placing them by hand. Placements that cannot come back appear in the usual
 * failure history, not here.
 */
export function RestartRecoveryBanner({ pending, placements }: RestartRecoveryBannerProps) {
  const { t } = useSkulkTranslation();

  // Nothing is coming back (an empty record, or the Settings toggle is off).
  if (!pending || placements === 0) return null;

  let message: string;
  if (placements === null) {
    message = t(
      'restartRecovery.banner.waiting',
      'Restarting: waiting for the machines to rejoin before bringing models back',
    );
  } else if (placements === 1) {
    message = t('restartRecovery.banner.restoringOne', 'Bringing back 1 model after the restart');
  } else {
    message = t('restartRecovery.banner.restoringMany', 'Bringing back {count} models after the restart', {
      count: placements,
    });
  }

  return (
    <Banner role="status" aria-live="polite">
      <Text>{message}</Text>
    </Banner>
  );
}
