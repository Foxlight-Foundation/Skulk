import { useCallback, useEffect, useRef, useState } from 'react';
import { QRCodeCanvas } from 'qrcode.react';
import styled from 'styled-components';

import { addToast } from '../../hooks/useToast';
import { useSkulkTranslation } from '../../i18n/tolgee';
import {
  createPairingInvitation,
  PairingInvitationRequestError,
  type CreatedPairingInvitation,
  type PairingInvitationState,
  useGetPairingInvitationsQuery,
  useRevokePairingInvitationMutation,
} from '../../store/endpoints/pairing';
import { Button } from '../common/Button';

const pairingCodeDisplayMilliseconds = 5 * 60 * 1_000;
const defaultInvitationLifetimeSeconds = 5 * 60;
const defaultMaximumPairings = 1;

const invitationLifetimeOptions = [
  5 * 60,
  60 * 60,
  24 * 60 * 60,
  7 * 24 * 60 * 60,
  30 * 24 * 60 * 60,
  90 * 24 * 60 * 60,
] as const;

const Fieldset = styled.fieldset`
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 14px;
`;

const Legend = styled.legend`
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-family: ${({ theme }) => theme.fonts.body};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.textSecondary};
  padding: 0 6px;
`;

const Intro = styled.p`
  margin: 0;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  line-height: 1.5;
  color: ${({ theme }) => theme.colors.textMuted};
`;

const FormGrid = styled.div`
  display: grid;
  grid-template-columns: minmax(0, 1fr) 96px;
  gap: 10px;
`;

const Control = styled.label`
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const Select = styled.select`
  box-sizing: border-box;
  height: 36px;
  width: 100%;
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.bg};
  color: ${({ theme }) => theme.colors.text};
  padding: 0 9px;
  font: inherit;

  &:focus-visible {
    outline: 2px solid ${({ theme }) => theme.colors.goldDim};
    outline-offset: 1px;
  }
`;

const QrPanel = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
`;

const QrFrame = styled.div`
  box-sizing: border-box;
  padding: 12px;
  border: 1px solid ${({ theme }) => theme.colors.borderStrong};
  border-radius: ${({ theme }) => theme.radii.lg};
  background: #ffffff;
  line-height: 0;
  box-shadow: 0 10px 28px ${({ theme }) => theme.colors.shadow};

  canvas {
    width: min(280px, 68vw) !important;
    height: auto !important;
  }
`;

const QrTitle = styled.div`
  text-align: center;
  color: ${({ theme }) => theme.colors.text};
  font-size: ${({ theme }) => theme.fontSizes.md};
  font-weight: 600;
`;

const QrMeta = styled.div`
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 6px 12px;
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  text-align: center;
`;

const SecretWarning = styled.div`
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.warningBg};
  color: ${({ theme }) => theme.colors.warningOnSurface};
  padding: 10px 12px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  line-height: 1.45;
`;

const ButtonRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  width: 100%;

  > button {
    flex: 1;
  }
`;

const ErrorText = styled.div`
  color: ${({ theme }) => theme.colors.errorOnSurface};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  line-height: 1.45;
`;

const InvitationList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  border-top: 1px solid ${({ theme }) => theme.colors.borderLight};
  padding-top: 12px;
`;

const ListTitle = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.textSecondary};
  text-transform: uppercase;
  letter-spacing: 0.08em;
`;

const InvitationRow = styled.div`
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 10px;
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surfaceSunken};
  padding: 10px;
`;

const InvitationCopy = styled.div`
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
`;

const InvitationHeadline = styled.div`
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: ${({ theme }) => theme.fontSizes.sm};
`;

const StatePill = styled.span<{ $state: PairingInvitationState }>`
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 2px 7px;
  background: ${({ $state, theme }) =>
    $state === 'active' ? theme.colors.infoBg : theme.colors.surfaceElevated};
  color: ${({ $state, theme }) =>
    $state === 'active' ? theme.colors.info : theme.colors.textMuted};
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 10px;
`;

const InvitationMeta = styled.div`
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  line-height: 1.4;
`;

/** Dashboard pairing invitation creation, display, and revocation controls. */
export function PairingSettings() {
  const { t } = useSkulkTranslation();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [validForSeconds, setValidForSeconds] = useState(defaultInvitationLifetimeSeconds);
  const [maxPairings, setMaxPairings] = useState(defaultMaximumPairings);
  const [created, setCreated] = useState<CreatedPairingInvitation | null>(null);
  const [displayUntil, setDisplayUntil] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const {
    data: invitations,
    isError: invitationListFailed,
    refetch: refetchInvitations,
  } = useGetPairingInvitationsQuery(undefined, {
    pollingInterval: 15_000,
    skipPollingIfUnfocused: true,
    refetchOnFocus: true,
    refetchOnReconnect: true,
  });
  const [revokeInvitation, revokeResult] = useRevokePairingInvitationMutation();

  const resetPairingDisplay = useCallback(() => {
    setCreated(null);
    setDisplayUntil(null);
    setValidForSeconds(defaultInvitationLifetimeSeconds);
    setMaxPairings(defaultMaximumPairings);
    setNow(Date.now());
  }, []);

  useEffect(() => {
    if (displayUntil === null) return;
    const updateDisplayClock = () => {
      const nextNow = Date.now();
      setNow(nextNow);
      if (nextNow >= displayUntil) resetPairingDisplay();
    };
    const timer = window.setInterval(updateDisplayClock, 1_000);
    document.addEventListener('visibilitychange', updateDisplayClock);
    window.addEventListener('focus', updateDisplayClock);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', updateDisplayClock);
      window.removeEventListener('focus', updateDisplayClock);
    };
  }, [displayUntil, resetPairingDisplay]);

  const generatePairingCode = useCallback(async () => {
    setCreating(true);
    setError(null);
    try {
      const invitation = await createPairingInvitation({ validForSeconds, maxPairings });
      const nextNow = Date.now();
      setCreated(invitation);
      setNow(nextNow);
      setDisplayUntil(nextNow + pairingCodeDisplayMilliseconds);
      void refetchInvitations();
      addToast({
        type: 'success',
        message: t('settings.pairing.created', 'Pairing code ready'),
      });
    } catch (caught: unknown) {
      setError(
        caught instanceof PairingInvitationRequestError
          ? caught.message
          : t('settings.pairing.createFailed', 'Skulk could not generate a pairing code.'),
      );
    } finally {
      setCreating(false);
    }
  }, [maxPairings, refetchInvitations, t, validForSeconds]);

  const revoke = useCallback(
    async (invitationId: string) => {
      setError(null);
      try {
        await revokeInvitation(invitationId).unwrap();
        if (created?.invitation.invitationId === invitationId) resetPairingDisplay();
        addToast({
          type: 'success',
          message: t('settings.pairing.revoked', 'Pairing invitation revoked'),
        });
      } catch {
        setError(t('settings.pairing.revokeFailed', 'Skulk could not revoke this invitation.'));
      }
    },
    [created?.invitation.invitationId, resetPairingDisplay, revokeInvitation, t],
  );

  const downloadQr = useCallback(() => {
    if (created === null || canvasRef.current === null) return;
    const link = document.createElement('a');
    link.download = `skulk-pairing-${created.invitation.invitationId}.png`;
    link.href = canvasRef.current.toDataURL('image/png');
    link.click();
  }, [created]);

  const remainingSeconds =
    displayUntil === null ? 0 : Math.max(0, Math.ceil((displayUntil - now) / 1_000));

  return (
    <Fieldset>
      <Legend>{t('settings.pairing.legend', 'Pairing')}</Legend>
      {created === null ? (
        <>
          <Intro>
            {t(
              'settings.pairing.intro',
              'Generate a protected code for the Skulk Operator app. For security, open this dashboard through localhost on the operator gateway. The code is a bearer secret and is shown here for five minutes.',
            )}
          </Intro>
          <FormGrid>
            <Control>
              {t('settings.pairing.validFor', 'Valid for')}
              <Select
                aria-label={t('settings.pairing.validFor', 'Valid for')}
                value={validForSeconds}
                onChange={(event) => setValidForSeconds(Number(event.target.value))}
              >
                {invitationLifetimeOptions.map((seconds) => (
                  <option key={seconds} value={seconds}>
                    {durationLabel(seconds, t)}
                  </option>
                ))}
              </Select>
            </Control>
            <Control>
              {t('settings.pairing.devicesAllowed', 'Devices')}
              <Select
                aria-label={t('settings.pairing.devicesAllowed', 'Devices')}
                value={maxPairings}
                onChange={(event) => setMaxPairings(Number(event.target.value))}
              >
                {Array.from({ length: 20 }, (_, index) => index + 1).map((count) => (
                  <option key={count} value={count}>
                    {count}
                  </option>
                ))}
              </Select>
            </Control>
          </FormGrid>
          <Button block loading={creating} onClick={() => void generatePairingCode()}>
            {t('settings.pairing.generate', 'Generate pairing code')}
          </Button>
        </>
      ) : (
        <QrPanel>
          <QrTitle>{t('settings.pairing.scanWithApp', 'Scan with Skulk Operator')}</QrTitle>
          <QrFrame>
            <QRCodeCanvas
              ref={canvasRef}
              bgColor="#FFFFFF"
              fgColor="#000000"
              imageSettings={{
                src: '/skulk-qr-mark.svg',
                height: 58,
                width: 58,
                excavate: false,
              }}
              level="M"
              marginSize={4}
              size={280}
              value={created.pairingCode}
            />
          </QrFrame>
          <QrMeta>
            <span>
              {t('settings.pairing.expires', 'Expires')} ·{' '}
              {new Date(created.invitation.expiresAt).toLocaleString()}
            </span>
            <span>
              {created.invitation.maxPairings}{' '}
              {created.invitation.maxPairings === 1
                ? t('settings.pairing.device', 'device')
                : t('settings.pairing.devices', 'devices')}
            </span>
            <span>
              {t('settings.pairing.visibleFor', 'Visible here for')} ·{' '}
              {formatCountdown(remainingSeconds)}
            </span>
          </QrMeta>
          <SecretWarning>
            {t(
              'settings.pairing.secretWarning',
              'Anyone with this code can attempt to pair until it expires or is revoked. Share it only with devices you trust.',
            )}
          </SecretWarning>
          <ButtonRow>
            <Button onClick={downloadQr} variant="outline">
              {t('settings.pairing.download', 'Download PNG')}
            </Button>
            <Button
              loading={revokeResult.isLoading}
              onClick={() => void revoke(created.invitation.invitationId)}
              variant="danger"
            >
              {t('settings.pairing.revokeNow', 'Revoke now')}
            </Button>
          </ButtonRow>
        </QrPanel>
      )}

      {error ? <ErrorText role="alert">{error}</ErrorText> : null}
      {invitationListFailed ? (
        <ErrorText role="status">
          {t(
            'settings.pairing.listUnavailable',
            'Invitation history is temporarily unavailable. New pairing codes can still be generated.',
          )}
        </ErrorText>
      ) : null}

      {invitations && invitations.length > 0 ? (
        <InvitationList>
          <ListTitle>{t('settings.pairing.recent', 'Recent invitations')}</ListTitle>
          {[...invitations].reverse().map((invitation) => (
            <InvitationRow key={invitation.invitationId}>
              <InvitationCopy>
                <InvitationHeadline>
                  <span>{new Date(invitation.createdAt).toLocaleDateString()}</span>
                  <StatePill $state={invitation.state}>
                    {stateLabel(invitation.state, t)}
                  </StatePill>
                </InvitationHeadline>
                <InvitationMeta>
                  {invitation.successfulPairings}/{invitation.maxPairings}{' '}
                  {t('settings.pairing.paired', 'paired')} ·{' '}
                  {t('settings.pairing.expires', 'Expires')}{' '}
                  {new Date(invitation.expiresAt).toLocaleString()}
                </InvitationMeta>
              </InvitationCopy>
              {invitation.state === 'active' ? (
                <Button
                  loading={revokeResult.isLoading}
                  onClick={() => void revoke(invitation.invitationId)}
                  size="sm"
                  variant="danger"
                >
                  {t('settings.pairing.revoke', 'Revoke')}
                </Button>
              ) : null}
            </InvitationRow>
          ))}
        </InvitationList>
      ) : null}
    </Fieldset>
  );
}

function formatCountdown(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, '0')}`;
}

function stateLabel(
  state: PairingInvitationState,
  t: (key: string, fallback: string) => string,
): string {
  switch (state) {
    case 'active':
      return t('settings.pairing.stateActive', 'active');
    case 'expired':
      return t('settings.pairing.stateExpired', 'expired');
    case 'exhausted':
      return t('settings.pairing.stateUsed', 'used');
    case 'attempt-limit':
      return t('settings.pairing.stateLimited', 'limited');
    case 'revoked':
      return t('settings.pairing.stateRevoked', 'revoked');
  }
}

function durationLabel(
  seconds: (typeof invitationLifetimeOptions)[number],
  t: (key: string, fallback: string) => string,
): string {
  switch (seconds) {
    case 5 * 60:
      return t('settings.pairing.durationFiveMinutes', '5 minutes');
    case 60 * 60:
      return t('settings.pairing.durationOneHour', '1 hour');
    case 24 * 60 * 60:
      return t('settings.pairing.durationOneDay', '1 day');
    case 7 * 24 * 60 * 60:
      return t('settings.pairing.durationSevenDays', '7 days');
    case 30 * 24 * 60 * 60:
      return t('settings.pairing.durationThirtyDays', '30 days');
    case 90 * 24 * 60 * 60:
      return t('settings.pairing.durationNinetyDays', '90 days');
  }
}
