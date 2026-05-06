import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';
import { FiRefreshCw } from 'react-icons/fi';
import { Button } from '../common/Button';
import { QrCode } from '../common/QrCode';
import { useRemoteAccess, type RemoteAccessInfo } from '../../hooks/useRemoteAccess';

const TAILSCALE_DOCS_URL = 'https://foxlight-foundation.github.io/Skulk/tailscale/';
const QR_RENEW_EARLY_MS = 30_000;
const PAIRING_OPERATOR_TOKEN_STORAGE_KEY = 'skulkCompanionPairingToken';

interface CompanionPairingQrPayload {
  version: 1;
  clusterId: string;
  clusterName: string;
  pairingNonce: string;
  expiresAt: string;
  exchangeUrl: string;
  clusterPublicKey: string;
  lanUrl: string | null;
  tailscaleUrl: string | null;
  preferredUrl: string | null;
}

interface CompanionPairingSessionResponse {
  qrPayload: CompanionPairingQrPayload;
}

type SessionState =
  | { status: 'idle' | 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; payload: CompanionPairingQrPayload };

const Scroll = styled.div`
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding-right: 2px;
`;

const Section = styled.section`
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const Header = styled.div`
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
`;

const TitleGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
`;

const Title = styled.h3`
  margin: 0;
  color: ${({ theme }) => theme.colors.text};
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.lg};
`;

const BodyText = styled.p`
  margin: 0;
  color: ${({ theme }) => theme.colors.textSecondary};
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  line-height: 1.5;
`;

const MutedText = styled(BodyText)`
  color: ${({ theme }) => theme.colors.textMuted};
`;

const StatusGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
`;

const StatusCell = styled.div`
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  background: ${({ theme }) => theme.colors.surfaceSunken};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 10px;
  min-width: 0;
`;

const Label = styled.div`
  color: ${({ theme }) => theme.colors.textMuted};
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  margin-bottom: 4px;
`;

const Value = styled.div<{ $warn?: boolean; $ok?: boolean }>`
  color: ${({ $ok, $warn, theme }) =>
    $ok ? theme.colors.healthy : $warn ? theme.colors.warningOnSurface : theme.colors.text};
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  overflow-wrap: anywhere;
`;

const Advisory = styled.div<{ $level: 'info' | 'warning' | 'error' }>`
  border: 1px solid
    ${({ $level, theme }) =>
      $level === 'error'
        ? theme.colors.errorBg
        : $level === 'warning'
          ? theme.colors.warningBg
          : theme.colors.borderLight};
  background: ${({ $level, theme }) =>
    $level === 'error'
      ? theme.colors.errorBg
      : $level === 'warning'
        ? theme.colors.warningBg
        : theme.colors.infoBg};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 10px 12px;
  color: ${({ $level, theme }) =>
    $level === 'error'
      ? theme.colors.errorOnSurface
      : $level === 'warning'
        ? theme.colors.warningOnSurface
        : theme.colors.textSecondary};
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  line-height: 1.45;
`;

const Link = styled.a`
  color: ${({ theme }) => theme.colors.gold};
  text-decoration: none;

  &:hover {
    text-decoration: underline;
  }
`;

const QrWrap = styled.div`
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
`;

const Instructions = styled.ol`
  margin: 0;
  padding-left: 18px;
  color: ${({ theme }) => theme.colors.textSecondary};
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  line-height: 1.55;
`;

function pairingReadiness(access: RemoteAccessInfo): {
  canPair: boolean;
  level: 'info' | 'warning' | 'error';
  message: string;
} {
  if (!access.preferredUrl) {
    return {
      canPair: false,
      level: 'error',
      message: 'Skulk does not currently have a reachable LAN or Tailscale URL for companion pairing.',
    };
  }
  if (!access.tailscale.running || !access.tailscale.url) {
    return {
      canPair: true,
      level: 'warning',
      message: 'Pairing is available on this local network. Remote access away from home needs Tailscale on this cluster node and on the phone.',
    };
  }
  return {
    canPair: true,
    level: 'info',
    message: 'Tailscale is ready. SkulkOps can use the private Tailscale URL after pairing.',
  };
}

async function createPairingSession(signal: AbortSignal): Promise<CompanionPairingQrPayload> {
  const operatorToken = window.localStorage.getItem(PAIRING_OPERATOR_TOKEN_STORAGE_KEY);
  const response = await fetch('/v1/companion/pairing-sessions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(operatorToken ? { 'X-Skulk-Operator-Token': operatorToken } : {}),
    },
    body: JSON.stringify({}),
    signal,
  });
  if (!response.ok) {
    if (response.status === 403) {
      throw new Error(
        'Pairing requires a local dashboard session. If you are opening this dashboard over LAN or Tailscale, configure SKULK_COMPANION_PAIRING_TOKEN and store it in this browser as skulkCompanionPairingToken.',
      );
    }
    throw new Error('Could not create a SkulkOps pairing code.');
  }
  const session = (await response.json()) as CompanionPairingSessionResponse;
  return session.qrPayload;
}

function formatTimeRemaining(expiresAt: string, nowMs: number): string {
  const remainingMs = new Date(expiresAt).getTime() - nowMs;
  if (!Number.isFinite(remainingMs) || remainingMs <= 0) return 'rotating now';
  const totalSeconds = Math.ceil(remainingMs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

export function PairingTab() {
  const access = useRemoteAccess();
  const [session, setSession] = useState<SessionState>({ status: 'idle' });
  const [nowMs, setNowMs] = useState(() => Date.now());
  const mountedRef = useRef(false);
  const requestIdRef = useRef(0);

  const readiness = useMemo(
    () => (access.status === 'ok' ? pairingReadiness(access.data) : null),
    [access],
  );

  const loadSession = useCallback(
    async (signal: AbortSignal, options: { showLoading?: boolean } = {}) => {
      if (!readiness?.canPair) {
        setSession({ status: 'idle' });
        return;
      }
      const requestId = ++requestIdRef.current;
      if (options.showLoading ?? true) setSession({ status: 'loading' });
      try {
        const payload = await createPairingSession(signal);
        if (!signal.aborted && mountedRef.current && requestId === requestIdRef.current) {
          setSession({ status: 'ready', payload });
        }
      } catch (error) {
        if (signal.aborted || !mountedRef.current || requestId !== requestIdRef.current) return;
        setSession({
          status: 'error',
          message: error instanceof Error ? error.message : 'Could not create a SkulkOps pairing code.',
        });
      }
    },
    [readiness],
  );

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (!readiness?.canPair) {
      setSession({ status: 'idle' });
      return;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      void loadSession(controller.signal, { showLoading: true });
    }, 0);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [loadSession, readiness]);

  useEffect(() => {
    if (session.status !== 'ready') return;
    const expiresMs = new Date(session.payload.expiresAt).getTime();
    if (!Number.isFinite(expiresMs)) return;
    const delay = Math.max(1000, expiresMs - Date.now() - QR_RENEW_EARLY_MS);
    let controller: AbortController | null = null;
    const timer = window.setTimeout(() => {
      controller = new AbortController();
      void loadSession(controller.signal, { showLoading: false });
    }, delay);
    return () => {
      window.clearTimeout(timer);
      controller?.abort();
    };
  }, [loadSession, session]);

  if (access.status === 'loading') {
    return (
      <Scroll>
        <Section>
          <Title>SkulkOps Pairing</Title>
          <MutedText>Checking local reachability...</MutedText>
        </Section>
      </Scroll>
    );
  }

  if (access.status === 'error') {
    return (
      <Scroll>
        <Section>
          <Title>SkulkOps Pairing</Title>
          <Advisory $level="error">Could not read this node's pairing readiness.</Advisory>
        </Section>
      </Scroll>
    );
  }

  const remaining = session.status === 'ready'
    ? formatTimeRemaining(session.payload.expiresAt, nowMs)
    : null;

  return (
    <Scroll>
      <Section>
        <Header>
          <TitleGroup>
            <Title>SkulkOps Pairing</Title>
            <BodyText>Scan this code from SkulkOps to pair a phone with this cluster.</BodyText>
          </TitleGroup>
          {session.status === 'loading' && (
            <Button variant="ghost" size="sm" loading aria-label="Creating pairing code" />
          )}
          {session.status === 'error' && readiness.canPair && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                const controller = new AbortController();
                void loadSession(controller.signal);
              }}
            >
              <FiRefreshCw size={14} /> Retry
            </Button>
          )}
        </Header>

        <StatusGrid>
          <StatusCell>
            <Label>Tailscale</Label>
            <Value $ok={access.data.tailscale.running} $warn={!access.data.tailscale.running}>
              {access.data.tailscale.running
                ? access.data.tailscale.url ?? 'running'
                : 'not running'}
            </Value>
          </StatusCell>
          <StatusCell>
            <Label>LAN</Label>
            <Value>{access.data.local.url ?? 'unavailable'}</Value>
          </StatusCell>
        </StatusGrid>

        <Advisory $level={readiness.level}>
          {readiness.message}{' '}
          {(!access.data.tailscale.running || !access.data.tailscale.url) && (
            <Link href={TAILSCALE_DOCS_URL} target="_blank" rel="noreferrer">
              Set up Tailscale.
            </Link>
          )}
        </Advisory>

        {!readiness.canPair && (
          <MutedText>Pairing will become available once Skulk can advertise a reachable URL.</MutedText>
        )}

        {session.status === 'error' && (
          <Advisory $level="error">{session.message}</Advisory>
        )}

        {session.status === 'ready' && readiness.canPair && (
          <QrWrap>
            <QrCode
              value={JSON.stringify(session.payload)}
              alt="SkulkOps companion pairing QR code"
            />
            <div>
              <Instructions>
                <li>Open SkulkOps on your phone.</li>
                <li>Tap Scan QR.</li>
                <li>Point the camera at this code.</li>
              </Instructions>
              <MutedText>
                This code renews automatically while this tab is open. Current code rotates in {remaining}.
              </MutedText>
            </div>
          </QrWrap>
        )}
      </Section>
    </Scroll>
  );
}
