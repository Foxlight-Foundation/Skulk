import { act, forwardRef } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { darkTheme } from '../../theme/theme';
import { PairingSettings } from './PairingSettings';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const createInvitationMock = vi.hoisted(() => vi.fn());
const refetchMock = vi.hoisted(() => vi.fn());
const revokeMock = vi.hoisted(() => vi.fn());

vi.mock('qrcode.react', () => ({
  QRCodeCanvas: forwardRef<
    HTMLCanvasElement,
    {
      value: string;
      level: string;
      imageSettings: { src: string; excavate: boolean };
    }
  >(({ value, level, imageSettings }, ref) => (
    <canvas
      data-error-correction={level}
      data-logo-excavated={String(imageSettings.excavate)}
      data-logo-src={imageSettings.src}
      data-pairing-code={value}
      ref={ref}
    />
  )),
}));

vi.mock('../../store/endpoints/pairing', () => ({
  createPairingInvitation: createInvitationMock,
  PairingInvitationRequestError: class PairingInvitationRequestError extends Error {},
  useGetPairingInvitationsQuery: () => ({
    data: [],
    isError: false,
    refetch: refetchMock,
  }),
  useRevokePairingInvitationMutation: () => [revokeMock, { isLoading: false }],
}));

vi.mock('../../hooks/useToast', () => ({ addToast: vi.fn() }));

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}));

let root: Root | null = null;
let container: HTMLDivElement | null = null;

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date('2026-08-13T12:00:00.000Z'));
  createInvitationMock.mockResolvedValue({
    pairingCode: 'skulk://pair?z=secret-bearing-code',
    invitation: {
      invitationId: '00000000-0000-4000-8000-000000000001',
      createdAt: '2026-08-13T12:00:00.000Z',
      expiresAt: '2026-08-20T12:00:00.000Z',
      successfulPairings: 0,
      maxPairings: 1,
      activeAttempts: 0,
      totalAttempts: 0,
      state: 'active',
    },
  });
  refetchMock.mockResolvedValue(undefined);
  revokeMock.mockReturnValue({ unwrap: vi.fn().mockResolvedValue(undefined) });
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  container = null;
  root = null;
  vi.useRealTimers();
  vi.clearAllMocks();
});

describe('PairingSettings', () => {
  it('keeps the bearer code local to the QR and resets after five minutes', async () => {
    await act(async () => {
      root?.render(
        <ThemeProvider theme={darkTheme}>
          <PairingSettings />
        </ThemeProvider>,
      );
    });

    const generate = [...(container?.querySelectorAll('button') ?? [])].find(
      (button) => button.textContent === 'Generate pairing code',
    );
    await act(async () => generate?.click());

    expect(createInvitationMock).toHaveBeenCalledWith({
      validForSeconds: 300,
      maxPairings: 1,
    });
    expect(container?.querySelector('canvas')?.getAttribute('data-pairing-code')).toBe(
      'skulk://pair?z=secret-bearing-code',
    );
    expect(container?.querySelector('canvas')?.getAttribute('data-error-correction')).toBe('M');
    expect(container?.querySelector('canvas')?.getAttribute('data-logo-src')).toBe(
      '/skulk-qr-mark.svg',
    );
    expect(container?.querySelector('canvas')?.getAttribute('data-logo-excavated')).toBe('false');
    expect(container?.textContent).not.toContain('secret-bearing-code');
    expect(container?.textContent).toContain('Visible here for · 5:00');

    await act(async () => vi.advanceTimersByTime(5 * 60 * 1_000));
    expect(container?.textContent).toContain('Generate pairing code');
    expect(container?.querySelector('canvas')).toBeNull();
  });
});
