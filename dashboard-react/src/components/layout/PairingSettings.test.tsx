import { act, forwardRef } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { darkTheme } from '../../theme/theme';
import { PairingSettings } from './PairingSettings';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });

const copyCodeMock = vi.hoisted(() => vi.fn());
vi.mock('../../utils/clipboard', () => ({ copyToClipboard: copyCodeMock }));

const createInvitationMock = vi.hoisted(() => vi.fn());
const getInvitationsQueryMock = vi.hoisted(() => vi.fn());
const refetchMock = vi.hoisted(() => vi.fn());
const revokeMock = vi.hoisted(() => vi.fn());
const capacityState = vi.hoisted(() => ({
  value: { activeDevices: 0, maximumDevices: 5, availableSlots: 5 } as
    | { activeDevices: number; maximumDevices: number; availableSlots: number }
    | undefined,
}));

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
  pairingInvitationQueryErrorDetail: () => null,
  PairingInvitationRequestError: class PairingInvitationRequestError extends Error {},
  useGetPairingInvitationsQuery: (argument: unknown, options: unknown) => {
    getInvitationsQueryMock(argument, options);
    return {
      data: [],
      error: undefined,
      isError: false,
      refetch: refetchMock,
    };
  },
  useRevokePairingInvitationMutation: () => [revokeMock, { isLoading: false }],
  useGetPairingCapacityQuery: () => ({ data: capacityState.value }),
}));

vi.mock('../../hooks/useToast', () => ({ addToast: vi.fn() }));

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string, parameters?: Record<string, string | number>) =>
      fallback.replace(/\{(\w+)\}/g, (placeholder, name: string) =>
        parameters && name in parameters ? String(parameters[name]) : placeholder,
      ),
  }),
}));

let root: Root | null = null;
let container: HTMLDivElement | null = null;

beforeEach(() => {
  capacityState.value = { activeDevices: 0, maximumDevices: 5, availableSlots: 5 };
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

    expect(createInvitationMock).toHaveBeenCalledWith(
      {
        validForSeconds: 300,
        maxPairings: 1,
      },
      expect.stringContaining('configured operator gateway'),
    );
    expect(container?.querySelector('canvas')?.getAttribute('data-pairing-code')).toBe(
      'skulk://pair?z=secret-bearing-code',
    );
    expect(container?.querySelector('canvas')?.getAttribute('data-error-correction')).toBe('M');
    expect(container?.querySelector('canvas')?.getAttribute('data-logo-src')).toBe(
      '/skulk-qr-mark.svg',
    );
    expect(container?.querySelector('canvas')?.getAttribute('data-logo-excavated')).toBe('false');
    expect(copyCodeMock).not.toHaveBeenCalled();
    const copyCode = [...(container?.querySelectorAll('button') ?? [])].find(button => button.textContent === 'Copy code');
    await act(async () => copyCode?.click());
    expect(copyCodeMock).toHaveBeenCalledExactlyOnceWith('skulk://pair?z=secret-bearing-code');
    expect(container?.textContent).not.toContain('secret-bearing-code');
    expect(container?.textContent).toContain('Visible here for · 5:00');
    expect(getInvitationsQueryMock).toHaveBeenCalledWith(undefined, {
      pollingInterval: 15_000,
      skipPollingIfUnfocused: true,
      refetchOnFocus: true,
      refetchOnReconnect: true,
    });

    await act(async () => vi.advanceTimersByTime(5 * 60 * 1_000));
    expect(container?.textContent).toContain('Generate pairing code');
    expect(container?.querySelector('canvas')).toBeNull();
  });

  it('stops new pairing at the device limit and says how to free a slot', async () => {
    capacityState.value = { activeDevices: 5, maximumDevices: 5, availableSlots: 0 };
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
    expect(generate?.disabled).toBe(true);
    expect(container?.textContent).toContain('5 of 5 device slots in use');
    expect(container?.textContent).toContain(
      'This cluster already has 5 paired devices, the most it allows. Revoke a device before pairing another.',
    );
    await act(async () => generate?.click());
    expect(createInvitationMock).not.toHaveBeenCalled();
  });

  it('offers only as many devices as there are free slots', async () => {
    capacityState.value = { activeDevices: 3, maximumDevices: 5, availableSlots: 2 };
    await act(async () => {
      root?.render(
        <ThemeProvider theme={darkTheme}>
          <PairingSettings />
        </ThemeProvider>,
      );
    });

    const devicesTrigger = container?.querySelector<HTMLButtonElement>(
      'button[aria-label="Devices"]',
    );
    await act(async () => devicesTrigger?.click());
    const choices = [...document.querySelectorAll('[role="option"]')].map(
      (option) => option.textContent,
    );
    expect(choices).toEqual(['1', '2']);
    expect(container?.textContent).toContain('3 of 5 device slots in use');
    expect(container?.textContent).not.toContain('the most it allows');
  });
});

