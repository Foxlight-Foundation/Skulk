import { describe, expect, it } from 'vitest';

import type { SkulkTranslate } from '../i18n/tolgee';
import { deriveInstanceStatus } from './instanceStatus';

// Tolgee's translator type is heavily overloaded; the status helper only uses
// the key, the English fallback, and named parameters.
const t = ((_key: string, fallback: string, params?: Record<string, unknown>) =>
  params
    ? fallback.replace(/\{(\w+)\}/g, (_, name: string) => String(params[name] ?? ''))
    : fallback) as unknown as SkulkTranslate;

describe('deriveInstanceStatus', () => {
  const idle = { 'runner-1': { RunnerIdle: {} } };

  it('reports the video engine install while the runner waits idle', () => {
    expect(deriveInstanceStatus(['runner-1'], idle, t, true)).toEqual({
      status: 'loading',
      message: 'Installing video engine...',
    });
  });

  it('reads an idle runner without an install as connecting', () => {
    expect(deriveInstanceStatus(['runner-1'], idle, t)).toEqual({
      status: 'loading',
      message: 'Connecting...',
    });
  });

  it('lets a loading runner win over the install message', () => {
    const loading = { 'runner-1': { RunnerLoading: { layersLoaded: 2, totalLayers: 4 } } };
    expect(deriveInstanceStatus(['runner-1'], loading, t, true).progress).toBe(50);
  });

  it('keeps a failed runner failed', () => {
    const failed = { 'runner-1': { RunnerFailed: { errorMessage: 'boom' } } };
    expect(deriveInstanceStatus(['runner-1'], failed, t, true)).toEqual({
      status: 'failed',
      message: 'boom',
    });
  });
});
