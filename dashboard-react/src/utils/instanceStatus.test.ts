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

  it('reports the video engine install and its size while the runner waits idle', () => {
    expect(deriveInstanceStatus(['runner-1'], idle, t, 7 * 1024 ** 3)).toEqual({
      status: 'loading',
      message: 'Installing video engine (about 7 GB)...',
    });
  });

  it('reports the install without a size when the task carries none', () => {
    expect(deriveInstanceStatus(['runner-1'], idle, t, 0)).toEqual({
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
    expect(deriveInstanceStatus(['runner-1'], loading, t, 7 * 1024 ** 3).progress).toBe(50);
  });

  it('keeps a failed runner failed', () => {
    const failed = { 'runner-1': { RunnerFailed: { errorMessage: 'boom' } } };
    expect(deriveInstanceStatus(['runner-1'], failed, t, 7 * 1024 ** 3)).toEqual({
      status: 'failed',
      message: 'boom',
    });
  });
});
