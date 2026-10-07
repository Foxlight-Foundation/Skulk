import { describe, expect, it } from 'vitest';

import type { SkulkTranslate } from '../i18n/tolgee';
import { engineInstallFromPlacement, engineInstallNotice, engineInstallSize } from './engineInstall';

const t = ((_key: string, fallback: string, params?: Record<string, unknown>) =>
  params
    ? fallback.replace(/\{(\w+)\}/g, (_, name: string) => String(params[name] ?? ''))
    : fallback) as unknown as SkulkTranslate;

describe('engineInstallSize', () => {
  it('rounds to whole gigabytes like the server notice', () => {
    expect(engineInstallSize(7 * 1024 ** 3)).toBe('7 GB');
    expect(engineInstallSize(6.6 * 1024 ** 3)).toBe('7 GB');
    expect(engineInstallSize(200 * 1024 ** 2)).toBe('1 GB');
  });

  it('has no size for an unknown download', () => {
    expect(engineInstallSize(0)).toBeNull();
    expect(engineInstallSize(Number.NaN)).toBeNull();
  });
});

describe('engineInstallNotice', () => {
  it('words the video engine install and falls back to the server notice', () => {
    const comfy = { engine: 'comfy', node_ids: ['a'], approximate_download_bytes: 7 * 1024 ** 3, detail: 'server' };
    expect(engineInstallNotice(t, comfy)).toBe(
      'The video engine (about 7 GB) will be installed with this model, so placement will take longer.',
    );
    expect(engineInstallNotice(t, { ...comfy, approximate_download_bytes: 0 })).toBe('server');
    expect(engineInstallNotice(t, { ...comfy, engine: 'future' })).toBe('server');
  });
});

describe('engineInstallFromPlacement', () => {
  it('reads the install from an accepted placement', () => {
    expect(engineInstallFromPlacement({
      instance_id: 'i',
      engine_install: { engine: 'comfy', node_ids: ['a', 7], approximate_download_bytes: 5, detail: 'd' },
    })).toEqual({ engine: 'comfy', node_ids: ['a'], approximate_download_bytes: 5, detail: 'd' });
  });

  it('reads nothing from older servers or malformed bodies', () => {
    expect(engineInstallFromPlacement({ instance_id: 'i' })).toBeNull();
    expect(engineInstallFromPlacement({ engine_install: null })).toBeNull();
    expect(engineInstallFromPlacement({ engine_install: { engine: 'comfy' } })).toBeNull();
    expect(engineInstallFromPlacement(null)).toBeNull();
  });
});
