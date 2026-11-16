import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { CatalogEntry, NativeCatalogBinding, RuntimeRelease } from '../../store/endpoints/plugins';
import { darkTheme } from '../../theme/theme';
import { CatalogReviewPanel, CatalogRetryReviewPanel } from './CatalogReviewPanel';

vi.mock('../../i18n/tolgee', () => ({ useSkulkTranslation: () => ({
  t: (_key: string, fallback: string, values?: Record<string, unknown>) => fallback.replace(/\{(\w+)\}/g, (_match, key: string) => String(values?.[key] ?? '')),
}) }));
Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });

const native: NativeCatalogBinding = {
  runtime_protocol: 5, release_protocol: 3, manifest_revision: 'c'.repeat(64),
  kernel: { version: '0.7.0rc4', source_commit: 'f'.repeat(40), portable: 'a'.repeat(64), managed: 'b'.repeat(64), handshake: 'd'.repeat(64), transport: 'e'.repeat(64), composition: '1'.repeat(64), calls: '2'.repeat(64) },
  processing_domains: ['external', 'local', 'private_site'],
};
const entry: CatalogEntry = {
  bundle_id: 'example.native', bundle_version: '1.0.0', title: 'Native Integrator', publisher: 'fixture', sequence: 1,
  release_digest: 'a'.repeat(64), runtime_platform: 'macos-arm64', artifact_sha256: 'b'.repeat(64), artifact_bytes: 1, transfer_bytes: 1,
  platforms: ['darwin'], skulk_build_sha256: 'c'.repeat(64), skulk_requires: '>=2.0.3', permissions: ['Use the connected provider'],
  descriptors: ['example.generate@1.0.0'], surfaces: [], operations: false, steward_risks: ['inference'], expires_at: 1_900_000_000, matches_host: true, native,
};
const review: RuntimeRelease = { runtime_digest: entry.release_digest, source_revision: 1, publisher: entry.publisher, bundle_id: entry.bundle_id,
  version: entry.bundle_version, sequence: entry.sequence, platform: 'macos-arm64', python_requires: '==3.13.*', skulk_build_sha256: entry.skulk_build_sha256,
  permissions: entry.permissions, artifact_bytes: entry.transfer_bytes, expires_at: entry.expires_at, native };
let host: HTMLDivElement;
let root: Root;
beforeEach(() => { host = document.createElement('div'); document.body.appendChild(host); root = createRoot(host); });
afterEach(async () => { await act(async () => root.unmount()); host.remove(); });

it.each(['catalog', 'retry', 'finish'] as const)('discloses native processing and exact identities before %s consent', async (path) => {
  const submit = vi.fn();
  await act(async () => root.render(<ThemeProvider theme={darkTheme}>{path === 'catalog'
    ? <CatalogReviewPanel offer={{ entry, installed: null, state: 'available' }} onInstall={submit} onCancel={() => {}} />
    : <CatalogRetryReviewPanel title="Native Integrator" updating={false} stopped={path === 'retry'} review={review} onRetry={submit} onCancel={() => {}} />}</ThemeProvider>));
  const section = host.querySelector('section[aria-label="Declared processing locations"]');
  expect(section?.checkVisibility()).toBe(true);
  expect(section?.textContent).toContain('External services');
  expect(section?.textContent).toContain('On this host');
  expect(section?.textContent).toContain('Private deployment');
  expect(section?.textContent).toContain('Requests and their content may be sent to external services.');
  const action = [...host.querySelectorAll('button')].find((button) => button.textContent !== 'Cancel');
  expect(action?.disabled).toBe(true);
  const details = host.querySelector('details');
  expect(details?.open).toBe(false);
  await act(async () => details?.querySelector('summary')?.click());
  expect(details?.open).toBe(true);
  expect(details?.textContent).toContain(native.manifest_revision);
  expect(details?.textContent).toContain(native.kernel.source_commit);
  expect(details?.textContent).toContain(native.kernel.version);
  expect(details?.textContent).toContain(native.kernel.calls);
  expect(submit).not.toHaveBeenCalled();
  await act(async () => host.querySelector<HTMLInputElement>('input[type="checkbox"]')?.click());
  expect(action?.disabled).toBe(false);
  await act(async () => action?.click());
  expect(submit).toHaveBeenCalledTimes(1);
});
