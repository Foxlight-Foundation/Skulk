import { describe, expect, it } from 'vitest';
import type { CatalogEntry, CatalogInstallation, CatalogListing, ManagedRuntime, RuntimeInstallation } from '../../store/endpoints/plugins';
import {
  catalogOffers, clearJourney, decodeInvitation, encodeInvitation, isStartRefusal, parseVideoReadiness,
  platformLabel, readJourneys, saveJourney, startCatalogInstall, type InstallJourney,
} from './catalogJourney';

function entry(sequence: number, overrides: Partial<CatalogEntry> = {}): CatalogEntry {
  return {
    bundle_id: 'foxlight.video-studio', bundle_version: '0.1.0', title: 'Skulk Video Studio', publisher: 'foxlight', sequence,
    release_digest: 'd'.repeat(64), runtime_platform: 'macos-arm64', artifact_sha256: 'a'.repeat(64), artifact_bytes: 4096,
    transfer_bytes: 12_086_479, platforms: ['darwin'], skulk_build_sha256: 'b'.repeat(64), permissions: ['Render video on the fleet through the host API'],
    descriptors: ['video.readiness@1.0.0', 'video.render@1.0.0'], surfaces: ['Skulk Video Studio (MiniMax H3)'], operations: true,
    steward_risks: ['observation'], expires_at: 1_900_000_000, matches_host: true, ...overrides,
  };
}
function listing(entries: CatalogEntry[]): CatalogListing {
  return { publisher: 'foxlight', revision: 34, created_at: 1_790_000_000, expires_at: 1_800_000_000, catalog_sha256: 'c'.repeat(64), entries };
}
function runtime(sequence: number | null, overrides: Partial<ManagedRuntime> = {}): ManagedRuntime {
  return {
    plugin_id: 'managed.' + '2'.repeat(32), release: sequence === null ? null : { bundle_id: 'foxlight.video-studio', title: 'Skulk Video Studio', bundle_version: '0.1.0', publisher: 'foxlight', sequence },
    selected_digest: 'e'.repeat(64), selection_revision: 3, enabled: true, stale: false, error_code: null, operation_id: null, operation_state: null,
    service: { state: 'running', active_digest: 'e'.repeat(64), observed_at: 1 }, ...overrides,
  };
}
function memoryStorage(): Storage {
  const values = new Map<string, string>();
  return { getItem: (key) => values.get(key) ?? null, setItem: (key, value) => { values.set(key, value); }, removeItem: (key) => { values.delete(key); }, clear: () => values.clear(), key: () => null, get length() { return values.size; } };
}

describe('catalogOffers', () => {
  it('offers a fitting release, or says the listing is not for this host', () => {
    expect(catalogOffers(listing([entry(50)]), [])).toMatchObject([{ state: 'available', entry: { sequence: 50 } }]);
    expect(catalogOffers(listing([entry(50, { matches_host: false })]), [])).toMatchObject([{ state: 'unfit' }]);
  });

  it('offers an update over an installed older release, including one broken by a Skulk update', () => {
    const broken = runtime(49, { service: { state: 'failed', active_digest: null, observed_at: 1, error_code: 'verification_failed' } });
    const [offer] = catalogOffers(listing([entry(49, { matches_host: false }), entry(50)]), [broken]);
    expect(offer).toMatchObject({ state: 'update', entry: { sequence: 50 }, installed: { plugin_id: broken.plugin_id } });
  });

  it('shows an up-to-date installation as installed, and matches only by the selected release', () => {
    expect(catalogOffers(listing([entry(50)]), [runtime(50)])).toMatchObject([{ state: 'installed' }]);
    // An installation with no readable release names no bundle, so it is not this one.
    expect(catalogOffers(listing([entry(51)]), [runtime(null)])).toMatchObject([{ state: 'available', installed: null }]);
    // An uninstalled record keeps nothing installed.
    expect(catalogOffers(listing([entry(50)]), [runtime(49, { uninstalled: true, enabled: false })])).toMatchObject([{ state: 'available', installed: null }]);
  });
});

describe('platformLabel', () => {
  it('names artifact families and falls back to operating systems', () => {
    expect(platformLabel(entry(1))).toBe('macOS, Apple Silicon');
    expect(platformLabel(entry(1, { runtime_platform: null, platforms: ['darwin', 'linux'] }))).toBe('macOS or Linux');
  });
});

describe('invitations', () => {
  const invitation = { baseUrl: 'https://kite5.example.ts.net/', publisher: 'foxlight', publicKey: 'f'.repeat(64), trustExpiresAt: 2_000_000_000 };

  it('round-trips the discovery facts and an optional credential', () => {
    expect(decodeInvitation(encodeInvitation(invitation), 1_900_000_000)).toEqual(invitation);
    expect(decodeInvitation(encodeInvitation({ ...invitation, token: 'secret-token' }), 1_900_000_000)).toEqual({ ...invitation, token: 'secret-token' });
  });

  it('refuses anything that is not a current, well-formed invitation', () => {
    expect(decodeInvitation('https://kite5.example.ts.net/', 1)).toBeNull();
    expect(decodeInvitation('skulk-catalog:not+base64', 1)).toBeNull();
    expect(decodeInvitation(encodeInvitation(invitation), 2_000_000_001)).toBeNull();
    expect(decodeInvitation(encodeInvitation({ ...invitation, baseUrl: 'http://plain.example/' }), 1)).toBeNull();
    expect(decodeInvitation(encodeInvitation({ ...invitation, publicKey: 'short' }), 1)).toBeNull();
  });
});

describe('startCatalogInstall', () => {
  const offer = catalogOffers(listing([entry(51)]), [runtime(50)])[0];
  const bound: CatalogInstallation = {
    plugin_id: 'managed.' + '2'.repeat(32), listing: entry(51),
    source: { revision: 7, configured: true, credential_reference: null, credential_ready: true, trust_revision: 1 },
    review: { runtime_digest: '9'.repeat(64), source_revision: 7, publisher: 'foxlight', bundle_id: 'foxlight.video-studio', version: '0.1.0', sequence: 51, platform: 'macos-arm64', python_requires: '>=3.13', skulk_build_sha256: 'b'.repeat(64), permissions: [], artifact_bytes: 12_000_000, expires_at: 1_900_000_000 },
  };
  const resolved = <T,>(value: T) => ({ unwrap: () => Promise.resolve(value) });
  const rejected = (error: unknown) => ({ unwrap: () => Promise.reject(error) });

  it('upgrades the installation in place and records the journey before the download is sent', async () => {
    const bindCalls: unknown[] = [];
    const installCalls: { pluginId: string; request: RuntimeInstallation['request'] }[] = [];
    globalThis.localStorage.removeItem('skulk-plugin-install-journeys');
    const outcome = await startCatalogInstall({
      bind: (request) => { bindCalls.push(request); return resolved(bound); },
      install: (args) => { installCalls.push(args); expect(readJourneys().map((item) => item.installOperationId)).toContain(args.request.operation_id); return resolved({} as RuntimeInstallation); },
    }, offer, listing([entry(51)]), () => null, () => '1'.repeat(32));
    expect(bindCalls).toEqual([{ catalog_sha256: 'c'.repeat(64), bundle_id: 'foxlight.video-studio', sequence: 51, runtime_platform: 'macos-arm64', plugin_id: offer.installed!.plugin_id }]);
    expect(installCalls).toEqual([{ pluginId: bound.plugin_id, request: { operation_id: '1'.repeat(32), runtime_digest: '9'.repeat(64), expected_source_revision: 7 } }]);
    expect(isStartRefusal(outcome)).toBe(false);
    expect((outcome as InstallJourney).activationOperationId).toBeNull();
    clearJourney(bound.plugin_id);
    expect(readJourneys()).toEqual([]);
  });

  it('returns the server sentence for a refused bind, and keeps a journey when the download reply is lost', async () => {
    const detail = { status: 409, data: { detail: 'This host could not reach the plugin catalog.' } };
    const refused = await startCatalogInstall({ bind: () => rejected(detail), install: () => resolved({} as RuntimeInstallation) }, offer, listing([entry(51)]), (error) => (error as typeof detail).data.detail, () => '1'.repeat(32));
    expect(refused).toEqual({ stage: 'bind', detail: 'This host could not reach the plugin catalog.' });
    const lost = await startCatalogInstall({ bind: () => resolved(bound), install: () => rejected(new TypeError('network')) }, offer, listing([entry(51)]), () => null, () => '1'.repeat(32));
    expect(isStartRefusal(lost)).toBe(false);
    expect(readJourneys().map((item) => item.pluginId)).toContain(bound.plugin_id);
    clearJourney(bound.plugin_id);
  });
});

describe('journeys and readiness', () => {
  it('keeps journeys per installation and ignores malformed storage', () => {
    const storage = memoryStorage();
    const journey: InstallJourney = { pluginId: 'managed.a', title: 'A', bundleId: 'a', sequence: 1, publisher: 'p', runtimeDigest: 'd', transferBytes: 1, installOperationId: 'o', activationOperationId: null, descriptors: [], startedAt: 1 };
    saveJourney(journey, storage);
    saveJourney({ ...journey, activationOperationId: 'x' }, storage);
    expect(readJourneys(storage)).toEqual([{ ...journey, activationOperationId: 'x' }]);
    clearJourney('managed.a', storage);
    expect(readJourneys(storage)).toEqual([]);
    storage.setItem('skulk-plugin-install-journeys', '{broken');
    expect(readJourneys(storage)).toEqual([]);
  });

  it('reads the Studio readiness report defensively', () => {
    expect(parseVideoReadiness({ ready: true, video_models: ['m'], lanes: [{ node: 'kite4', backends: ['comfy-rocm'] }, { bad: 1 }], default_model: 'm', default_host: 'kite4', reasons: [] }))
      .toEqual({ ready: true, api_reachable: true, video_models: ['m'], lanes: [{ node: 'kite4', backends: ['comfy-rocm'], placed_models: [] }], default_model: 'm', default_host: 'kite4', reasons: [] });
    expect(parseVideoReadiness({ lanes: [] })).toBeNull();
  });
});
