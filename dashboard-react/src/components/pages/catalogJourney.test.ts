import { beforeEach, describe, expect, it } from 'vitest';
import type { CatalogEntry, CatalogInstallation, CatalogListing, ManagedRuntime, RuntimeInstallation } from '../../store/endpoints/plugins';
import {
  catalogOffers, clearJourney, decodeInvitation, encodeInvitation, isStartRefusal, journeyInProgress,
  platformLabel, readJourneys, saveJourney, startCatalogInstall, type InstallJourney,
} from './catalogJourney';

function entry(sequence: number, overrides: Partial<CatalogEntry> = {}): CatalogEntry {
  return {
    bundle_id: 'example.studio', bundle_version: '0.1.0', title: 'Example Studio', publisher: 'example', sequence,
    release_digest: 'd'.repeat(64), runtime_platform: 'macos-arm64', artifact_sha256: 'a'.repeat(64), artifact_bytes: 4096,
    transfer_bytes: 12_086_479, platforms: ['darwin'], skulk_build_sha256: 'b'.repeat(64), permissions: ['Use models on the fabric through the host API'],
    descriptors: ['studio.plan@1.0.0', 'studio.render@1.0.0'], surfaces: ['Example Studio'], operations: true,
    steward_risks: ['observation'], expires_at: 1_900_000_000, matches_host: true, ...overrides,
  };
}
function listing(entries: CatalogEntry[]): CatalogListing {
  return { publisher: 'example', revision: 34, created_at: 1_790_000_000, expires_at: 1_800_000_000, catalog_sha256: 'c'.repeat(64), entries };
}
function runtime(sequence: number | null, overrides: Partial<ManagedRuntime> = {}): ManagedRuntime {
  return {
    plugin_id: 'managed.' + '2'.repeat(32), release: sequence === null ? null : { bundle_id: 'example.studio', title: 'Example Studio', bundle_version: '0.1.0', publisher: 'example', sequence },
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
  const invitation = { baseUrl: 'https://catalog.example.ts.net/', publisher: 'example', publicKey: 'f'.repeat(64), trustExpiresAt: 2_000_000_000 };

  it('round-trips the discovery facts and an optional credential', () => {
    expect(decodeInvitation(encodeInvitation(invitation), 1_900_000_000)).toEqual(invitation);
    expect(decodeInvitation(encodeInvitation({ ...invitation, token: 'secret-token' }), 1_900_000_000)).toEqual({ ...invitation, token: 'secret-token' });
  });

  it('refuses anything that is not a current, well-formed invitation', () => {
    expect(decodeInvitation('https://catalog.example.ts.net/', 1)).toBeNull();
    expect(decodeInvitation('skulk-catalog:not+base64', 1)).toBeNull();
    expect(decodeInvitation(encodeInvitation(invitation), 2_000_000_001)).toBeNull();
    expect(decodeInvitation(encodeInvitation({ ...invitation, baseUrl: 'http://plain.example/' }), 1)).toBeNull();
    expect(decodeInvitation(encodeInvitation({ ...invitation, publicKey: 'short' }), 1)).toBeNull();
  });
});

describe('startCatalogInstall', () => {
  const offer = catalogOffers(listing([entry(51)]), [runtime(50)])[0];
  const fresh = catalogOffers(listing([entry(51)]), [])[0];
  const bound = (pluginId: string): CatalogInstallation => ({
    plugin_id: pluginId, listing: entry(51),
    source: { revision: 7, configured: true, credential_reference: null, credential_ready: true, trust_revision: 1 },
    review: { runtime_digest: '9'.repeat(64), source_revision: 7, publisher: 'example', bundle_id: 'example.studio', version: '0.1.0', sequence: 51, platform: 'macos-arm64', python_requires: '>=3.13', skulk_build_sha256: 'b'.repeat(64), permissions: [], artifact_bytes: 12_000_000, expires_at: 1_900_000_000 },
  });
  const resolved = <T,>(value: T) => ({ unwrap: () => Promise.resolve(value) });
  const rejected = (error: unknown) => ({ unwrap: () => Promise.reject(error) });
  const statusOf = (error: unknown) => { const status = (error as { status?: unknown }).status; return typeof status === 'number' && status < 500 ? status : null; };
  const ids = (...values: string[]) => { const queue = [...values]; return () => queue.shift() ?? 'f'.repeat(32); };
  beforeEach(() => globalThis.localStorage.removeItem('skulk-plugin-install-journeys'));

  it('upgrades the installation in place, saving the journey before each request', async () => {
    const bindCalls: unknown[] = [];
    const installCalls: { pluginId: string; request: RuntimeInstallation['request'] }[] = [];
    const outcome = await startCatalogInstall({
      bind: (request) => {
        bindCalls.push(request);
        expect(readJourneys()).toMatchObject([{ pluginId: offer.installed!.plugin_id, runtimeDigest: null }]);
        return resolved(bound(offer.installed!.plugin_id));
      },
      install: (args) => { installCalls.push(args); expect(readJourneys().map((item) => item.installOperationId)).toContain(args.request.operation_id); return resolved({} as RuntimeInstallation); },
    }, offer, listing([entry(51)]), () => null, ids('1'.repeat(32)), statusOf);
    expect(bindCalls).toEqual([{ catalog_sha256: 'c'.repeat(64), bundle_id: 'example.studio', sequence: 51, runtime_platform: 'macos-arm64', plugin_id: offer.installed!.plugin_id }]);
    expect(installCalls).toEqual([{ pluginId: offer.installed!.plugin_id, request: { operation_id: '1'.repeat(32), runtime_digest: '9'.repeat(64), expected_source_revision: 7 } }]);
    expect(isStartRefusal(outcome)).toBe(false);
    expect(outcome).toMatchObject({ runtimeDigest: '9'.repeat(64), activationOperationId: null });
    clearJourney(offer.installed!.plugin_id);
    expect(readJourneys()).toEqual([]);
  });

  it('names a new installation itself, so a bind whose reply was lost is continued, not duplicated', async () => {
    const bindCalls: { plugin_id?: string }[] = [];
    const lost = await startCatalogInstall({
      bind: (request) => { bindCalls.push(request); return rejected(new TypeError('network')); },
      install: () => resolved({} as RuntimeInstallation),
    }, fresh, listing([entry(51)]), () => null, ids('a'.repeat(32), '2'.repeat(32)), statusOf);
    expect(lost).toEqual({ stage: 'bind', detail: null, unconfirmed: true });
    expect(readJourneys()).toMatchObject([{ pluginId: `managed.${'a'.repeat(32)}`, runtimeDigest: null }]);
    const again = await startCatalogInstall({
      bind: (request) => { bindCalls.push(request); return resolved(bound(request.plugin_id!)); },
      install: () => resolved({} as RuntimeInstallation),
    }, fresh, listing([entry(51)]), () => null, ids('b'.repeat(32), '3'.repeat(32)), statusOf);
    expect(bindCalls.map((call) => call.plugin_id)).toEqual([`managed.${'a'.repeat(32)}`, `managed.${'a'.repeat(32)}`]);
    expect(again).toMatchObject({ pluginId: `managed.${'a'.repeat(32)}`, runtimeDigest: '9'.repeat(64) });
    expect(readJourneys()).toHaveLength(1);
  });

  it('follows a release this browser is already installing instead of starting it again', async () => {
    const following = { ...bound(`managed.${'e'.repeat(32)}`) };
    const journey = { pluginId: following.plugin_id, title: 'Example Studio', bundleId: 'example.studio', sequence: 51, publisher: 'example', runtimeDigest: '9'.repeat(64), transferBytes: 1, installOperationId: '6'.repeat(32), activationOperationId: null, startedAt: 1 };
    saveJourney(journey);
    let binds = 0;
    const outcome = await startCatalogInstall({ bind: () => { binds += 1; return resolved(following); }, install: () => resolved({} as RuntimeInstallation) }, fresh, listing([entry(51)]), () => null, ids(), statusOf);
    expect(outcome).toEqual(journey);
    expect(binds).toBe(0);
    // Another release of the bundle is a separate install.
    expect(journeyInProgress('example.studio', 52)).toBeNull();
  });

  it('returns the host\'s sentence for a decided refusal, keeping only a new installation\'s identity', async () => {
    const detail = { status: 409, data: { detail: 'This host could not reach the plugin catalog.' } };
    const refusedStarters = { bind: () => rejected(detail), install: () => resolved({} as RuntimeInstallation) };
    const refused = await startCatalogInstall(refusedStarters, fresh, listing([entry(51)]), (error) => (error as typeof detail).data.detail, ids('c'.repeat(32)), statusOf);
    expect(refused).toEqual({ stage: 'bind', detail: 'This host could not reach the plugin catalog.', status: 409 });
    // The host may have registered the new installation before refusing, so the
    // next attempt must continue it.
    expect(readJourneys()).toMatchObject([{ pluginId: `managed.${'c'.repeat(32)}`, runtimeDigest: null }]);
    const binds: (string | undefined)[] = [];
    await startCatalogInstall({ bind: (request) => { binds.push(request.plugin_id); return rejected(detail); }, install: () => resolved({} as RuntimeInstallation) }, fresh, listing([entry(51)]), () => null, ids('d'.repeat(32)), statusOf);
    expect(binds).toEqual([`managed.${'c'.repeat(32)}`]);
    globalThis.localStorage.removeItem('skulk-plugin-install-journeys');
    // An existing installation needs no saved identity: the attempt is forgotten.
    await startCatalogInstall(refusedStarters, offer, listing([entry(51)]), () => null, ids(), statusOf);
    expect(readJourneys()).toEqual([]);
    // A lost download reply keeps the journey, so its operation is read back.
    const lost = await startCatalogInstall({ bind: () => resolved(bound(offer.installed!.plugin_id)), install: () => rejected(new TypeError('network')) }, offer, listing([entry(51)]), () => null, ids(), statusOf);
    expect(isStartRefusal(lost)).toBe(false);
    expect(readJourneys().map((item) => item.pluginId)).toContain(offer.installed!.plugin_id);
    clearJourney(offer.installed!.plugin_id);
    // A 4xx without a sentence is still a decided refusal, reported with its status.
    const invalid = await startCatalogInstall({ bind: () => resolved(bound(offer.installed!.plugin_id)), install: () => rejected({ status: 422, data: {} }) }, offer, listing([entry(51)]), () => null, ids(), statusOf);
    expect(invalid).toEqual({ stage: 'download', detail: null, status: 422 });
    expect(readJourneys()).toEqual([]);
  });
});

describe('journeys', () => {
  it('keeps journeys per installation and ignores malformed storage', () => {
    const storage = memoryStorage();
    const journey: InstallJourney = { pluginId: 'managed.a', title: 'A', bundleId: 'a', sequence: 1, publisher: 'p', runtimeDigest: 'd', transferBytes: 1, installOperationId: 'o', activationOperationId: null, startedAt: 1 };
    saveJourney(journey, storage);
    saveJourney({ ...journey, activationOperationId: 'x' }, storage);
    expect(readJourneys(storage)).toEqual([{ ...journey, activationOperationId: 'x' }]);
    clearJourney('managed.a', storage);
    expect(readJourneys(storage)).toEqual([]);
    storage.setItem('skulk-plugin-install-journeys', '{broken');
    expect(readJourneys(storage)).toEqual([]);
  });
});
