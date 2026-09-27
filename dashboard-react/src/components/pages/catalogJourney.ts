import type { CatalogEntry, CatalogInstallation, CatalogInstallRequest, CatalogListing, ManagedRuntime, RuntimeInstallation } from '../../store/endpoints/plugins';

/** What a catalog card offers this host for one bundle. */
export type CatalogOfferState = 'available' | 'installed' | 'update' | 'unfit';

/** One bundle in a catalog: the release to offer and the installation it would replace. */
export interface CatalogOffer {
  /** The newest listing built for this host, else the newest listing. */
  entry: CatalogEntry;
  /** A current (not uninstalled) installation of the same bundle, if any. */
  installed: ManagedRuntime | null;
  state: CatalogOfferState;
}

/**
 * Group a verified listing by bundle and decide what each card offers.
 *
 * A bundle is offered at its newest release that fits this host. An installed
 * bundle with a newer fitting release is an update: that is also how an
 * installation broken by a Skulk update gets its matching release back.
 */
export function catalogOffers(listing: CatalogListing, runtimes: ManagedRuntime[]): CatalogOffer[] {
  const bundles = new Map<string, CatalogEntry[]>();
  for (const entry of listing.entries) {
    bundles.set(entry.bundle_id, [...(bundles.get(entry.bundle_id) ?? []), entry]);
  }
  const offers: CatalogOffer[] = [];
  for (const [bundleId, entries] of bundles) {
    const newestFirst = [...entries].sort((left, right) => right.sequence - left.sequence);
    const fitting = newestFirst.find((entry) => entry.matches_host) ?? null;
    // Only an installation whose selected release names this bundle is matched,
    // so its sequence is always known here.
    const installed = installedRuntime(bundleId, runtimes);
    let state: CatalogOfferState;
    let entry = fitting ?? newestFirst[0];
    if (installed?.release) {
      const installedSequence = installed.release.sequence;
      if (fitting && fitting.sequence > installedSequence) state = 'update';
      else {
        state = 'installed';
        entry = newestFirst.find((candidate) => candidate.sequence === installedSequence) ?? entry;
      }
    } else {
      state = fitting ? 'available' : 'unfit';
    }
    offers.push({ entry, installed, state });
  }
  return offers.sort((left, right) => displayTitle(left.entry).localeCompare(displayTitle(right.entry)));
}

/** The installation of a bundle that a catalog install would upgrade. */
export function installedRuntime(bundleId: string, runtimes: ManagedRuntime[]): ManagedRuntime | null {
  const candidates = runtimes.filter((runtime) => runtime.release?.bundle_id === bundleId && !runtime.uninstalled);
  candidates.sort((left, right) => Number(right.enabled) - Number(left.enabled) || (right.release?.sequence ?? 0) - (left.release?.sequence ?? 0));
  return candidates[0] ?? null;
}

/** Title from the signed record, else the bundle id. */
export function displayTitle(entry: Pick<CatalogEntry, 'title' | 'bundle_id'>): string {
  return entry.title?.trim() || entry.bundle_id;
}

/** Two-letter mark for a card, from the title's words. */
export function monogram(title: string): string {
  const words = title.split(/[\s._-]+/).filter(Boolean);
  const letters = words.length > 1 ? words[0][0] + words[1][0] : title.slice(0, 2);
  return letters.toUpperCase();
}

const PLATFORM_LABELS: Record<string, string> = {
  'macos-arm64': 'macOS, Apple Silicon',
  'macos-x86_64': 'macOS, Intel',
  'linux-glibc-x86_64': 'Linux, x86-64',
  'linux-glibc-aarch64': 'Linux, ARM64',
};

const OS_LABELS: Record<string, string> = { darwin: 'macOS', linux: 'Linux', windows: 'Windows' };

/** Where a release runs, in words: its artifact family, else its operating systems. */
export function platformLabel(entry: Pick<CatalogEntry, 'runtime_platform' | 'platforms'>): string {
  if (entry.runtime_platform) return PLATFORM_LABELS[entry.runtime_platform] ?? entry.runtime_platform;
  const systems = entry.platforms.map((name) => OS_LABELS[name] ?? name);
  return systems.join(' or ');
}

/** Whether any of the release's actions can spend money. */
export function canSpendMoney(entry: Pick<CatalogEntry, 'steward_risks'>): boolean {
  return entry.steward_risks.includes('billable');
}

/** Size in megabytes with one decimal, the way the catalog is reviewed. */
export function formatMegabytes(bytes: number): string {
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
}

/** A capability id without its version, for display: `images.edit@1.0.0` becomes `images.edit`. */
export function capabilityName(descriptor: string): string {
  return descriptor.split('@', 1)[0];
}

/** The discovery facts an invitation carries: where the catalog is and whose key signs it. */
export interface CatalogInvitation {
  baseUrl: string;
  publisher: string;
  /** Ed25519 public key, 64 lowercase hex characters. */
  publicKey: string;
  /** Unix seconds after which this host stops trusting the key for discovery. */
  trustExpiresAt: number;
  /** Optional bearer for a catalog behind a credential; never displayed. */
  token?: string;
}

/** Prefix of an invitation code, so a pasted code is recognizable. */
export const INVITATION_PREFIX = 'skulk-catalog:';

function base64UrlEncode(text: string): string {
  const bytes = new TextEncoder().encode(text);
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '');
}

function base64UrlDecode(text: string): string | null {
  if (!/^[A-Za-z0-9_-]+$/.test(text)) return null;
  try {
    const padded = text.replaceAll('-', '+').replaceAll('_', '/') + '='.repeat((4 - (text.length % 4)) % 4);
    const binary = atob(padded);
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } catch {
    return null;
  }
}

/**
 * Encode an invitation as one pasteable code.
 *
 * The code carries the same facts the explicit form asks for. It grants
 * nothing by itself: every release is still verified against the publisher
 * key before anything installs.
 */
export function encodeInvitation(invitation: CatalogInvitation): string {
  const body = { v: 1, url: invitation.baseUrl, publisher: invitation.publisher, key: invitation.publicKey, expires: invitation.trustExpiresAt, ...(invitation.token ? { token: invitation.token } : {}) };
  return INVITATION_PREFIX + base64UrlEncode(JSON.stringify(body));
}

/** Read an invitation code, or null when it is not one this dashboard understands. */
export function decodeInvitation(code: string, nowSeconds = Date.now() / 1000): CatalogInvitation | null {
  const trimmed = code.trim();
  if (!trimmed.startsWith(INVITATION_PREFIX) || trimmed.length > 4096) return null;
  const text = base64UrlDecode(trimmed.slice(INVITATION_PREFIX.length));
  if (text === null) return null;
  let body: unknown;
  try { body = JSON.parse(text); } catch { return null; }
  if (!body || typeof body !== 'object') return null;
  const { v, url, publisher, key, expires, token } = body as Record<string, unknown>;
  if (v !== 1 || typeof url !== 'string' || typeof publisher !== 'string' || typeof key !== 'string' || typeof expires !== 'number') return null;
  if (!isCatalogAddress(url) || !/^[a-z0-9][a-z0-9._-]{0,63}$/.test(publisher) || !/^[0-9a-f]{64}$/.test(key)) return null;
  if (!Number.isInteger(expires) || expires <= nowSeconds) return null;
  if (token !== undefined && (typeof token !== 'string' || token.length === 0 || token.length > 4096 || /\s/.test(token))) return null;
  return { baseUrl: url, publisher, publicKey: key, trustExpiresAt: expires, ...(token ? { token } : {}) };
}

/** An explicit HTTPS directory, the only catalog address the host accepts. */
export function isCatalogAddress(value: string): boolean {
  try {
    const parsed = new URL(value);
    return parsed.protocol === 'https:' && parsed.pathname.endsWith('/') && !parsed.search && !parsed.hash && !parsed.username && !parsed.password;
  } catch {
    return false;
  }
}

/**
 * An install in progress, kept in the browser so the page can pick it up
 * again after it was left. It records only operation identities the host
 * already accepted; resuming reads their status and never repeats them.
 */
export interface InstallJourney {
  pluginId: string;
  title: string;
  bundleId: string;
  sequence: number;
  publisher: string;
  runtimeDigest: string;
  /** Download size the listing declared, for progress. */
  transferBytes: number;
  installOperationId: string;
  activationOperationId: string | null;
  startedAt: number;
}

const JOURNEY_KEY = 'skulk-plugin-install-journeys';
const JOURNEY_LIMIT = 8;

function isJourney(value: unknown): value is InstallJourney {
  if (!value || typeof value !== 'object') return false;
  const journey = value as Record<string, unknown>;
  return typeof journey.pluginId === 'string' && typeof journey.title === 'string' && typeof journey.bundleId === 'string'
    && typeof journey.sequence === 'number' && typeof journey.publisher === 'string' && typeof journey.runtimeDigest === 'string'
    && typeof journey.transferBytes === 'number' && typeof journey.installOperationId === 'string'
    && (journey.activationOperationId === null || typeof journey.activationOperationId === 'string')
    && typeof journey.startedAt === 'number';
}

/** Installs this browser started and has not finished watching. Storage may be unavailable. */
export function readJourneys(storage: Pick<Storage, 'getItem'> | null = safeStorage()): InstallJourney[] {
  if (!storage) return [];
  try {
    const parsed: unknown = JSON.parse(storage.getItem(JOURNEY_KEY) ?? '[]');
    return Array.isArray(parsed) ? parsed.filter(isJourney).slice(0, JOURNEY_LIMIT) : [];
  } catch {
    return [];
  }
}

/** Record or replace the journey for one installation. */
export function saveJourney(journey: InstallJourney, storage: Pick<Storage, 'getItem' | 'setItem'> | null = safeStorage()): void {
  if (!storage) return;
  const others = readJourneys(storage).filter((item) => item.pluginId !== journey.pluginId);
  try { storage.setItem(JOURNEY_KEY, JSON.stringify([journey, ...others].slice(0, JOURNEY_LIMIT))); } catch { /* best effort */ }
}

/** Forget the journey for one installation once it is finished or abandoned. */
export function clearJourney(pluginId: string, storage: Pick<Storage, 'getItem' | 'setItem'> | null = safeStorage()): void {
  if (!storage) return;
  try { storage.setItem(JOURNEY_KEY, JSON.stringify(readJourneys(storage).filter((item) => item.pluginId !== pluginId))); } catch { /* best effort */ }
}

function safeStorage(): Storage | null {
  try { return globalThis.localStorage ?? null; } catch { return null; }
}

/** An RTK request result: resolves with the response or rejects with the error. */
export interface Unwrappable<T> { unwrap(): Promise<T> }

/** The two host requests that begin an install; injected so the flow is testable. */
export interface InstallStarters {
  bind: (request: CatalogInstallRequest) => Unwrappable<CatalogInstallation>;
  install: (args: { pluginId: string; request: RuntimeInstallation['request'] }) => Unwrappable<RuntimeInstallation>;
}

/** Why starting an install stopped: the server's own sentence and status, when it gave them. */
export interface InstallStartRefusal { stage: 'bind' | 'download'; detail: string | null; status?: number }

/**
 * Bind the installation to the reviewed listing and ask the host to download
 * it, from the consent click itself so it runs exactly once.
 *
 * The journey is saved before the download request is sent. If that
 * response is lost the page reads the operation back instead of sending it
 * again; a refusal with a reason clears the journey and returns the reason.
 */
export async function startCatalogInstall(
  starters: InstallStarters,
  offer: CatalogOffer,
  listing: CatalogListing,
  refusalDetail: (error: unknown) => string | null,
  newOperationId: () => string,
  refusedStatus: (error: unknown) => number | null = () => null,
): Promise<InstallJourney | InstallStartRefusal> {
  let bound: CatalogInstallation;
  try {
    bound = await starters.bind({
      catalog_sha256: listing.catalog_sha256, bundle_id: offer.entry.bundle_id, sequence: offer.entry.sequence,
      ...(offer.entry.runtime_platform ? { runtime_platform: offer.entry.runtime_platform } : {}),
      ...(offer.installed ? { plugin_id: offer.installed.plugin_id } : {}),
    }).unwrap();
  } catch (error) {
    const status = refusedStatus(error);
    return { stage: 'bind', detail: refusalDetail(error), ...(status !== null ? { status } : {}) };
  }
  const journey: InstallJourney = {
    pluginId: bound.plugin_id, title: displayTitle(offer.entry), bundleId: offer.entry.bundle_id, sequence: offer.entry.sequence,
    publisher: offer.entry.publisher, runtimeDigest: bound.review.runtime_digest,
    transferBytes: bound.review.artifact_bytes || offer.entry.transfer_bytes,
    installOperationId: newOperationId(), activationOperationId: null, startedAt: Date.now(),
  };
  saveJourney(journey);
  try {
    await starters.install({ pluginId: journey.pluginId, request: { operation_id: journey.installOperationId, runtime_digest: journey.runtimeDigest, expected_source_revision: bound.source.revision } }).unwrap();
  } catch (error) {
    // Only a 4xx is a decided refusal; a 5xx, timeout or lost reply may have
    // landed, so the journey stays and its operation is read back.
    const status = refusedStatus(error);
    if (status !== null) { clearJourney(journey.pluginId); return { stage: 'download', detail: refusalDetail(error), status }; }
  }
  return journey;
}

/** Whether a start result is a refusal rather than a journey. */
export function isStartRefusal(value: InstallJourney | InstallStartRefusal): value is InstallStartRefusal {
  return 'stage' in value;
}
