import type {
  CatalogEntry, CatalogInstallation, CatalogInstallRequest, CatalogListing, InstallOperations, ManagedOperationAddress, ManagedRuntime,
  RuntimeInstallation, RuntimeRelease, RuntimeSourceStatus,
} from '../../store/endpoints/plugins';
import { installNeedsRetry } from './pluginHealth';

/** What a catalog card offers this host for one bundle. */
export type CatalogOfferState = 'available' | 'installed' | 'update' | 'unfit' | 'retry';

/**
 * An installation of a bundle whose install has not finished, with the signed
 * release it installs: one that stopped and waits for a retry, or a first
 * install still under way or staged.
 */
export interface UnfinishedInstall {
  pluginId: string;
  review: RuntimeRelease;
  /** Whether a release was already selected, so finishing it updates that release. */
  updating: boolean;
  /** Whether the host reports it stopped (`recovery_required`); otherwise it is under way or staged. */
  stopped: boolean;
}

/** One bundle in a catalog: the release to offer and the installation it would replace. */
export interface CatalogOffer {
  /** The newest listing built for this host, else the newest listing; for a retry, the release it retries when still listed. */
  entry: CatalogEntry;
  /** A current (not uninstalled) installation of the same bundle, if any. */
  installed: ManagedRuntime | null;
  state: CatalogOfferState;
  /** The installation of this bundle whose install has not finished, when the offer is to finish it. */
  retry?: UnfinishedInstall | null;
}

/**
 * The installations whose install has not finished: any whose install
 * stopped and waits for the owner's retry, and any first install (no release
 * selected yet) still under way or staged. A first install names its bundle
 * only through its install, so Browse would otherwise offer that bundle again
 * and register a second installation beside it.
 */
export function unfinishedInstalls(runtimes: ManagedRuntime[], operations: InstallOperations | undefined): UnfinishedInstall[] {
  const found: UnfinishedInstall[] = [];
  for (const runtime of runtimes) {
    const operation = operations?.[runtime.plugin_id];
    if (!operation) continue;
    const stopped = installNeedsRetry(runtime, operation);
    if (stopped || (!runtime.uninstalled && runtime.selected_digest === null)) {
      found.push({ pluginId: runtime.plugin_id, review: operation.review, updating: runtime.selected_digest !== null, stopped });
    }
  }
  return found;
}

/**
 * Group a verified listing by bundle and decide what each card offers.
 *
 * A bundle is offered at its newest release that fits this host. An installed
 * bundle with a newer fitting release is an update: that is also how an
 * installation broken by a Skulk update gets its matching release back.
 *
 * An installation of the bundle whose install stopped is offered for retry
 * instead of another install: the host refuses to bind it to a release until
 * it is retried, and a new installation would leave it behind. The installed
 * one's own stopped install comes first; a stopped first install counts while
 * nothing else of the bundle is installed.
 */
export function catalogOffers(listing: CatalogListing, runtimes: ManagedRuntime[], unfinished: UnfinishedInstall[] = []): CatalogOffer[] {
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
    const retry = installed
      ? unfinished.find((item) => item.stopped && item.pluginId === installed.plugin_id) ?? null
      : unfinished.find((item) => item.review.bundle_id === bundleId) ?? null;
    if (retry) {
      state = 'retry';
      // The card names the release the host installs: that sequence's listing
      // for this host, since several platforms can share a sequence.
      const sequence = retry.review.sequence;
      entry = newestFirst.find((candidate) => candidate.sequence === sequence && candidate.matches_host)
        ?? newestFirst.find((candidate) => candidate.sequence === sequence && candidate.runtime_platform === retry.review.platform)
        ?? entry;
    }
    offers.push({ entry, installed, state, retry });
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

/**
 * The revision for discovery trust the connect panel creates.
 *
 * The host keeps each catalog address's trust history and refuses a trust
 * older than the one it accepted there, or a different one at the same
 * revision. The current revision only describes the catalog in use now, so a
 * returning catalog could sit at or above it. The panel's opening time, in
 * seconds, is above any revision an earlier panel created and above the
 * current one, so re-adding a catalog with a new code is never refused as a
 * rollback.
 */
export function trustRevision(current: number | null, openedAtSeconds: number): number {
  return Math.max((current ?? 0) + 1, openedAtSeconds);
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
  /** Digest of the bound release; null until the host confirms the binding. */
  runtimeDigest: string | null;
  /** Download size the listing declared, for progress. */
  transferBytes: number;
  installOperationId: string;
  activationOperationId: string | null;
  startedAt: number;
  /**
   * Set when the host reported the install stopped (`recovery_required`).
   * Nothing is followed then, but the record stays: it is this browser's
   * consent to the release's permissions, which a retry of exactly that
   * release carries to activation.
   */
  interrupted?: boolean;
}

const JOURNEY_KEY = 'skulk-plugin-install-journeys';
const JOURNEY_LIMIT = 8;

function isJourney(value: unknown): value is InstallJourney {
  if (!value || typeof value !== 'object') return false;
  const journey = value as Record<string, unknown>;
  return typeof journey.pluginId === 'string' && typeof journey.title === 'string' && typeof journey.bundleId === 'string'
    && typeof journey.sequence === 'number' && typeof journey.publisher === 'string'
    && (journey.runtimeDigest === null || typeof journey.runtimeDigest === 'string')
    && typeof journey.transferBytes === 'number' && typeof journey.installOperationId === 'string'
    && (journey.activationOperationId === null || typeof journey.activationOperationId === 'string')
    && typeof journey.startedAt === 'number' && (journey.interrupted === undefined || typeof journey.interrupted === 'boolean');
}

/** A journey whose binding the host confirmed, so there is an operation to follow. */
export type BoundJourney = InstallJourney & { runtimeDigest: string };

/** Whether the host confirmed a journey's binding. */
export function isBound(journey: InstallJourney): journey is BoundJourney {
  return journey.runtimeDigest !== null;
}

/** Whether this browser follows a journey: its binding is confirmed and the host has not reported it stopped. */
export function isFollowed(journey: InstallJourney): journey is BoundJourney {
  return isBound(journey) && !journey.interrupted;
}

/** The confirmed journey installing this release of a bundle, if this browser still follows one. */
export function journeyInProgress(bundleId: string, sequence: number, journeys: InstallJourney[] = readJourneys()): BoundJourney | null {
  return journeys.find((item): item is BoundJourney => isFollowed(item) && item.bundleId === bundleId && item.sequence === sequence) ?? null;
}

/**
 * The journey an offer's card follows: one installing its release or, while
 * nothing of the bundle is installed, any install of the bundle this browser
 * follows. A first install of an older release, such as a retry, is still the
 * bundle's only installation; offering a newer release beside it would
 * register a second one. With an installation, only an install on it counts.
 */
export function offerProgress(offer: Pick<CatalogOffer, 'entry' | 'installed'>, journeys: InstallJourney[] = readJourneys()): BoundJourney | null {
  return journeyInProgress(offer.entry.bundle_id, offer.entry.sequence, journeys)
    ?? journeys.find((item): item is BoundJourney => isFollowed(item) && item.bundleId === offer.entry.bundle_id
      && (offer.installed === null || item.pluginId === offer.installed.plugin_id)) ?? null;
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

/**
 * Stop following one installation's journey because its install stopped,
 * keeping the record of the consent it carries for a retry.
 */
export function markInterrupted(pluginId: string, storage: Pick<Storage, 'getItem' | 'setItem'> | null = safeStorage()): void {
  const journey = readJourneys(storage).find((item) => item.pluginId === pluginId);
  if (journey) saveJourney({ ...journey, interrupted: true }, storage);
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

/**
 * Why starting an install stopped: the server's own sentence and status, when
 * it gave them. `unconfirmed` means no decided answer arrived for the binding;
 * its installation identity stays saved, so installing again continues it.
 */
export interface InstallStartRefusal {
  stage: 'bind' | 'download' | 'retry';
  detail: string | null;
  status?: number;
  unconfirmed?: boolean;
  /**
   * Why a retry stopped before anything was sent: the host reports no install
   * for the installation, its release source or credential is not ready, or
   * the host could not be read.
   */
  code?: 'nothing-to-retry' | 'source-unavailable' | 'unreadable';
}

/**
 * Bind the installation to the reviewed listing and ask the host to download
 * it, from the consent click itself so it runs exactly once.
 *
 * The journey is saved before each request. A new installation's identity is
 * chosen here rather than by the host, so a binding whose reply was lost is
 * continued, not duplicated, by the next attempt for the same bundle, as is
 * one the host registered before refusing the release. If the download reply
 * is lost the page reads the operation back instead of sending it again. A
 * decided refusal (a 4xx) returns the host's reason.
 */
export async function startCatalogInstall(
  starters: InstallStarters,
  offer: CatalogOffer,
  listing: CatalogListing,
  refusalDetail: (error: unknown) => string | null,
  newOperationId: () => string,
  refusedStatus: (error: unknown) => number | null = () => null,
): Promise<BoundJourney | InstallStartRefusal> {
  // A release this browser is already installing, in this tab or another, is
  // followed rather than started again.
  const inProgress = offerProgress(offer);
  if (inProgress) return inProgress;
  const saved = readJourneys();
  const unconfirmed = saved.find((item) => item.bundleId === offer.entry.bundle_id && !isBound(item));
  // A stopped install of the bundle keeps its installation: the host refuses
  // another release there until it is retried and says so, which beats
  // registering a new installation beside it.
  const stopped = saved.find((item) => item.bundleId === offer.entry.bundle_id && item.interrupted);
  const pluginId = offer.installed?.plugin_id ?? offer.retry?.pluginId ?? unconfirmed?.pluginId ?? stopped?.pluginId ?? `managed.${newOperationId()}`;
  const pending: InstallJourney = {
    pluginId, title: displayTitle(offer.entry), bundleId: offer.entry.bundle_id, sequence: offer.entry.sequence,
    publisher: offer.entry.publisher, runtimeDigest: null, transferBytes: offer.entry.transfer_bytes,
    installOperationId: newOperationId(), activationOperationId: null, startedAt: Date.now(),
  };
  saveJourney(pending);
  let bound: CatalogInstallation;
  try {
    bound = await starters.bind({
      catalog_sha256: listing.catalog_sha256, bundle_id: offer.entry.bundle_id, sequence: offer.entry.sequence,
      ...(offer.entry.runtime_platform ? { runtime_platform: offer.entry.runtime_platform } : {}),
      plugin_id: pluginId,
    }).unwrap();
  } catch (error) {
    // Only a 4xx is decided. Anything else may have registered and bound the
    // installation, so its identity stays saved for the next attempt. So does
    // a new installation's after a refusal: the host keeps an installation it
    // registered before refusing the release, and the next attempt continues
    // it rather than registering another.
    const status = refusedStatus(error);
    if (status === null) return { stage: 'bind', detail: refusalDetail(error), unconfirmed: true };
    if (offer.installed) clearJourney(pluginId);
    return { stage: 'bind', detail: refusalDetail(error), status };
  }
  const journey: BoundJourney = {
    ...pending, pluginId: bound.plugin_id, runtimeDigest: bound.review.runtime_digest,
    transferBytes: bound.review.artifact_bytes || offer.entry.transfer_bytes,
  };
  if (journey.pluginId !== pluginId) clearJourney(pluginId);
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

/** Whether a start or retry result is a refusal rather than a journey. */
export function isStartRefusal(value: BoundJourney | InstallStartRefusal | RetryConsentNeeded): value is InstallStartRefusal {
  return 'stage' in value;
}

/** A stopped install to retry, as the page offering the retry knows it. */
export interface RetryRequest {
  pluginId: string;
  /** Catalog title when known, else the bundle id; a saved journey's title wins. */
  title: string;
  publisher: string;
  sequence: number;
  transferBytes: number;
  updating: boolean;
}

/** The host requests a retry makes; injected so the flow is testable. */
export interface RetryStarters {
  readInstall: (pluginId: string) => Unwrappable<{ operation: RuntimeInstallation | null }>;
  readSource: (pluginId: string) => Unwrappable<RuntimeSourceStatus>;
  recover: (args: ManagedOperationAddress & { expectedSourceRevision: number }) => Unwrappable<RuntimeInstallation>;
}

/** A retry this browser holds no consent for: the signed release to review before anything is sent. */
export interface RetryConsentNeeded {
  pluginId: string;
  review: RuntimeRelease;
  /** Whether the install stopped; otherwise it is under way or staged and is only followed. */
  stopped: boolean;
}

/** Whether a retry result asks for the release to be reviewed first. */
export function isConsentNeeded(value: BoundJourney | InstallStartRefusal | RetryConsentNeeded): value is RetryConsentNeeded {
  return 'review' in value;
}

/**
 * Retry an installation's stopped install and return the journey that
 * finishes it, the way a fresh catalog install finishes: staged, then
 * activated, then running.
 *
 * Activation accepts the release's permissions, so a retry goes ahead only
 * under consent to exactly the release the host retries: the journey this
 * browser saved when the owner accepted it, or `reviewedDigest` when the owner
 * has just reviewed it again. Without either nothing is sent, and the signed
 * release comes back to be reviewed. Consent once given is saved before any
 * request, so a refused or failed retry can be tried again without asking.
 *
 * Only a stopped install is sent to the host's recover route, under the
 * release source's current revision; the host retries the original operation
 * and never changes its release. An install already running again or staged
 * is followed instead. A decided refusal (a 4xx) returns the host's reason; a
 * lost reply keeps the journey, so its operation is read back and never sent
 * twice.
 */
export async function startInstallRetry(
  starters: RetryStarters,
  request: Pick<RetryRequest, 'pluginId' | 'title'>,
  reviewedDigest: string | null,
  refusalDetail: (error: unknown) => string | null,
  refusedStatus: (error: unknown) => number | null = () => null,
): Promise<BoundJourney | InstallStartRefusal | RetryConsentNeeded> {
  const { pluginId } = request;
  const unreadable = (error: unknown): InstallStartRefusal => {
    const status = refusedStatus(error);
    return status === null ? { stage: 'retry', detail: null, code: 'unreadable' } : { stage: 'retry', detail: refusalDetail(error), status };
  };
  let operation: RuntimeInstallation | null;
  try { operation = (await starters.readInstall(pluginId).unwrap()).operation; } catch (error) { return unreadable(error); }
  if (!operation) return { stage: 'retry', detail: null, code: 'nothing-to-retry' };
  const review = operation.review;
  const consented = readJourneys().find((item) => item.pluginId === pluginId && item.runtimeDigest === review.runtime_digest) ?? null;
  if (!consented && reviewedDigest !== review.runtime_digest) return { pluginId, review, stopped: operation.state === 'recovery_required' };
  const journey: BoundJourney = {
    pluginId, title: consented?.title ?? request.title, bundleId: review.bundle_id, sequence: review.sequence, publisher: review.publisher,
    runtimeDigest: review.runtime_digest, transferBytes: review.artifact_bytes || (consented?.transferBytes ?? 0),
    installOperationId: operation.request.operation_id,
    // An activation already sent for this very install is read back, never sent again.
    activationOperationId: operation.state !== 'recovery_required' && consented?.installOperationId === operation.request.operation_id ? consented.activationOperationId : null,
    // A retry follows the same operation again, so its start time must tell
    // it from the attempt that stopped.
    startedAt: Math.max(Date.now(), (consented?.startedAt ?? 0) + 1),
  };
  if (operation.state !== 'recovery_required') { saveJourney(journey); return journey; }
  saveJourney({ ...journey, interrupted: true });
  let source: RuntimeSourceStatus;
  try { source = await starters.readSource(pluginId).unwrap(); } catch (error) { return unreadable(error); }
  if (!source.configured || !source.credential_ready) return { stage: 'retry', detail: null, code: 'source-unavailable' };
  saveJourney(journey);
  try {
    await starters.recover({ pluginId, operationId: operation.request.operation_id, expectedSourceRevision: source.revision }).unwrap();
  } catch (error) {
    // Only a 4xx is decided; anything else may have landed, so the journey
    // stays followed and the operation is read back.
    const status = refusedStatus(error);
    if (status !== null) { saveJourney({ ...journey, interrupted: true }); return { stage: 'retry', detail: refusalDetail(error), status }; }
  }
  return journey;
}
