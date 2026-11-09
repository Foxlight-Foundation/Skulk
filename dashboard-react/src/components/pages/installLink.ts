/**
 * Install links from the capability storefront (apps.foxlight.ai).
 *
 * The storefront's Install button opens `skulk://install?bundle=<id>&version=<semver>`;
 * Skulk Desktop turns that into this page's address,
 * `/plugins?install=<id>&version=<semver>`, and an owner on a headless node
 * can paste the `skulk://` link into Browse instead. Either way the link only
 * names a listing: Browse looks the bundle up in this host's own verified
 * catalog and shows the ordinary review, so nothing in the link is trusted
 * beyond the two identifiers, and both must match the shapes the capability
 * SDK signs. `skulk://pair` invitations are never read here.
 */
import type { CatalogOffer } from './catalogJourney';

/** A listing an install link asks for. */
export interface InstallLinkRequest {
  bundleId: string;
  /** The version the storefront showed, when the link names one. */
  version: string | null;
}

const BUNDLE_ID = /^[A-Za-z0-9._:@-]{1,128}$/;
const VERSION = /^(0|[1-9]\d{0,8})\.(0|[1-9]\d{0,8})\.(0|[1-9]\d{0,8})$/;

function request(bundleId: string | null, version: string | null): InstallLinkRequest | null {
  if (bundleId === null || !BUNDLE_ID.test(bundleId)) return null;
  if (version !== null && !VERSION.test(version)) return null;
  return { bundleId, version };
}

/**
 * Read a pasted `skulk://install?bundle=…&version=…` link; null unless it is
 * exactly that, with nothing else in it.
 */
export function parseInstallLink(text: string): InstallLinkRequest | null {
  const trimmed = text.trim();
  if (!/^skulk:\/\/install\/?\?/i.test(trimmed) || trimmed.includes('#')) return null;
  let params: URLSearchParams;
  try {
    params = new URLSearchParams(trimmed.slice(trimmed.indexOf('?') + 1));
  } catch {
    return null;
  }
  const names = [...params.keys()];
  if (names.length !== 2 || new Set(names).size !== 2 || !names.includes('bundle') || !names.includes('version')) return null;
  return request(params.get('bundle'), params.get('version'));
}

/** Read `?install=<id>&version=<semver>` from this page's address. */
export function installRequestFromSearch(search: string): InstallLinkRequest | null {
  try {
    const params = new URLSearchParams(search);
    if (params.getAll('install').length !== 1 || params.getAll('version').length > 1) return null;
    return request(params.get('install'), params.get('version'));
  } catch {
    return null;
  }
}

/**
 * What Browse does with an install link once this host's catalog is read:
 * open the ordinary review for a release this host can install or update to,
 * or say why it will not.
 */
export type InstallLinkOutcome =
  | { kind: 'review'; offer: CatalogOffer; requestedVersion: string | null; versionDiffers: boolean }
  | { kind: 'installed' | 'unfinished' | 'unfit'; offer: CatalogOffer }
  | { kind: 'missing'; bundleId: string };

/** Resolve an install link against this host's own catalog offers. */
export function resolveInstallLink(request: InstallLinkRequest, offers: CatalogOffer[]): InstallLinkOutcome {
  const offer = offers.find((item) => item.entry.bundle_id === request.bundleId);
  if (!offer) return { kind: 'missing', bundleId: request.bundleId };
  switch (offer.state) {
    case 'available':
    case 'update':
      return {
        kind: 'review', offer, requestedVersion: request.version,
        versionDiffers: request.version !== null && request.version !== offer.entry.bundle_version,
      };
    case 'installed':
      return { kind: 'installed', offer };
    case 'retry':
      return { kind: 'unfinished', offer };
    case 'unfit':
      return { kind: 'unfit', offer };
  }
}
