import type { SkulkTranslate } from '../i18n/tolgee';
import type { PlacementEngineInstall } from '../types/models';

/**
 * Word an on-demand engine's approximate download in whole gigabytes, rounded
 * the way the server words its own notice ("about 7 GB").
 *
 * @param bytes - Approximate download in bytes.
 * @returns The size, such as `7 GB`, or null when the size is unknown.
 */
export function engineInstallSize(bytes: number): string | null {
  if (!Number.isFinite(bytes) || bytes <= 0) return null;
  return `${Math.max(1, Math.round(bytes / 1024 ** 3))} GB`;
}

/**
 * The sentence an operator reads before a placement installs an on-demand
 * engine, shared by the placement preview and the quick-launch toast.
 *
 * @param t - Translation function.
 * @param install - The engine install the placement triggers.
 * @returns The dashboard's wording for the video engine, otherwise the
 *   server-written notice.
 */
export function engineInstallNotice(t: SkulkTranslate, install: PlacementEngineInstall): string {
  const size = engineInstallSize(install.approximate_download_bytes);
  if (install.engine === 'comfy' && size) {
    return t(
      'modelCard.videoEngineInstallNotice',
      'The video engine (about {size}) will be installed with this model, so placement will take longer.',
      { size },
    );
  }
  return install.detail;
}

/**
 * Read the engine install from an accepted `POST /place_instance` response.
 *
 * @param body - The parsed response body.
 * @returns The engine install, or null when the placement installs none or the
 *   server predates the field.
 */
export function engineInstallFromPlacement(body: unknown): PlacementEngineInstall | null {
  if (typeof body !== 'object' || body === null) return null;
  const install = (body as Record<string, unknown>).engine_install;
  if (typeof install !== 'object' || install === null) return null;
  const { engine, node_ids: nodeIds, approximate_download_bytes: bytes, detail } =
    install as Record<string, unknown>;
  if (typeof engine !== 'string' || typeof detail !== 'string' || typeof bytes !== 'number') {
    return null;
  }
  return {
    engine,
    node_ids: Array.isArray(nodeIds) ? nodeIds.filter((id): id is string => typeof id === 'string') : [],
    approximate_download_bytes: bytes,
    detail,
  };
}
