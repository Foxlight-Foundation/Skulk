import type { OperatorDevice } from '../../store/endpoints/devices';

/**
 * Whether a not-yet-revoked device can no longer refresh its credentials.
 *
 * Such a device must pair again and no longer occupies one of the cluster's
 * device slots, matching how the server counts pairing capacity.
 *
 * @param device Paired-device projection from the node.
 * @param now Current time in milliseconds, injectable for tests.
 */
export function isRefreshExpired(device: OperatorDevice, now: number = Date.now()): boolean {
  if (device.state !== 'active') return false;
  return device.refreshExpiresAt === null || Date.parse(device.refreshExpiresAt) <= now;
}
