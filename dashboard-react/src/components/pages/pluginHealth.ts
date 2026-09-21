import type { ManagedRuntime, PluginNodes, ManagedOperation } from '../../store/endpoints/plugins';

/** Evidence-derived summary; unknown observations must never imply health. */
export type PluginHealth = 'healthy' | 'attention' | 'uninstalled' | 'unknown' | 'updating' | 'disabled';
/** Card filtering is presentation-only and must not unmount operation owners. */
export type PluginFilter = 'all' | 'healthy' | 'attention' | 'uninstalled';

/** Combine current runtime, operation and node observations conservatively. */
export function derivePluginHealth(runtime: ManagedRuntime, nodes: PluginNodes | undefined, unavailable: boolean, operationState: ManagedOperation['state'] | null = runtime.operation_state): PluginHealth {
  // An uninstalled installation has no running service to observe, so it is
  // always stale; that does not make its state unknown.
  if (unavailable || (runtime.stale && !runtime.uninstalled)) return 'unknown';
  if (operationState === 'failed' || operationState === 'recovery_required' || runtime.error_code) return 'attention';
  if (operationState === 'accepted' || operationState === 'applying') return 'updating';
  if (runtime.uninstalled) return 'uninstalled';
  if (!runtime.enabled) return 'disabled';
  if (!runtime.service || !nodes?.available || !nodes.nodes.length) return 'unknown';
  if (runtime.service.state === 'failed' || nodes.nodes.some(node => node.status === 'failed')) return 'attention';
  if (runtime.service.state === 'running' && runtime.selected_digest !== null && runtime.service.active_digest === runtime.selected_digest && nodes.nodes.every(node => node.status === 'ready' || node.status === 'healthy')) return 'healthy';
  return 'unknown';
}
