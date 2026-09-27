import { describe, expect, it } from 'vitest';
import type { ManagedRuntime, PluginNodes } from '../../store/endpoints/plugins';
import { derivePluginHealth, recordedServiceFailure } from './pluginHealth';

const runtime: ManagedRuntime = { plugin_id: 'fixture', selected_digest: 'verified-release', selection_revision: 1, enabled: true, stale: false, error_code: null, operation_id: null, operation_state: null, service: { state: 'running', active_digest: 'verified-release', observed_at: 1 } };
const nodes: PluginNodes = { pluginId: 'fixture', available: true, nodes: [{ nodeId: 'fictional-node', bundleId: 'fixture.bundle', version: '1', status: 'ready', configurable: false }] };

describe('plugin health evidence', () => {
  it('requires matching runtime and ready node evidence for healthy', () => {
    expect(derivePluginHealth(runtime, nodes, false)).toBe('healthy');
    expect(derivePluginHealth(runtime, undefined, false)).toBe('unknown');
    expect(derivePluginHealth({ ...runtime, service: { ...runtime.service!, active_digest: 'older-release' } }, nodes, false)).toBe('unknown');
  });
  it('does not claim health from stale, unavailable or unknown evidence', () => {
    expect(derivePluginHealth({ ...runtime, stale: true }, nodes, false)).toBe('unknown');
    expect(derivePluginHealth(runtime, nodes, true)).toBe('unknown');
    expect(derivePluginHealth(runtime, { ...nodes, available: false }, false)).toBe('unknown');
    expect(derivePluginHealth(runtime, { ...nodes, nodes: [{ ...nodes.nodes[0], status: 'unrecognized' }] }, false)).toBe('unknown');
  });
  it('keeps a recorded process failure as attention after its observation ages', () => {
    const failed: ManagedRuntime = { ...runtime, stale: true, service: { state: 'failed', active_digest: null, observed_at: 1, error_code: 'verification_failed' } };
    expect(derivePluginHealth(failed, undefined, false)).toBe('attention');
    // Without a recorded class a stale failure is still unconfirmed evidence.
    expect(derivePluginHealth({ ...failed, service: { ...failed.service!, error_code: null } }, undefined, false)).toBe('unknown');
    // Lifecycle work and an operator's disable still take precedence.
    expect(derivePluginHealth(failed, undefined, false, 'applying')).toBe('updating');
    expect(derivePluginHealth({ ...failed, enabled: false }, undefined, false)).toBe('disabled');
    // An unreachable inventory never reads as a confirmed failure.
    expect(derivePluginHealth(failed, undefined, true)).toBe('unknown');
    expect(recordedServiceFailure(failed)).toBe('verification_failed');
    expect(recordedServiceFailure(runtime)).toBeNull();
  });
  it('gives operations and lifecycle precedence over a running process', () => {
    expect(derivePluginHealth(runtime, nodes, false, 'applying')).toBe('updating');
    expect(derivePluginHealth(runtime, nodes, false, 'recovery_required')).toBe('attention');
    expect(derivePluginHealth({ ...runtime, enabled: false, uninstalled: true, service: null }, undefined, false)).toBe('uninstalled');
    // An uninstalled installation is always stale, since nothing runs to be observed.
    expect(derivePluginHealth({ ...runtime, enabled: false, uninstalled: true, stale: true, service: null }, undefined, false)).toBe('uninstalled');
    expect(derivePluginHealth({ ...runtime, enabled: false }, nodes, false)).toBe('disabled');
  });
});
