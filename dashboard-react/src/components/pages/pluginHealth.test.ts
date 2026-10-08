import { describe, expect, it } from 'vitest';
import type { ManagedRuntime, PluginNodes, RuntimeInstallation } from '../../store/endpoints/plugins';
import { derivePluginHealth, installNeedsRetry, recordedServiceFailure } from './pluginHealth';

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
    // So is a disabled one; it reads as disabled rather than status unavailable.
    expect(derivePluginHealth({ ...runtime, enabled: false, stale: true, service: null }, undefined, false)).toBe('disabled');
    expect(derivePluginHealth({ ...runtime, enabled: false }, nodes, false)).toBe('disabled');
  });
  it('reads a stopped install that waits for a retry as needing attention', () => {
    const stopped: RuntimeInstallation = {
      request: { operation_id: 'o'.repeat(32), runtime_digest: 'next-release', expected_source_revision: 1 },
      review: { runtime_digest: 'next-release', source_revision: 1, publisher: 'fixture', bundle_id: 'fixture.bundle', version: '2', sequence: 2, platform: 'macos-arm64', python_requires: '>=3.13', skulk_build_sha256: 'b', permissions: [], artifact_bytes: 1, expires_at: 1 },
      state: 'recovery_required', downloaded_bytes: 1, error_code: 'installation_failed',
    };
    const fresh: ManagedRuntime = { ...runtime, selected_digest: null, enabled: false, stale: true, service: null };
    expect(installNeedsRetry(fresh, stopped)).toBe(true);
    expect(derivePluginHealth(fresh, undefined, false, null, stopped)).toBe('attention');
    // A running installation whose update stopped also waits for the retry.
    expect(derivePluginHealth(runtime, nodes, false, null, stopped)).toBe('attention');
    // Not when the stopped release is the one selected, the install is not stopped, or nothing is known.
    expect(installNeedsRetry({ ...runtime, selected_digest: 'next-release' }, stopped)).toBe(false);
    expect(installNeedsRetry(fresh, { ...stopped, state: 'staged' })).toBe(false);
    expect(installNeedsRetry({ ...fresh, uninstalled: true }, stopped)).toBe(false);
    expect(installNeedsRetry(fresh, null)).toBe(false);
    expect(derivePluginHealth(fresh, undefined, false, null, null)).toBe('disabled');
    // An unreachable inventory stays unknown.
    expect(derivePluginHealth(fresh, undefined, true, null, stopped)).toBe('unknown');
  });
});
