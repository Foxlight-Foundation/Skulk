import { describe, expect, it, vi } from 'vitest';
import type { CapabilityNodeSummary } from '../../types/capabilityNodes';
import { buildCapabilityActions, runDescriptorAction } from './capabilityActions';

const t = (_key: string, fallback: string, params?: Record<string, string | number>) =>
  fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? ''));

function summary(overrides: Partial<CapabilityNodeSummary> = {}): CapabilityNodeSummary {
  return {
    pluginId: 'foxlight.video-studio',
    nodeId: 'studio',
    bundleId: 'foxlight.video-studio',
    version: '1.0.0',
    title: 'Video Studio',
    status: 'ready',
    ownerAvailable: true,
    surfaces: [
      { surfaceId: 'studio', title: 'Studio', kind: 'link', url: 'http://127.0.0.1:8188/', ready: true },
      { surfaceId: 'comfy', title: 'ComfyUI', kind: 'link', url: 'http://127.0.0.1:8189/', ready: false },
    ],
    actions: [
      { actionId: 'open-studio', title: 'Open Studio', kind: 'surface', surfaceId: 'studio' },
      { actionId: 'docs', title: 'Docs', kind: 'link', url: 'https://example.invalid/docs' },
      { actionId: 'plan', title: 'Plan', kind: 'descriptor', capabilityId: 'video.plan', payload: { mode: 't2va' } },
      { actionId: 'dangling', title: 'Dangling', kind: 'surface', surfaceId: 'missing' },
    ],
    operationsActive: 0,
    observedAt: new Date().toISOString(),
    ...overrides,
  };
}

describe('buildCapabilityActions', () => {
  it('lists surfaces, then manifest actions, then details, on the local host', () => {
    const items = buildCapabilityActions(summary(), { isLocalHost: true, hostName: 'kite6', t });
    expect(items.map((item) => item.id)).toEqual([
      'surface:studio',
      'surface:comfy',
      'action:docs',
      'action:plan',
      'details',
    ]);
    const comfy = items[1];
    expect(comfy?.kind === 'open-link' && comfy.ready).toBe(false);
    const plan = items[3];
    expect(plan?.kind === 'call' && plan.enabled && plan.payload).toEqual({ mode: 't2va' });
  });

  it('disables descriptor calls and adds the host hint when connected elsewhere', () => {
    const items = buildCapabilityActions(summary(), { isLocalHost: false, hostName: 'kite6', t });
    const plan = items.find((item) => item.id === 'action:plan');
    expect(plan?.kind === 'call' && plan.enabled).toBe(false);
    const last = items[items.length - 1];
    expect(last?.kind).toBe('manage-on-host');
    expect(last?.title).toBe('Manage on kite6');
  });
});

describe('runDescriptorAction', () => {
  it('resolves the exact descriptor before posting the call envelope', async () => {
    const calls: Array<{ url: string; body?: unknown }> = [];
    const fetchImpl = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      calls.push({ url, body: init?.body ? JSON.parse(String(init.body)) : undefined });
      if (url === '/v1/capabilities') {
        return new Response(
          JSON.stringify({
            capabilities: [{ id: 'video.plan', version: '1.0.0' }],
            revisions: { 'video.plan@1.0.0': 'rev-1' },
          }),
          { status: 200 },
        );
      }
      return new Response(JSON.stringify({ call_id: 'x', ok: true, result: { plan: 'ok' } }), {
        status: 200,
      });
    }) as unknown as typeof fetch;
    const result = await runDescriptorAction('node-a', 'video.plan', { mode: 't2va' }, fetchImpl);
    expect(result.ok).toBe(true);
    const envelope = calls[1]?.body as Record<string, unknown>;
    expect(envelope.capability_id).toBe('video.plan');
    expect(envelope.version).toBe('1.0.0');
    expect(envelope.descriptor_revision).toBe('rev-1');
    expect(envelope.caller_node).toBe('node-a');
    expect(envelope.target_node).toBe('node-a');
    expect(envelope.payload).toEqual({ mode: 't2va' });
  });

  it('refuses a capability the host does not serve', async () => {
    const fetchImpl = vi.fn(async () =>
      new Response(JSON.stringify({ capabilities: [], revisions: {} }), { status: 200 }),
    ) as unknown as typeof fetch;
    await expect(runDescriptorAction('node-a', 'video.plan', {}, fetchImpl)).rejects.toThrow(
      'not served',
    );
  });
});
