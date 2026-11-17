import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { userEvent } from 'vitest/browser';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { darkTheme } from '../../theme/theme';
import { operatorSession } from '../../auth/operatorSession';
import { InferenceAccessPanel } from './InferenceAccessPanel';
import { inferenceGrantKey, selectInferenceGrant, type InferenceAccessPolicy, type InferenceGrant } from '../../utils/inferenceAccess';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
const translate = (_key: string, fallback: string) => fallback;
vi.mock('../../i18n/tolgee', () => ({ useSkulkTranslation: () => ({ t: translate }) }));
const owner = { identifier: 'direct-owner', kind: 'operator' as const, issuer_node_id: `core-profile:${'a'.repeat(32)}` };
const grant: InferenceGrant = { plugin_id: 'managed.test', runtime_digest: 'b'.repeat(64), node_id: 'installed-child', capability_id: 'example.chat', version: '1.0.0', descriptor_revision: 'c'.repeat(64), behavior: 'inference', processing_domain: 'external', io_mode: 'bidirectional' };
let root: Root | null = null;
let container: HTMLDivElement;
let requests: { path: string; method: string; body: Record<string, unknown> | null }[];
let policy: InferenceAccessPolicy;
let conflict = false;
let clients: { identifier: string; label: string; expires_unix_ms: null }[];
const clientId = 'd'.repeat(32);
const token = `ski_${clientId}_${'e'.repeat(64)}`;

async function mount() {
  policy = { protocol: 1, revision: '0'.repeat(64), enabled: false, callers: [] };
  clients = []; requests = []; conflict = false;
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = new URL(input instanceof Request ? input.url : String(input), location.origin).pathname;
    const method = init?.method ?? 'GET';
    const body = typeof init?.body === 'string' ? JSON.parse(init.body) as Record<string, unknown> : null;
    requests.push({ path, method, body });
    if (path.endsWith('/owner')) return Response.json(owner);
    if (path.endsWith('/policy')) {
      if (method === 'PUT') {
        if (conflict) return Response.json({ detail: 'stale' }, { status: 409 });
        policy = { ...policy, ...body, revision: 'f'.repeat(64) } as InferenceAccessPolicy;
      }
      return Response.json(policy);
    }
    if (path.endsWith('/clients') && method === 'POST') {
      const client = { identifier: clientId, label: String(body?.label), expires_unix_ms: null };
      clients.push(client); return Response.json({ client, token, revision: '1'.repeat(64) }, { status: 201 });
    }
    if (method === 'DELETE') { clients = []; return Response.json({ revision: '2'.repeat(64), clients }); }
    if (path.endsWith('/clients')) return Response.json({ revision: '0'.repeat(64), clients });
    if (path === '/v1/native-inference') return Response.json([{ plugin_id: grant.plugin_id, runtime_digest: grant.runtime_digest, target: grant, descriptor: { title: 'Example hosted chat', description: 'A signed provider', io_mode: grant.io_mode } }]);
    throw new Error(`Unexpected request ${path}`);
  }));
  container = document.createElement('div'); document.body.append(container); root = createRoot(container);
  await act(async () => { root?.render(<ThemeProvider theme={darkTheme}><InferenceAccessPanel /></ThemeProvider>); });
  await expect.poll(() => container.textContent).toContain('Example hosted chat');
}

afterEach(async () => { await act(async () => root?.unmount()); container?.remove(); root = null; operatorSession.useDirectAccess(); vi.unstubAllGlobals(); });

describe('explicit inference access', () => {
  it('loads authenticated setup without automatically granting or enabling anything', async () => {
    await mount();
    expect(container.querySelectorAll('input[type="checkbox"]:checked')).toHaveLength(0);
    expect(requests.every(request => request.method === 'GET')).toBe(true);
    expect(container.textContent).toContain('External processing');
    expect(Array.from(container.querySelectorAll('button')).find(button => button.textContent === 'Save inference access')?.disabled).toBe(true);
  });

  it('saves the actual owner issuer and exact selected release after explicit review', async () => {
    await mount();
    const checkboxes = container.querySelectorAll<HTMLInputElement>('input[type="checkbox"]');
    await userEvent.click(checkboxes[0]); await userEvent.click(checkboxes[1]);
    await userEvent.click(Array.from(container.querySelectorAll('button')).find(button => button.textContent === 'Save inference access')!);
    await expect.poll(() => container.textContent).toContain('Inference access saved.');
    const update = requests.find(request => request.method === 'PUT')?.body;
    expect(update).toEqual({ expected_revision: '0'.repeat(64), enabled: true, callers: [{ principal: owner, grants: [grant], maximum_seconds: null }] });
  });

  it('does not retry or replace the unsaved draft after a conflicting save', async () => {
    await mount(); conflict = true;
    await userEvent.click(container.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')[1]);
    await userEvent.click(Array.from(container.querySelectorAll('button')).find(button => button.textContent === 'Save inference access')!);
    await expect.poll(() => container.textContent).toContain('Reload and review');
    expect(requests.filter(request => request.method === 'PUT')).toHaveLength(1);
    expect(container.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')[1].checked).toBe(true);
  });

  it('preserves other callers and configured duration ceilings when editing the owner', async () => {
    await mount();
    clients = [{ identifier: clientId, label: 'Existing client', expires_unix_ms: null }];
    const other = { principal: { ...owner, identifier: `client:${clientId}`, kind: 'client' as const }, grants: [grant], maximum_seconds: 17 };
    policy = { ...policy, enabled: true, callers: [other, { principal: owner, grants: [], maximum_seconds: 23 }] };
    await userEvent.click(Array.from(container.querySelectorAll('button')).find(button => button.textContent === 'Reload access')!);
    await expect.poll(() => container.textContent).toContain('23');
    await userEvent.click(container.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')[1]);
    await userEvent.click(Array.from(container.querySelectorAll('button')).find(button => button.textContent === 'Save inference access')!);
    await expect.poll(() => container.textContent).toContain('Inference access saved.');
    expect(requests.find(request => request.method === 'PUT')?.body?.callers).toEqual([other, { principal: owner, grants: [grant], maximum_seconds: 23 }]);
  });

  it('creates an unprivileged one-time client key, clears disclosure, and revokes it', async () => {
    await mount();
    await userEvent.fill(container.querySelector<HTMLInputElement>('input[maxlength="128"]')!, 'Qualification client');
    await userEvent.click(Array.from(container.querySelectorAll('button')).find(button => button.textContent === 'Create client key')!);
    await expect.poll(() => container.querySelector<HTMLInputElement>('input[type="password"]')?.value).toBe(token);
    expect(requests.filter(request => request.method === 'PUT')).toHaveLength(0);
    expect(JSON.stringify(localStorage)).not.toContain(token);
    await userEvent.click(Array.from(container.querySelectorAll('button')).find(button => button.textContent === 'I saved the key')!);
    expect(container.querySelector('input[type="password"]')).toBeNull();
    const checkboxes = container.querySelectorAll<HTMLInputElement>('input[type="checkbox"]');
    await userEvent.click(checkboxes[0]); await userEvent.click(checkboxes[1]);
    await userEvent.click(Array.from(container.querySelectorAll('button')).find(button => button.textContent === 'Save inference access')!);
    await expect.poll(() => container.textContent).toContain('Inference access saved.');
    await userEvent.click(Array.from(container.querySelectorAll('button')).find(button => button.textContent === 'Revoke client key')!);
    await expect.poll(() => container.textContent).toContain('Client key revoked.');
    expect(requests.filter(request => request.method === 'DELETE')).toHaveLength(1);
    await userEvent.click(Array.from(container.querySelectorAll('button')).find(button => button.textContent === 'Remove retained permissions')!);
    await userEvent.click(Array.from(container.querySelectorAll('button')).find(button => button.textContent === 'Save inference access')!);
    await expect.poll(() => policy.callers).toEqual([]);
  });

  it('replaces an old release explicitly and keeps unrelated grants', () => {
    const old = { ...grant, runtime_digest: '1'.repeat(64) };
    const other = { ...grant, capability_id: 'example.catalog', behavior: 'read' as const };
    const chosen = selectInferenceGrant([old, other], grant, true);
    expect(chosen).toEqual([other, grant]);
    expect(inferenceGrantKey(old)).not.toBe(inferenceGrantKey(grant));
  });

  it('clears disclosed credentials and drafts when the operator session ends', async () => {
    await mount();
    await userEvent.fill(container.querySelector<HTMLInputElement>('input[maxlength="128"]')!, 'Session-bound client');
    await userEvent.click(Array.from(container.querySelectorAll('button')).find(button => button.textContent === 'Create client key')!);
    await expect.poll(() => container.querySelector<HTMLInputElement>('input[type="password"]')?.value).toBe(token);
    await userEvent.click(container.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')[1]);
    await act(async () => operatorSession.disconnect());
    expect(container.querySelector('input[type="password"]')).toBeNull();
    expect(container.querySelector('input[type="checkbox"]')).toBeNull();
    await userEvent.click(Array.from(container.querySelectorAll('button')).find(button => button.textContent === 'Use direct host access')!);
    await expect.poll(() => container.textContent).toContain('Example hosted chat');
    expect(container.querySelector('input[type="password"]')).toBeNull();
    expect(container.querySelectorAll('input[type="checkbox"]:checked')).toHaveLength(0);
    expect(container.querySelector<HTMLSelectElement>('select')?.value).toBe('direct-owner');
  });

  it('discards a late credential body even when the session returns to direct in one batch', async () => {
    await mount();
    let finishBody: () => void = () => { throw new Error('Body was not requested'); };
    const response = new Response(new ReadableStream<Uint8Array>({ start(controller) {
      finishBody = () => { controller.enqueue(new TextEncoder().encode(JSON.stringify({ client: { identifier: clientId, label: 'Late client', expires_unix_ms: null }, token, revision: '1'.repeat(64) }))); controller.close(); };
    } }), { headers: { 'Content-Type': 'application/json' }, status: 201 });
    await userEvent.fill(container.querySelector<HTMLInputElement>('input[maxlength="128"]')!, 'Late client');
    vi.mocked(fetch).mockResolvedValueOnce(response);
    await userEvent.click(Array.from(container.querySelectorAll('button')).find(button => button.textContent === 'Create client key')!);
    await expect.poll(() => response.bodyUsed).toBe(true);
    await act(async () => operatorSession.useDirectAccess());
    await expect.poll(() => container.textContent).toContain('Example hosted chat');
    await act(async () => { finishBody(); });
    expect(container.querySelector('input[type="password"]')).toBeNull();
    expect(container.querySelector<HTMLSelectElement>('select')?.value).toBe('direct-owner');
    expect(vi.mocked(fetch).mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(1);
  });
});
