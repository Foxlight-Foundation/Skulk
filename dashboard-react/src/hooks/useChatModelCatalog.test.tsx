import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it, vi } from 'vitest';
import { operatorSession } from '../auth/operatorSession';
import { useChatModelCatalog } from './useChatModelCatalog';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });

it('authenticates discovery and withdraws cached owner choices on identity change', async () => {
  const host = document.createElement('div'); document.body.append(host);
  const root = createRoot(host);
  const fetch = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
    expect(new Headers(init?.headers).get('X-Skulk-Dashboard')).toBe('pairing-v1');
    return new Response(JSON.stringify({ data: [{ id: 'native/provider/chat@0.2.0/code', name: 'Native Code' }] }));
  });
  vi.stubGlobal('fetch', fetch);
  function Catalog() {
    const catalog = useChatModelCatalog();
    return <p>{Object.keys(catalog.models).join(',') || 'No authorized models'}</p>;
  }
  try {
    await act(async () => { operatorSession.useDirectAccess(); root.render(<Catalog />); });
    await vi.waitFor(() => expect(host.textContent).toContain('native/provider/'));
    expect(fetch).toHaveBeenCalledTimes(1);
    await act(async () => { operatorSession.disconnect(); });
    expect(host.textContent).toBe('No authorized models');
    expect(fetch).toHaveBeenCalledTimes(1);
    await act(async () => { operatorSession.useDirectAccess(); });
    await vi.waitFor(() => expect(host.textContent).toContain('native/provider/'));
    expect(fetch).toHaveBeenCalledTimes(2);
  } finally {
    await act(async () => { root.unmount(); operatorSession.useDirectAccess(); });
    host.remove(); vi.unstubAllGlobals();
  }
});
