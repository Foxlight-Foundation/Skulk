import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { darkTheme } from '../../theme/theme';
import { ModelSearchModal } from './ModelSearchModal';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({ t: (_key: string, fallback: string, values?: Record<string, unknown>) =>
    fallback.replace(/\{(\w+)\}/g, (match, key: string) => values?.[key] === undefined ? match : String(values[key])) }),
}));

it('offers native artifacts for download while excluding delegated chat models', async () => {
  const requests: string[] = [];
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    requests.push(url);
    if (url === '/models') return new Response(JSON.stringify({ data: [
      { id: 'org/native-artifact', name: 'Native artifact', base_model: 'Native artifact', storage_size_megabytes: 100 },
      { id: 'capability/fixture.chat/remote', name: 'Delegated provider', base_model: 'Delegated provider', inference_capability: {
        contract: 'chat.completions@0.1.0', expires_at: Date.now() / 1000 + 60,
      } },
    ] }));
    if (url.startsWith('/models/search')) return new Response('[]');
    return new Response(JSON.stringify({ status: 'ok' }));
  }));
  const container = document.createElement('div'); document.body.append(container);
  const root = createRoot(container);
  const started = vi.fn();
  try {
    await act(async () => { root.render(<ThemeProvider theme={darkTheme}>
      <ModelSearchModal open onClose={vi.fn()} existingModelIds={new Set()} activeDownloads={[]} onDownloadStarted={started} />
    </ThemeProvider>); });
    await expect.poll(() => container.textContent).toContain('Native artifact');
    expect(container.textContent).not.toContain('Delegated provider');
    await act(async () => { await userEvent.click(container.querySelector<HTMLButtonElement>('[aria-label="Download org/native-artifact"]')!); });
    await expect.poll(() => started.mock.calls).toEqual([['org/native-artifact']]);
    expect(requests.some(url => url.includes('capability'))).toBe(false);
  } finally {
    await act(async () => root.unmount());
    container.remove();
    vi.unstubAllGlobals();
  }
});
