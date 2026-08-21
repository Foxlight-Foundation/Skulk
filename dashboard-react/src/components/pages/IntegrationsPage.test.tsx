import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { darkTheme } from '../../theme/theme';
import { IntegrationsPage } from './IntegrationsPage';
import type { InstanceCardData } from '../layout/InstancePanel';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}));

vi.mock('../../hooks/useToast', () => ({
  addToast: vi.fn(),
}));

const copyToClipboard = vi.fn(() => Promise.resolve());
vi.mock('../../utils/clipboard', () => ({
  copyToClipboard: (text: string) => copyToClipboard(text),
}));

let root: Root | null = null;
let container: HTMLDivElement | null = null;

function jsonResponse(payload: object): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

const REMOTE_ACCESS = {
  local: { ip: '192.168.1.50', port: 52415, url: 'http://192.168.1.50:52415' },
  tailscale: { running: false, ip: null, dnsName: null, port: 52415, url: null },
  preferredUrl: 'http://192.168.1.50:52415',
  operatorUrl: null,
};

const CATALOG = {
  data: [
    {
      id: 'org/Big-70B',
      context_length: 131072,
      resolved_capabilities: {
        supports_thinking: true,
        supports_thinking_toggle: true,
        thinking_format: 'token_delimited',
        supports_image_input: false,
        supports_tool_calling: true,
      },
    },
  ],
};

function readyInstance(modelId: string, status: string): InstanceCardData {
  return {
    instanceId: `instance-${modelId}`,
    modelId,
    sharding: 'Pipeline',
    instanceType: 'MlxRing',
    engine: 'mlx',
    nodeStatuses: [],
    status,
  } as unknown as InstanceCardData;
}

beforeEach(() => {
  copyToClipboard.mockClear();
  vi.stubGlobal('fetch', (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes('remote-access')) return Promise.resolve(jsonResponse(REMOTE_ACCESS));
    if (url.includes('/models')) return Promise.resolve(jsonResponse(CATALOG));
    return Promise.resolve(jsonResponse({}));
  });
});

afterEach(() => {
  if (root) {
    const current = root;
    act(() => current.unmount());
  }
  root = null;
  container = null;
  vi.unstubAllGlobals();
});

async function render(instances: InstanceCardData[]): Promise<HTMLDivElement> {
  container = document.createElement('div');
  document.body.appendChild(container);
  const created = createRoot(container);
  root = created;
  await act(async () => {
    created.render(
      <ThemeProvider theme={darkTheme}>
        <IntegrationsPage readyInstances={instances} />
      </ThemeProvider>,
    );
  });
  // Let the remote-access and catalog fetches settle.
  await act(async () => {
    await Promise.resolve();
  });
  return container;
}

/** Clicks the tool segment whose label matches exactly. */
async function selectTool(host: HTMLDivElement, label: string): Promise<void> {
  const button = Array.from(host.querySelectorAll('button')).find(
    element => element.textContent?.trim() === label,
  );
  expect(button, `expected a segment labelled ${label}`).toBeTruthy();
  await act(async () => {
    button!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
  });
}

describe('IntegrationsPage', () => {
  it('uses the routable node address rather than the page origin', async () => {
    const host = await render([readyInstance('org/Big-70B', 'ready')]);
    const text = host.textContent ?? '';
    expect(text).toContain('http://192.168.1.50:52415/v1');
    expect(text).toContain('http://192.168.1.50:52415/ollama');
  });

  it('fills the default Claude Code snippet with the ready model', async () => {
    const host = await render([readyInstance('org/Big-70B', 'ready')]);
    const code = host.querySelector('pre')?.textContent ?? '';
    expect(code).toContain('ANTHROPIC_BASE_URL=http://192.168.1.50:52415');
    expect(code).toContain('org/Big-70B');
    expect(code).toContain('claude');
  });

  it('swaps the snippets when another tool is chosen', async () => {
    const host = await render([readyInstance('org/Big-70B', 'ready')]);
    await selectTool(host, 'Hermes');
    const bodies = Array.from(host.querySelectorAll('pre'))
      .map(node => node.textContent ?? '')
      .join('\n');
    expect(bodies).toContain('provider: custom');
    expect(bodies).toContain('base_url: http://192.168.1.50:52415/v1');
    expect(bodies).not.toContain('ANTHROPIC_BASE_URL');
  });

  it('renders the AnythingLLM recipe with the generic OpenAI provider', async () => {
    const host = await render([readyInstance('org/Big-70B', 'ready')]);
    await selectTool(host, 'AnythingLLM');
    const bodies = Array.from(host.querySelectorAll('pre'))
      .map(node => node.textContent ?? '')
      .join('\n');
    expect(bodies).toContain("LLM_PROVIDER='generic-openai'");
    expect(bodies).toContain('GENERIC_OPEN_AI_MODEL_TOKEN_LIMIT=131072');
  });

  it('copies the exact snippet body', async () => {
    const host = await render([readyInstance('org/Big-70B', 'ready')]);
    const firstCode = host.querySelector('pre')?.textContent ?? '';
    const copyButton = Array.from(host.querySelectorAll('button')).find(element =>
      element.getAttribute('aria-label') === 'Copy',
    );
    expect(copyButton).toBeTruthy();
    await act(async () => {
      copyButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(copyToClipboard).toHaveBeenCalledWith(firstCode);
  });

  it('ignores instances that are not serving yet', async () => {
    const host = await render([readyInstance('org/Loading-8B', 'loading')]);
    const text = host.textContent ?? '';
    expect(text).toContain('No models are running yet');
    const code = host.querySelector('pre')?.textContent ?? '';
    expect(code).toContain('your-model-id');
    expect(code).not.toContain('org/Loading-8B');
  });
});
