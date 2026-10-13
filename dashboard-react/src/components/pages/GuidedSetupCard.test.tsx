import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import { darkTheme } from '../../theme/theme';
import { GuidedSetupCard } from './GuidedSetupCard';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({ t: (_key: string, fallback: string, params?: Record<string, unknown>) => fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? '')) }),
}));

const pluginId = 'managed.' + '3'.repeat(32);
const node = `/v1/plugins/${pluginId}/nodes/runpod`;
const guidedSchema: Record<string, unknown> = {
  type: 'object',
  additionalProperties: false,
  required: ['spending_limit_usd'],
  properties: {
    spending_limit_usd: { type: 'number', title: 'Spend no more than', exclusiveMinimum: 0, maximum: 1000, 'x-skulk-unit': 'usd' },
    pod_backup_minutes: { type: 'integer', title: 'Wait this long before a machine deletes itself', default: 10, 'x-skulk-unit': 'minutes', 'x-skulk-advanced': true },
  },
  'x-skulk-guided': { credentials: ['runpod-api-key'], turn_on: true },
};

let root: Root;
let host: HTMLDivElement;
let guided: boolean;
let keyReady: boolean;
let credentialRevision: number;
let operations: { operationId: string; nodeId: string; actionId: string; phase: string; code: string | null; correctiveAction: string | null }[];
let setupStatus: number;
// The start lands but its reply is lost.
let dropStartReply: boolean;
let credentialPosts: Record<string, unknown>[];
let setupPosts: Record<string, unknown>[];
let turnedOn: number;

function json(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }); }
async function contains(text: string) { await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain(text), { timeout: 5000 }); }); }
function button(label: string) { return [...host.querySelectorAll('button')].find((item) => item.textContent === label) ?? null; }
function field(label: string): HTMLInputElement {
  const target = [...host.querySelectorAll('label')].find((item) => item.textContent === label);
  if (!target) throw new Error(`Missing field: ${label}`);
  return host.querySelector<HTMLInputElement>(`[id="${target.htmlFor}"]`)!;
}
async function type(element: HTMLInputElement, text: string) {
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(element, text);
    element.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

beforeEach(async () => {
  guided = true;
  keyReady = false;
  credentialRevision = 0;
  operations = [];
  setupStatus = 200;
  dropStartReply = false;
  credentialPosts = [];
  setupPosts = [];
  turnedOn = 0;
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), location.href), init);
    const path = new URL(request.url).pathname;
    if (path === `${node}/setup-actions`) {
      // Once setup completes, the plugin no longer offers it.
      const offered = guided && !operations.some((item) => item.phase === 'complete');
      for (const item of operations) if (item.phase === 'running') item.phase = 'complete';
      return json({
        nodeId: 'runpod', revision: 0, schemaDigest: 'a'.repeat(64), credentialRevision, credentialSchemaDigest: 'c'.repeat(64),
        actions: offered ? [{
          actionId: 'clean-up-here', title: 'Set up RunPod on this computer', description: 'Rents a GPU machine only after you approve each one.',
          requiresApproval: true, parametersSchema: guided ? guidedSchema : {}, schemaDigest: 'd'.repeat(64),
        }] : [],
        operations,
      });
    }
    if (path === `${node}/credentials`) {
      if (request.method === 'POST') {
        credentialPosts.push(await request.json() as Record<string, unknown>);
        keyReady = true;
        credentialRevision += 1;
        return json({});
      }
      return json({ nodeId: 'runpod', revision: credentialRevision, schemaDigest: 'c'.repeat(64),
        credentials: [{ credentialId: 'runpod-api-key', title: 'RunPod API key', description: 'In your RunPod account, open Settings, then API Keys.', required: false, ready: keyReady }] });
    }
    if (path === `${node}/setup-operations` && request.method === 'POST') {
      const body = await request.json() as Record<string, unknown>;
      setupPosts.push(body);
      if (setupStatus !== 200) return json({ detail: 'Turn this node off before setting up cleanup.' }, setupStatus);
      const started = { operationId: String(body.operationId), nodeId: 'runpod', actionId: 'clean-up-here', phase: 'running', code: 'cleanup_runs_while_signed_in',
        correctiveAction: 'Cleanup runs while you are signed in to this computer.' };
      operations = [started];
      if (dropStartReply) throw new TypeError('network connection lost');
      return json(started);
    }
    return json({ detail: 'not found' }, 404);
  });
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(async () => { root.unmount(); });
  host.remove();
  apiSlice.util.resetApiState();
  vi.unstubAllGlobals();
});

async function render() {
  const store = configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer }, middleware: (defaults) => defaults().concat(apiSlice.middleware) });
  await act(async () => {
    root.render(<Provider store={store}><ThemeProvider theme={darkTheme}>
      <GuidedSetupCard address={{ pluginId, nodeId: 'runpod' }} onTurnOn={() => { turnedOn += 1; }} />
    </ThemeProvider></Provider>);
  });
}

it('asks for the key and one amount, with everything else under Advanced', async () => {
  await render();
  await contains('Set up RunPod on this computer');
  await contains('RunPod API key');
  expect(host.querySelector('input[type="password"]')).not.toBeNull();
  expect(host.querySelector('details')?.open).toBe(false);
  expect(button('Set up')?.disabled).toBe(true);
  await type(field('Spend no more than'), '10');
  expect(button('Set up')?.disabled).toBe(false);
});

it('saves the key, runs the setup with fresh fences, and turns the plugin on once', async () => {
  await render();
  await contains('RunPod API key');
  await type(host.querySelector<HTMLInputElement>('input[type="password"]')!, 'rp-secret-key');
  await type(field('Spend no more than'), '10');
  await act(async () => { button('Set up')!.click(); });
  await contains('Set up. Turning it on now.');
  await contains('Cleanup runs while you are signed in to this computer.');
  expect(credentialPosts).toHaveLength(1);
  expect(credentialPosts[0]).toMatchObject({ operation: 'replace', credentialId: 'runpod-api-key', value: 'rp-secret-key', expectedRevision: 0 });
  expect(setupPosts).toHaveLength(1);
  // The key moved the credential fence; the setup is bound to the fresh one.
  expect(setupPosts[0]).toMatchObject({
    actionId: 'clean-up-here', expectedCredentialRevision: 1, expectedRequiresApproval: true, values: { spending_limit_usd: 10 },
  });
  // The secret never stays in an input.
  expect(host.querySelector<HTMLInputElement>('input[type="password"]')?.value ?? '').toBe('');
  await act(async () => { await vi.waitFor(() => expect(turnedOn).toBe(1)); });
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 50)); });
  expect(turnedOn).toBe(1);
});

it("shows the plugin's own words when it refuses", async () => {
  keyReady = true;
  setupStatus = 409;
  await render();
  await contains('Saved. Enter a new one only to replace it.');
  await type(field('Spend no more than'), '5');
  await act(async () => { button('Set up')!.click(); });
  await contains('Turn this node off before setting up cleanup.');
  // A saved key is not sent again.
  expect(credentialPosts).toHaveLength(0);
  expect(turnedOn).toBe(0);
});

it('asks for a missing key before starting', async () => {
  await render();
  await contains('RunPod API key');
  await type(field('Spend no more than'), '5');
  await act(async () => { button('Set up')!.click(); });
  await contains('Add your RunPod API key first.');
  expect(setupPosts).toHaveLength(0);
});

it('shows nothing when the plugin offers no guided setup', async () => {
  guided = false;
  await render();
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 100)); });
  expect(host.textContent).toBe('');
});

it('follows a start whose reply was lost instead of starting again', async () => {
  keyReady = true;
  dropStartReply = true;
  await render();
  await contains('Saved. Enter a new one only to replace it.');
  await type(field('Spend no more than'), '10');
  await act(async () => { button('Set up')!.click(); });
  await contains('Set up. Turning it on now.');
  expect(setupPosts).toHaveLength(1);
  await act(async () => { await vi.waitFor(() => expect(turnedOn).toBe(1)); });
});
