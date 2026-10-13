import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import type { BurstBudget, BurstOperation } from '../../store/endpoints/burst';
import type { SetupActions, SetupOperation } from '../../store/endpoints/plugins';
import { darkTheme } from '../../theme/theme';
import { BurstRentalsView } from './BurstRentalsView';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string, params?: Record<string, unknown>) =>
      fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? `{${name}}`)),
  }),
}));
// A short wait, so an unconfirmed reply settles within a test.
vi.mock('./setupRunTiming', () => ({ UNCONFIRMED_WAIT_MS: 100, UNCONFIRMED_POLL_MS: 100 }));

const SETUP_PATH = '/v1/plugins/managed.runpod/nodes/gpu/setup-actions';
const OPERATIONS_PATH = '/v1/plugins/managed.runpod/nodes/gpu/setup-operations';

let root: Root;
let host: HTMLDivElement;
let calls: { method: string; path: string; body: string }[];
let budget: BurstBudget;
let rentals: BurstOperation[];
let operations: SetupOperation[];
let refuseSetup: { status: number; detail: string } | null;
let limitSchema: Record<string, unknown>;
let startOverSchema: Record<string, unknown>;
let loseReply: boolean;

/** The change-limit form RunPod's provider publishes. */
const RUNPOD_LIMIT_SCHEMA = {
  type: 'object',
  required: ['spending_limit_usd'],
  properties: {
    spending_limit_usd: { type: 'number', exclusiveMinimum: 0, maximum: 1000, 'x-skulk-unit': 'usd' },
    maximum_rentals: { type: 'integer', minimum: 1, 'x-skulk-advanced': true },
  },
};

function makeStore() {
  return configureStore({
    reducer: { [apiSlice.reducerPath]: apiSlice.reducer },
    middleware: (defaults) => defaults().concat(apiSlice.middleware),
  });
}

function available(update: Partial<BurstBudget> = {}): BurstBudget {
  return {
    pluginId: 'managed.runpod',
    nodeId: 'gpu',
    state: 'available',
    message: null,
    limitUsd: '10.000000',
    spentUsd: '2.500000',
    heldUsd: '1.400000',
    leftUsd: '6.100000',
    rentals: 2,
    countingSince: 1_800_000_000,
    changeLimitAction: 'campaign-limits',
    startOverAction: 'start-over',
    ...update,
  };
}

function rental(update: Partial<BurstOperation> = {}): BurstOperation {
  const reference = { pluginId: 'managed.runpod', nodeId: 'gpu', proposalId: 'a'.repeat(64), proposalDigest: 'a'.repeat(64) };
  return {
    pluginId: 'managed.runpod',
    nodeId: 'gpu',
    operationId: '1'.repeat(32),
    phase: 'ready',
    message: 'The model is ready on the remote machine.',
    instanceId: 'instance-1',
    modelId: 'example/large',
    resourceExpiresAt: 1_900_000_000,
    acquisition: { operationId: '1'.repeat(32), reference, phase: 'acknowledged', updatedAt: 1, code: null, correctiveAction: null },
    release: null,
    ...update,
  };
}

function snapshot(): SetupActions {
  return {
    nodeId: 'gpu',
    revision: 3,
    schemaDigest: 'c'.repeat(64),
    credentialRevision: 4,
    credentialSchemaDigest: 'd'.repeat(64),
    actions: [
      { actionId: 'campaign-limits', title: 'Change spending limit', description: '', parametersSchema: limitSchema, schemaDigest: 'e'.repeat(64), requiresApproval: true },
      { actionId: 'start-over', title: 'Start over', description: '', parametersSchema: startOverSchema, schemaDigest: 'f'.repeat(64), requiresApproval: true },
    ],
    operations,
  };
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

async function contains(text: string) {
  await act(async () => {
    await vi.waitFor(() => expect(host.textContent).toContain(text), { timeout: 5000 });
  });
}

function button(text: string) {
  const result = [...host.querySelectorAll('button')].find((item) => item.textContent === text);
  if (!result) throw new Error(`Missing button ${text}`);
  return result;
}

async function enabled(text: string) {
  await act(async () => {
    await vi.waitFor(() => expect(button(text).disabled).toBe(false), { timeout: 5000 });
  });
}

async function click(text: string) {
  await act(async () => {
    button(text).click();
  });
}

async function render() {
  await act(async () => {
    root.render(
      <Provider store={makeStore()}>
        <ThemeProvider theme={darkTheme}>
          <BurstRentalsView nodeId="gpu" pluginId="managed.runpod" />
        </ThemeProvider>
      </Provider>,
    );
  });
}

function setValue(input: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
  setter?.call(input, value);
  input.dispatchEvent(new Event('input', { bubbles: true }));
}

beforeEach(() => {
  calls = [];
  budget = available();
  rentals = [rental(), rental({ pluginId: 'managed.other', operationId: '2'.repeat(32), modelId: 'example/elsewhere' })];
  operations = [];
  refuseSetup = null;
  limitSchema = RUNPOD_LIMIT_SCHEMA;
  startOverSchema = { type: 'object', properties: {}, additionalProperties: false };
  loseReply = false;
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), location.href), init);
    const body = await request.text();
    const url = new URL(request.url);
    calls.push({ method: request.method, path: url.pathname, body });
    if (url.pathname === '/v1/burst/budget') return json(budget);
    if (url.pathname === '/v1/burst/operations') return json({ scope: 'local_api_host', observedAt: 1, rentals });
    if (url.pathname.startsWith('/v1/burst/operations/')) return json(rental({ phase: 'releasing' }));
    if (url.pathname === '/v1/plugins') return json([{ pluginId: 'managed.runpod', available: true, nodes: [{ nodeId: 'gpu', bundleId: 'foxlight.runpod', version: '0.6.0', status: 'ready', configurable: true }] }]);
    if (url.pathname === SETUP_PATH) return json(snapshot());
    if (url.pathname === OPERATIONS_PATH) {
      if (refuseSetup) return json({ detail: refuseSetup.detail }, refuseSetup.status);
      // The reply is lost: the host may or may not have taken the change.
      if (loseReply) return json({ detail: 'upstream unavailable' }, 503);
      const mutation = JSON.parse(body) as { operationId: string; actionId: string };
      // The provider applies the change before the next read of its forms.
      operations = [{ operationId: mutation.operationId, nodeId: 'gpu', actionId: mutation.actionId, phase: 'complete', code: null, correctiveAction: null }];
      return json({ ...operations[0], phase: 'queued' });
    }
    return json({ detail: 'unexpected' }, 404);
  });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
});

it('shows spending in dollars and cents', async () => {
  await render();
  await contains('$2.50 of $10.00 spent');
  expect(host.textContent).toContain('$6.10');
  expect(host.textContent).toContain('$1.40 held for machines running now');
  expect(host.textContent).toContain('Counting since');
});

it('says plainly when the limit cannot be read', async () => {
  budget = available({ state: 'unavailable', message: 'The cleanup service that keeps your spending limit did not answer. Check again in a moment.', limitUsd: null, spentUsd: null, heldUsd: null, leftUsd: null });
  await render();
  await contains('did not answer');
  expect(host.textContent).not.toContain('spent');
  expect(() => button('Change limit')).toThrow();
});

it('changes the limit through the provider action, bound to the forms it read', async () => {
  await render();
  await contains('$2.50 of $10.00 spent');
  await click('Change limit');
  const input = host.querySelector<HTMLInputElement>('input');
  expect(input?.value).toBe('10.00');
  await act(async () => setValue(input!, '20'));
  await enabled('Save');
  const budgetReads = calls.filter((call) => call.path === '/v1/burst/budget').length;
  await click('Save');
  await contains('Your limit is now $20.00.');
  const submitted = calls.find((call) => call.path === OPERATIONS_PATH);
  expect(submitted?.method).toBe('POST');
  const mutation = JSON.parse(submitted!.body) as Record<string, unknown>;
  expect(mutation).toMatchObject({
    actionId: 'campaign-limits',
    expectedRevision: 3,
    expectedCredentialRevision: 4,
    expectedActionSchemaDigest: 'e'.repeat(64),
    expectedRequiresApproval: true,
    values: { spending_limit_usd: 20 },
  });
  // A completed change reads the limit again.
  await act(async () => {
    await vi.waitFor(() => expect(calls.filter((call) => call.path === '/v1/burst/budget').length).toBeGreaterThan(budgetReads));
  });
});

it('shows the provider refusal in its own words', async () => {
  refuseSetup = { status: 409, detail: 'Turn this node off before changing its setup.' };
  await render();
  await contains('$2.50 of $10.00 spent');
  await click('Change limit');
  await enabled('Save');
  await click('Save');
  await contains('Turn this node off before changing its setup.');
});

it('starts over after a confirm step', async () => {
  await render();
  await contains('$2.50 of $10.00 spent');
  await click('Start over');
  await contains('Finished rentals stop counting, so your whole limit is available again.');
  await enabled('Start over');
  await click('Start over');
  await contains('Started over. Finished rentals no longer count.');
  const submitted = calls.find((call) => call.path === OPERATIONS_PATH);
  expect(JSON.parse(submitted!.body)).toMatchObject({ actionId: 'start-over', values: {} });
});

it('offers no changes where the limit is changed elsewhere', async () => {
  budget = available({ changeLimitAction: null, startOverAction: null, message: 'Change the spending limit on the cleanup computer.' });
  await render();
  await contains('Change the spending limit on the cleanup computer.');
  expect(() => button('Change limit')).toThrow();
  expect(() => button('Start over')).toThrow();
});

it('lists only this provider’s machines and releases one', async () => {
  await render();
  await contains('example/large');
  expect(host.textContent).not.toContain('example/elsewhere');
  expect(host.textContent).toContain('Ready');
  expect(host.textContent).toContain('Ends by');
  await click('Release');
  await contains('Releasing the machine.');
  expect(calls.some((call) => call.method === 'DELETE' && call.path === `/v1/burst/operations/${'1'.repeat(32)}`)).toBe(true);
});

it('says when no machines are running', async () => {
  rentals = [];
  await render();
  await contains('No rented machines are running.');
});

it('sends the amount as the field the provider’s own form names, within its bounds', async () => {
  limitSchema = {
    type: 'object',
    required: ['budget'],
    properties: { budget: { type: 'number', minimum: 5, maximum: 50, 'x-skulk-unit': 'usd' } },
  };
  await render();
  await contains('$2.50 of $10.00 spent');
  await click('Change limit');
  const input = host.querySelector<HTMLInputElement>('input')!;
  await act(async () => setValue(input, '80'));
  await contains('Enter an amount of at least $5.00 and at most $50.00.');
  expect(button('Save').disabled).toBe(true);
  await act(async () => setValue(input, '30'));
  await enabled('Save');
  await click('Save');
  await contains('Your limit is now $30.00.');
  const submitted = calls.find((call) => call.path === OPERATIONS_PATH);
  expect((JSON.parse(submitted!.body) as { values: unknown }).values).toEqual({ budget: 30 });
});

it('never offers an unconfirmed start-over again, even after closing it', async () => {
  loseReply = true;
  await render();
  await contains('$2.50 of $10.00 spent');
  await click('Start over');
  await enabled('Start over');
  await click('Start over');
  await contains('The provider did not confirm this change, so it may already have been made.');
  expect(() => button('Start over')).toThrow();
  await click('Done');
  await click('Start over');
  await contains('The provider did not confirm this change, so it may already have been made.');
  // Reopened, it still shows the unconfirmed change and offers no second submit.
  expect(() => button('Start over')).toThrow();
  expect(calls.filter((call) => call.path === OPERATIONS_PATH)).toHaveLength(1);
});

it('offers start over from one button only when its form takes no values', async () => {
  startOverSchema = {
    type: 'object',
    required: ['confirm'],
    properties: { confirm: { type: 'boolean' } },
  };
  await render();
  await contains('$2.50 of $10.00 spent');
  await click('Start over');
  await contains("This provider can't start over from here.");
  expect(calls.some((call) => call.path === OPERATIONS_PATH)).toBe(false);
});

it('says plainly when the provider’s limit form needs more than one amount', async () => {
  limitSchema = {
    type: 'object',
    required: ['amount', 'reason'],
    properties: { amount: { type: 'number' }, reason: { type: 'string' } },
  };
  await render();
  await contains('$2.50 of $10.00 spent');
  await click('Change limit');
  await contains("This provider's limit can't be changed here.");
  expect(calls.some((call) => call.path === OPERATIONS_PATH)).toBe(false);
});
