import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import type { ProposalOperation, ProposalReview } from '../../store/endpoints/plugins';
import { darkTheme } from '../../theme/theme';
import { NodeProposalsPanel, ProposalControls } from './NodeProposalsPanel';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({ useSkulkTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }));
let root: Root;
let host: HTMLDivElement;
let review: ProposalReview;
let operation: ProposalOperation | undefined;
let calls: { method: string; path: string; body: string }[];
let loseReply: boolean;
let failReview: boolean;
let store: ReturnType<typeof makeStore>;
const storageKey = 'skulk-proposal-operation-v1:managed.fixture:node';
function makeStore() { return configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer }, middleware: (defaults) => defaults().concat(apiSlice.middleware) }); }
async function contains(text: string) { await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain(text)); }); }
function button(text: string) { const result = [...host.querySelectorAll('button')].find((item) => item.textContent === text); if (!result) throw new Error(`Missing button ${text}`); return result; }
async function click(text: string) { await act(async () => { button(text).click(); }); }
async function render(actionsAvailable = true) { await act(async () => { root.render(<Provider store={store}><ThemeProvider theme={darkTheme}><NodeProposalsPanel pluginId="managed.fixture" nodeId="node" actionsAvailable={actionsAvailable} /></ThemeProvider></Provider>); }); }
beforeEach(async () => {
  localStorage.removeItem(storageKey);
  calls = []; loseReply = false; failReview = false; operation = undefined;
  review = { proposal: { reference: { pluginId: 'managed.fixture', nodeId: 'node', proposalId: 'opaque-intent', proposalDigest: 'a'.repeat(64) }, summary: 'Acquire GPU · maximum USD 1.00', state: 'pending_approval', expiresAt: 9999999999 }, approvalRevision: 'b'.repeat(64), observedAt: 100, fields: [{ label: 'Cleanup terms', value: '<script>unsafe markup</script> Independent termination after expiry.' }] };
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), location.href), init);
    const body = await request.text(); const path = new URL(request.url).pathname;
    calls.push({ method: request.method, path, body });
    let result: unknown; let status = 200;
    if (request.method === 'POST') {
      operation = { operationId: path.endsWith('/resume') ? operation!.operationId : JSON.parse(body).operationId, reference: review.proposal.reference, phase: 'uncertain', updatedAt: 101, code: 'submission_uncertain', correctiveAction: 'Observe cleanup; never replay.' };
      result = operation;
      if (loseReply) { result = {}; status = 503; }
    } else if (path.includes('/proposal-operations/')) { result = operation ?? {}; status = operation ? 200 : 404; }
    else if (path.endsWith('/proposals')) result = { proposals: [review.proposal], nextOffset: null };
    else { result = failReview ? {} : review; status = failReview ? 503 : 200; }
    return new Response(JSON.stringify(result), { status, headers: { 'Content-Type': 'application/json' } });
  });
  store = makeStore(); host = document.createElement('div'); document.body.appendChild(host); root = createRoot(host); await render();
});
afterEach(async () => { await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); }); host.remove(); localStorage.removeItem(storageKey); vi.unstubAllGlobals(); });

it('reviews safe text and retains a lost approval response across reconnect without replay', async () => {
  expect(calls).toHaveLength(0);
  await click('Review proposals'); await contains('Acquire GPU'); await click('Review'); await contains('Cleanup terms');
  expect(host.querySelector('script')).toBeNull();
  expect(calls.every((call) => call.method === 'GET')).toBe(true);
  loseReply = true;
  await click('Approve'); await contains('The approval was sent but not confirmed');
  await contains('Operation ID');
  const writes = calls.filter((call) => call.method === 'POST'); expect(writes).toHaveLength(1);
  const sent = JSON.parse(writes[0].body);
  expect(sent).toEqual({ operationId: expect.stringMatching(/^[a-f0-9]{32}$/), reference: review.proposal.reference, reviewRevision: review.approvalRevision });
  expect(localStorage.getItem(storageKey)).toBe(sent.operationId);
  expect(button('Approve').disabled).toBe(true);
  await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
  store = makeStore(); root = createRoot(host); await render(); await click('Review proposals'); await contains('Not yet known whether it started');
  expect(host.textContent).not.toContain('Continue the approval');
  expect(calls.filter((call) => call.method === 'POST')).toHaveLength(1);
});

it('requires a fresh review and an available owner action facet', async () => {
  await render(false); await click('Review proposals'); await contains('Acquire GPU'); await click('Review'); await contains('Cleanup terms');
  expect(button('Approve').disabled).toBe(true);
  await render(true); await click('Refresh'); await contains('Acquire GPU');
  await act(async () => { await vi.waitFor(() => expect(button('Review').disabled).toBe(false)); });
  failReview = true; await click('Review'); await contains('This proposal cannot be read right now');
  const approve = [...host.querySelectorAll('button')].find((item) => item.textContent === 'Approve');
  expect(!approve || approve.disabled).toBe(true);
  expect(calls.every((call) => call.method === 'GET')).toBe(true);
});

it('resumes retained approval only on an explicit action without replacement intent', async () => {
  operation = { operationId: 'd'.repeat(32), reference: review.proposal.reference, phase: 'approval_interrupted', updatedAt: 101, code: 'owner_interrupted', correctiveAction: 'Review and resume.' };
  localStorage.setItem(storageKey, operation.operationId);
  await click('Review proposals'); await contains('The approval was interrupted');
  expect(calls.every((call) => call.method === 'GET')).toBe(true);
  await click('Continue the approval'); await contains('Not yet known whether it started');
  const writes = calls.filter((call) => call.method === 'POST'); expect(writes).toHaveLength(1);
  expect(writes[0].path).toBe(`/v1/plugins/managed.fixture/nodes/node/proposal-operations/${'d'.repeat(32)}/resume`);
  expect(writes[0].body).toBe('');
});


it('keeps uncertain submission visible beside confirmed absence and stale cleanup', async () => {
  operation = { operationId: 'e'.repeat(32), reference: review.proposal.reference, phase: 'uncertain', updatedAt: 102, code: 'submission_uncertain', correctiveAction: 'Never replay.', reconciliation: { state: 'absent', observedAt: 101, stale: true, code: 'reconciliation_unavailable' } };
  localStorage.setItem(storageKey, operation.operationId);
  await click('Review proposals'); await contains('Confirmed gone');
  expect(host.textContent).toContain('Not yet known whether it started');
  expect(host.textContent).toContain('The cleanup service could not be reached');
  expect(host.textContent).toContain(new Date(101_000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }));
  expect(host.textContent).not.toContain('Continue the approval');
  expect(calls.every((call) => call.method === 'GET')).toBe(true);
});


it('observes controller acknowledgement without offering to replay it', async () => {
  operation = { operationId: 'f'.repeat(32), reference: review.proposal.reference, phase: 'acknowledged', updatedAt: 102, code: null, correctiveAction: null, reconciliation: { state: 'pending', observedAt: 101, stale: false, code: null } };
  localStorage.setItem(storageKey, operation.operationId);
  await click('Review proposals'); await contains('Request accepted');
  expect(host.textContent).toContain('The cleanup service has not seen it yet');
  expect(host.textContent).not.toContain('Continue the approval');
  expect(calls.every((call) => call.method === 'GET')).toBe(true);
});


it('keeps the default review to plain words, without identifiers or lookup by ID', async () => {
  await act(async () => { root.render(<Provider store={store}><ThemeProvider theme={darkTheme}><ProposalControls pluginId="managed.fixture" nodeId="node" actionsAvailable /></ThemeProvider></Provider>); });
  await contains('Waiting for your approval'); await contains('approve by');
  await click('Review'); await contains('Cleanup terms');
  await contains('After that it expires and nothing is done');
  expect(host.textContent).not.toContain('a'.repeat(64));
  expect(host.textContent).not.toContain('Proposal digest');
  expect(host.querySelector('input')).toBeNull();
  operation = { operationId: 'c'.repeat(32), reference: review.proposal.reference, phase: 'refused', updatedAt: 103, code: 'budget_exhausted', correctiveAction: 'Raise your spending limit, then ask again.' };
  localStorage.setItem(storageKey, operation.operationId);
  await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
  store = makeStore(); root = createRoot(host);
  await act(async () => { root.render(<Provider store={store}><ThemeProvider theme={darkTheme}><ProposalControls pluginId="managed.fixture" nodeId="node" actionsAvailable /></ThemeProvider></Provider>); });
  await contains('Not done: the plugin refused it · Raise your spending limit, then ask again.');
  expect(host.textContent).not.toContain('cannot be approved from here');
  expect(host.textContent).not.toContain('budget_exhausted');
  expect(host.textContent).not.toContain('c'.repeat(32));
});

it('says nothing about approval for a proposal already decided', async () => {
  review = { ...review, approvalRevision: null, proposal: { ...review.proposal, state: 'approved' } };
  await click('Review proposals'); await contains('Acquire GPU'); await click('Review'); await contains('Cleanup terms');
  expect(host.textContent).not.toContain('cannot be approved from here');
});

it('says approval is unavailable, without a cause, for a waiting proposal', async () => {
  review = { ...review, approvalRevision: null };
  await click('Review proposals'); await contains('Acquire GPU'); await click('Review'); await contains('Cleanup terms');
  await contains('This cannot be approved from here right now.');
  expect(button('Approve').disabled).toBe(true);
});
