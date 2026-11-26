import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import { applyEngineChange, engineChangeErrorDetail } from '../../store/endpoints/engines';
import { darkTheme } from '../../theme/theme';
import type { EnginePackage, EnginesReport } from '../../types/engines';
import { EnginesPanel } from './EnginesPanel';
import { engineBuildRows, engineProblemSentence, laneLabel } from './enginesView';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
const translate = (_key: string, fallback: string, params?: Record<string, unknown>) =>
  fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? ''));
vi.mock('../../i18n/tolgee', () => ({ useSkulkTranslation: () => ({ t: translate }) }));

const FIRST = 'a'.repeat(64);
const SECOND = 'b'.repeat(64);
const THIRD = 'c'.repeat(64);
const ENGINE = 'foxlight.tensorfold';

function held(overrides: Partial<EnginePackage> = {}): EnginePackage {
  return {
    engineId: ENGINE, lane: 'rocm', backendTag: 'tensorfold-rocm', status: 'active', build: SECOND,
    engineVersion: '1.3.2', generation: 2, publisher: 'foxlight', channel: null,
    installedBuilds: [FIRST, SECOND, THIRD], rollbackBuild: FIRST, problem: null,
    downloadBytes: null, overheadBytes: 2 * 1024 ** 3, loadResults: [], ...overrides,
  };
}
function reportOf(...nodes: { nodeId: string; package: EnginePackage }[]): EnginesReport {
  return { packagesEnabled: true, engines: nodes.length ? [{ engineId: ENGINE, name: 'tensorfold', nodes }] : [] };
}

describe('the builds a machine holds', () => {
  it('lists the build in use first, the earlier one next, then the rest', () => {
    expect(engineBuildRows(held({ installedBuilds: [THIRD, FIRST, SECOND] }))).toEqual([
      { build: SECOND, role: 'inUse' },
      { build: FIRST, role: 'earlier' },
      { build: THIRD, role: 'other' },
    ]);
  });

  it('marks the reported build of an unusable engine as the one that cannot be used', () => {
    expect(engineBuildRows(held({ status: 'unusable', problem: 'installation_damaged', rollbackBuild: null, installedBuilds: [FIRST, SECOND] }))).toEqual([
      { build: SECOND, role: 'unusable' },
      { build: FIRST, role: 'other' },
    ]);
  });

  it('lists the build in use even when the machine did not list its disk', () => {
    expect(engineBuildRows(held({ installedBuilds: undefined, rollbackBuild: null }))).toEqual([{ build: SECOND, role: 'inUse' }]);
  });

  it('has nothing to list for an engine that is only offered', () => {
    expect(engineBuildRows(held({ status: 'offered', installedBuilds: [], rollbackBuild: null }))).toEqual([]);
  });
});

describe('words for hardware and problems', () => {
  it('names each kind of hardware and passes an unknown one through', () => {
    expect(['metal', 'cuda', 'rocm', 'vulkan', 'cpu', 'npu'].map(lane => laneLabel(lane, translate)))
      .toEqual(['Apple GPU', 'NVIDIA GPU', 'AMD GPU (ROCm)', 'GPU (Vulkan)', 'Processor only', 'npu']);
  });

  it('explains a known problem and shows the code of an unknown one', () => {
    expect(engineProblemSentence('installation_damaged', translate)).toContain('not intact');
    expect(engineProblemSentence('protocol_unsupported', translate)).toBe('This version of Skulk cannot run it.');
    expect(engineProblemSentence('signature_invalid', translate)).toBe('The machine cannot use it (signature_invalid).');
    expect(engineProblemSentence(null, translate)).toContain('did not say why');
  });
});

describe('a machine\'s answer to a change', () => {
  it('replaces that machine\'s entry and leaves the others', () => {
    const before = reportOf({ nodeId: 'node-a', package: held() }, { nodeId: 'node-b', package: held({ lane: 'cuda' }) });
    const after = applyEngineChange(before, { nodeId: 'node-a', engineId: ENGINE, lane: 'rocm', action: 'activate', package: held({ build: FIRST, rollbackBuild: SECOND }) });
    expect(after.engines[0]!.nodes.map(entry => [entry.nodeId, entry.package.build])).toEqual([['node-a', FIRST], ['node-b', SECOND]]);
    expect(before.engines[0]!.nodes[0]!.package.build).toBe(SECOND);
  });

  it('drops the machine, and then the engine, when nothing is held or offered', () => {
    const before = reportOf({ nodeId: 'node-a', package: held() });
    expect(applyEngineChange(before, { nodeId: 'node-a', engineId: ENGINE, lane: 'rocm', action: 'remove', package: null }).engines).toEqual([]);
  });

  it('reads the sentence a node sends with a refusal in either shape', () => {
    expect(engineChangeErrorDetail({ status: 409, data: { error: { message: 'Stop it first.' } } })).toBe('Stop it first.');
    expect(engineChangeErrorDetail({ status: 409, data: { detail: 'Not there.' } })).toBe('Not there.');
    expect(engineChangeErrorDetail({ status: 'FETCH_ERROR', error: 'offline' })).toBeNull();
  });
});

describe('the engines panel', () => {
  let root: Root;
  let host: HTMLDivElement;
  let store: ReturnType<typeof makeStore>;
  let report: EnginesReport;
  let changes: { path: string; body: Record<string, unknown> }[];
  let answer: (path: string, body: Record<string, unknown>) => Response;

  function makeStore() {
    return configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer }, middleware: defaults => defaults().concat(apiSlice.middleware) });
  }
  const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
  const control = (label: string) => host.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`);
  const button = (text: string) => [...host.querySelectorAll('button')].find(candidate => candidate.textContent === text);
  async function click(target: HTMLButtonElement | null | undefined) {
    if (!target) throw new Error('Missing button');
    await act(async () => { target.click(); });
  }
  async function shows(text: string) {
    await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain(text)); });
  }
  async function render() {
    await act(async () => { root.render(<Provider store={store}><ThemeProvider theme={darkTheme}><EnginesPanel /></ThemeProvider></Provider>); });
  }

  beforeEach(() => {
    report = reportOf({ nodeId: 'node-a', package: held() });
    changes = [];
    answer = () => json({ error: { message: 'unexpected' } }, 500);
    vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = new Request(input, init);
      const path = new URL(request.url).pathname;
      if (path === '/state') return json({ nodeIdentities: { 'node-a': { friendlyName: 'Workshop PC' } } });
      if (path === '/v1/engines') return json(report);
      const body = await request.json() as Record<string, unknown>;
      changes.push({ path, body });
      return answer(path, body);
    });
    store = makeStore();
    host = document.createElement('div');
    document.body.appendChild(host);
    root = createRoot(host);
  });
  afterEach(async () => {
    await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
    host.remove();
    vi.unstubAllGlobals();
  });

  it('says so when no machine has or is offered an engine', async () => {
    report = reportOf();
    await render();
    await shows('No machine has an add-on engine');
    expect(host.querySelectorAll('article')).toHaveLength(0);
  });

  it('says add-on engines are turned off instead of that there are none', async () => {
    report = { packagesEnabled: false, engines: [] };
    await render();
    await shows('Add-on engines are turned off');
    expect(host.textContent).not.toContain('No machine has an add-on engine');
  });

  it('says which models the build in use cannot load', async () => {
    const refused = (build: string, modelId: string) => ({ build, outcome: 'refused' as const, code: 'unsupported_model', modelId, atUnixMs: 1_800_000_000_000 });
    report = reportOf({ nodeId: 'node-a', package: held({ unsupportedModels: [refused(SECOND, 'org/small'), refused(SECOND, 'org/tiny'), refused(FIRST, 'org/old')] }) });
    await render();
    await shows('This engine cannot load org/small, org/tiny. Skulk runs them on another engine.');
    // An earlier build's answer does not decide where a model goes now.
    expect(host.textContent).not.toContain('org/old');
  });

  it('names the machine, the hardware and each build with what it is', async () => {
    report = reportOf({ nodeId: 'node-a', package: held({ loadResults: [{ build: SECOND, outcome: 'refused', code: 'load_refused', modelId: 'org/model', atUnixMs: 1_800_000_000_000 }] }) });
    await render();
    await shows('Workshop PC');
    const text = host.textContent ?? '';
    expect(text).toContain('Installed');
    expect(text).toContain('AMD GPU (ROCm) · version 1.3.2 · sets aside 2GB of memory beside each model');
    expect(text).toContain('the engine refused org/model (load_refused');
    const builds = [...host.querySelectorAll('li')].map(item => item.textContent ?? '');
    expect(builds[0]).toContain('build bbbbbbbbbbbb');
    expect(builds[0]).toContain('Used for new launches');
    expect(builds[1]).toContain('build aaaaaaaaaaaa');
    expect(builds[1]).toContain('Earlier build');
    // The build in use is switched away from, never removed on its own.
    expect(control('Use build bbbbbbbbbbbb on Workshop PC')).toBeNull();
    expect(control('Remove build bbbbbbbbbbbb from Workshop PC')).toBeNull();
    expect(control('Use build aaaaaaaaaaaa on Workshop PC')).not.toBeNull();
    expect(control('Remove build cccccccccccc from Workshop PC')).not.toBeNull();
  });

  it('switches a machine to another build and shows the machine\'s answer', async () => {
    answer = (_path, body) => json({ nodeId: 'node-a', engineId: ENGINE, lane: 'rocm', action: 'activate', package: held({ build: String(body.build), rollbackBuild: SECOND }) });
    await render();
    await shows('Workshop PC');
    await click(control('Use build aaaaaaaaaaaa on Workshop PC'));
    await act(async () => { await vi.waitFor(() => expect(control('Use build bbbbbbbbbbbb on Workshop PC')).not.toBeNull()); });
    expect(changes).toEqual([{ path: `/v1/engines/${ENGINE}/activate`, body: { nodeId: 'node-a', lane: 'rocm', build: FIRST } }]);
    expect(control('Use build aaaaaaaaaaaa on Workshop PC')).toBeNull();
    expect([...host.querySelectorAll('li')][0]!.textContent).toContain('build aaaaaaaaaaaa');
  });

  it('removes one build only after a second click, and may be cancelled', async () => {
    answer = () => json({ nodeId: 'node-a', engineId: ENGINE, lane: 'rocm', action: 'remove', package: held({ installedBuilds: [FIRST, SECOND] }) });
    await render();
    await shows('Workshop PC');
    await click(control('Remove build cccccccccccc from Workshop PC'));
    expect(changes).toEqual([]);
    await click(button('Cancel'));
    expect(button('Delete this build')).toBeUndefined();
    await click(control('Remove build cccccccccccc from Workshop PC'));
    await click(button('Delete this build'));
    await act(async () => { await vi.waitFor(() => expect(host.textContent).not.toContain('build cccccccccccc')); });
    expect(changes).toEqual([{ path: `/v1/engines/${ENGINE}/remove`, body: { nodeId: 'node-a', lane: 'rocm', build: THIRD } }]);
  });

  it('removes the whole engine from a machine after saying what that means', async () => {
    answer = () => json({ nodeId: 'node-a', engineId: ENGINE, lane: 'rocm', action: 'remove', package: null });
    await render();
    await shows('Workshop PC');
    await click(control('Remove tensorfold from Workshop PC'));
    expect(host.textContent).toContain('Every build of tensorfold is deleted from Workshop PC. Model files are not touched');
    expect(changes).toEqual([]);
    await click(button('Remove the engine'));
    await shows('No machine has an add-on engine');
    expect(changes).toEqual([{ path: `/v1/engines/${ENGINE}/remove`, body: { nodeId: 'node-a', lane: 'rocm' } }]);
  });

  it('shows the machine\'s own sentence when it refuses, and changes nothing', async () => {
    answer = () => json({ error: { message: 'A model on the node is running on a build this would delete. Stop it first.' } }, 409);
    await render();
    await shows('Workshop PC');
    await click(control('Remove tensorfold from Workshop PC'));
    await click(button('Remove the engine'));
    await shows('A model on the node is running on a build this would delete. Stop it first.');
    expect(host.querySelector('[role=alert]')?.textContent).toContain('Stop it first.');
    expect(host.textContent).toContain('build bbbbbbbbbbbb');
  });

  it('offers no change for an engine a machine does not hold yet', async () => {
    report = reportOf({ nodeId: 'node-a', package: held({ status: 'offered', channel: 'candidate', installedBuilds: [], rollbackBuild: null, downloadBytes: 10 * 1024 ** 2, publisher: null }) });
    await render();
    await shows('Not installed yet');
    expect(host.textContent).toContain('Test build');
    expect(host.textContent).toContain('installs it the first time a model placed here needs it (about 10MB to download)');
    expect(host.querySelectorAll('article button')).toHaveLength(0);
  });

  it('explains an engine a machine cannot use and still lets it be switched or removed', async () => {
    report = reportOf({ nodeId: 'node-a', package: held({ status: 'unusable', problem: 'installation_damaged', rollbackBuild: null, installedBuilds: [FIRST, SECOND] }) });
    await render();
    await shows('Cannot be used');
    expect(host.querySelector('article [role=alert]')?.textContent).toContain('not intact');
    expect(control('Use build bbbbbbbbbbbb on Workshop PC')).toBeNull();
    expect(control('Use build aaaaaaaaaaaa on Workshop PC')).not.toBeNull();
    expect(control('Remove tensorfold from Workshop PC')).not.toBeNull();
  });
});
