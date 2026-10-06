// Copyright 2026 Foxlight Foundation

import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { userEvent } from 'vitest/browser';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { store } from '../../store';
import { apiSlice } from '../../store/api';
import { chatActions } from '../../store/slices/chatSlice';
import type { StewardState, StewardTransition } from '../../store/endpoints/steward';
import type { StewardActionProposal } from '../../store/endpoints/steward';
import { darkTheme } from '../../theme/theme';
import type { ModelInfo } from '../../types/models';
import { discoverSkulkSpeechSelection } from '../../audio/fabricSpeechDiscovery';
import { buildSkulkSpeechSynthesisRequest } from '../../audio/fabricSpeechRequest';
import type { InstanceCardData } from '../layout/InstancePanel';
import { StewardClusterPrompt, StewardControllerProvider, StewardChatView } from './StewardChatView';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });

const addToastSpy = vi.fn();
vi.mock('../../hooks/useToast', () => ({
  addToast: (toast: unknown) => addToastSpy(toast),
}));

vi.mock('../../i18n/tolgee', () => {
  const translate = (
    _key: string,
    fallback: string,
    params?: Record<string, string>,
  ) => {
    if (!params) return fallback;
    return Object.entries(params).reduce(
      (text, [name, value]) => text.replace(`{${name}}`, value),
      fallback,
    );
  };
  return {
    tolgee: { getLanguage: () => 'en' },
    useSkulkTranslation: () => ({ t: translate }),
  };
});

let root: Root | null = null;
let container: HTMLDivElement | null = null;

interface StewardStatusFixture {
  enabled: boolean;
  present: boolean;
  ready: boolean;
  steward_model: string | null;
  instance_id: string | null;
  desired_model: string | null;
  transition: StewardTransition;
  progress: number | null;
  state: StewardState;
}

function sseBody(events: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(`data: ${event}\n\n`));
      }
      controller.enqueue(encoder.encode('data: [DONE]\n\n'));
      controller.close();
    },
  });
}

function delta(delta: Record<string, string>): string {
  return JSON.stringify({ choices: [{ index: 0, delta }] });
}

function stubFetch(options: {
  status: StewardStatusFixture;
  sseEvents?: string[];
  openStream?: (signal: AbortSignal | null | undefined) => ReadableStream<Uint8Array>;
  chatStatus?: number;
  models?: ModelInfo[];
  proposals?: StewardActionProposal[];
  onDecision?: (body: unknown) => void;
  onChat?: (init: RequestInit | undefined) => void;
}): void {
  const fetchStub = async (
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> => {
    const url =
      typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    if (url.includes('/v1/chat/completions')) {
      options.onChat?.(init);
      if (options.chatStatus && options.chatStatus !== 200) {
        return new Response('chat unavailable', { status: options.chatStatus });
      }
      return new Response(options.openStream?.(init?.signal) ?? sseBody(options.sseEvents ?? []), {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      });
    }
    if (url.includes('/v1/steward/proposals/') && url.endsWith('/decision')) {
      const body = init?.body
        ? JSON.parse(String(init.body)) as unknown
        : input instanceof Request
          ? await input.clone().json() as unknown
          : undefined;
      options.onDecision?.(body);
      return new Response(JSON.stringify({ message: 'accepted' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    if (url.endsWith('/v1/steward/proposals')) {
      return new Response(JSON.stringify(options.proposals ?? []), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    if (url.endsWith('/v1/steward')) {
      return new Response(JSON.stringify(options.status), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    if (url.endsWith('/models')) {
      return new Response(JSON.stringify({ data: options.models ?? [] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    if (url.includes('/v1/audio/voices')) {
      return new Response(JSON.stringify({
        data: [{ id: 'skulk', name: 'Skulk', preferred_languages: ['en'] }],
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    return new Response('{}', { status: 404 });
  };
  vi.stubGlobal('fetch', fetchStub);
}

async function renderPage(readyInstances: InstanceCardData[] = []): Promise<void> {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(
      <Provider store={store}>
        <ThemeProvider theme={darkTheme}>
          <StewardChatView readyInstances={readyInstances} />
        </ThemeProvider>
      </Provider>,
    );
  });
}

async function waitFor(predicate: () => boolean, message: string): Promise<void> {
  const deadline = performance.now() + 5000;
  while (performance.now() < deadline) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error(message);
}

afterEach(async () => {
  addToastSpy.mockClear();
  await act(async () => root?.unmount());
  container?.remove();
  root = null;
  container = null;
  for (const conversation of Object.values(store.getState().chat.conversations)) {
    if (conversation.modelId === 'skulk/steward') store.dispatch(chatActions.deleteConversation(conversation.id));
  }
  store.dispatch(chatActions.setAutoSpeakAssistant(false));
  store.dispatch(apiSlice.util.resetApiState());
  vi.unstubAllGlobals();
});

const READY: StewardStatusFixture = {
  enabled: true,
  present: true,
  ready: true,
  steward_model: 'org/steward-4b',
  instance_id: 'inst-1',
  desired_model: 'org/steward-4b',
  transition: 'idle',
  progress: null,
  state: 'ready',
};

describe('StewardChatView', () => {
  it('continues Skulk voice discovery after an unusable speech model', async () => {
    const model = (id: string): ModelInfo => ({
      id,
      audio: {
        supports_streaming: true,
        supports_voice_listing: true,
        response_formats: ['pcm'],
      },
      resolved_capabilities: {
        supports_speech_synthesis: true,
        audio_response_formats: ['pcm'],
      } as ModelInfo['resolved_capabilities'],
    });
    const loadVoices = vi.fn(async (modelId: string) => {
      if (modelId === 'org/broken-tts') throw new Error('catalog unavailable');
      return [{ id: 'skulk', name: 'Skulk', preferredLanguages: ['en'] }];
    });

    const selection = await discoverSkulkSpeechSelection(
      [model('org/broken-tts'), model('org/working-tts')],
      new Set(['org/broken-tts', 'org/working-tts']),
      loadVoices,
      true,
    );

    expect(loadVoices).toHaveBeenCalledTimes(2);
    expect(selection?.model.modelId).toBe('org/working-tts');
    expect(selection?.voice.id).toBe('skulk');
  });

  it('pins the Skulk voice and deterministic seed on every fabric sentence', () => {
    const request = buildSkulkSpeechSynthesisRequest(
      'org/tts',
      'I am Skulk.',
      'English',
      new AbortController().signal,
    );
    const body: unknown = JSON.parse(String(request.body));

    expect(body).toMatchObject({
      model: 'org/tts',
      input: 'I am Skulk.',
      voice: 'skulk',
      seed: 42,
      stream: true,
      response_format: 'pcm',
    });
  });

  it('shows the disabled state when intelligent fabric is off', async () => {
    stubFetch({
      status: { ...READY, enabled: false, present: false, ready: false, state: 'disabled' },
    });
    await renderPage();
    await waitFor(
      () => container?.textContent?.includes('Intelligent Fabric is off') ?? false,
      'disabled state never rendered',
    );
  });

  it('holds the placing state until the steward is ready, not merely present', async () => {
    stubFetch({ status: { ...READY, ready: false, state: 'downloading' } });
    await renderPage();
    await waitFor(
      () => container?.textContent?.includes('Skulk is getting ready') ?? false,
      'placing state never rendered',
    );
    // The lifecycle word distinguishes "still staging weights" from "placing",
    // which is the difference between a minutes-long and a seconds-long wait.
    expect(container?.textContent?.includes('Status: downloading')).toBe(true);
    expect(container?.querySelector('textarea')).toBeNull();
  });

  it('streams a turn over chat-completions with the reserved model id', async () => {
    let chatInit: RequestInit | undefined;
    stubFetch({
      status: READY,
      sseEvents: [
        delta({ reasoning_content: 'get_cluster_state\n' }),
        delta({ content: 'All three nodes are healthy.' }),
      ],
      onChat: (init) => {
        chatInit = init;
      },
    });
    await renderPage();
    await waitFor(
      () => container?.textContent?.includes('Ask the cluster') ?? false,
      'empty chat state never rendered',
    );

    const textarea = container?.querySelector('textarea');
    expect(textarea).not.toBeNull();
    await userEvent.fill(textarea as HTMLTextAreaElement, 'Is the cluster healthy?');
    await userEvent.keyboard('{Enter}');

    await waitFor(
      () => container?.textContent?.includes('All three nodes are healthy.') ?? false,
      'steward reply never rendered',
    );
    expect(chatInit?.method).toBe('POST');
    const body: unknown = JSON.parse(String(chatInit?.body));
    expect((body as { model?: string }).model).toBe('skulk/steward');
    expect((body as { stream?: boolean }).stream).toBe(true);
    expect(container?.textContent).toContain('Is the cluster healthy?');
  });

  it('shows best-brain prestaging without blocking the serving steward', async () => {
    stubFetch({
      status: {
        ...READY,
        desired_model: 'org/steward-35b',
        transition: 'prestaging',
        progress: 0.42,
      },
    });
    await renderPage();
    await waitFor(
      () => container?.textContent?.includes('fabric transition: prestaging 42%') ?? false,
      'prestaging transition never rendered',
    );
    expect(container?.querySelector('textarea')).not.toBeNull();
  });

  it('shows an inert proposal and submits explicit approval separately', async () => {
    let decision: unknown;
    stubFetch({
      status: READY,
      proposals: [{
        proposal_id: 'proposal-1',
        action: 'restart_model',
        target: 'org/model',
        rationale: 'The runner is degraded.',
        evidence: ['Three failed probes.'],
        expected_effect: 'Replace the ordinary model instance.',
        created_at: new Date().toISOString(),
        expires_at: new Date(Date.now() + 60_000).toISOString(),
        status: 'pending',
        decided_at: null,
        decided_by: null,
        outcome: null,
      }],
      onDecision: (body) => { decision = body; },
    });
    await renderPage();
    await waitFor(
      () => container?.textContent?.includes('restart model: org/model') ?? false,
      'proposal never rendered',
    );
    expect(container?.textContent).toContain('Three failed probes.');

    const approveButton = [...(container?.querySelectorAll('button') ?? [])]
      .find((button) => button.textContent === 'Approve');
    expect(approveButton).toBeDefined();
    await userEvent.click(approveButton as HTMLButtonElement);

    await waitFor(() => decision !== undefined, 'approval was never submitted');
    expect(decision).toEqual({ approved: true });
  });
});

describe('StewardChatView stream errors', () => {
  it('clears speaking state when chat fails before the first content delta', async () => {
    const speechModel: ModelInfo = {
      id: 'org/tts',
      audio: {
        supports_streaming: true,
        supports_voice_listing: true,
        response_formats: ['pcm'],
      },
      resolved_capabilities: {
        supports_speech_synthesis: true,
        audio_response_formats: ['pcm'],
      } as ModelInfo['resolved_capabilities'],
    };
    vi.stubGlobal('AudioContext', class AudioContext {});
    store.dispatch(chatActions.setAutoSpeakAssistant(true));
    stubFetch({ status: READY, chatStatus: 503, models: [speechModel] });
    await renderPage([{
      instanceId: 'tts-1',
      modelId: 'org/tts',
      sharding: 'Pipeline',
      instanceType: 'MlxRing',
      engine: 'mlx',
      nodeStatuses: [],
      status: 'ready',
    }]);
    await waitFor(
      () => container?.querySelector('[aria-label="Speak draft"]') !== null,
      'speech discovery never completed',
    );
    const textarea = container?.querySelector('textarea');
    await userEvent.fill(textarea as HTMLTextAreaElement, 'hello?');
    await userEvent.keyboard('{Enter}');
    await waitFor(
      () => addToastSpy.mock.calls.length > 0,
      'chat failure never surfaced as a toast',
    );

    expect(container?.querySelector('[aria-label="Stop speech"]')).toBeNull();
    expect(container?.querySelector('[aria-label="Speak draft"]')).not.toBeNull();
  });

  it('surfaces a mid-stream error envelope instead of ending silently', async () => {
    stubFetch({
      status: READY,
      sseEvents: [
        delta({ reasoning_content: 'get_cluster_state\n' }),
        JSON.stringify({ error: { message: 'steward generation failed' } }),
      ],
    });
    await renderPage();
    await waitFor(
      () => container?.textContent?.includes('Ask the cluster') ?? false,
      'empty chat state never rendered',
    );
    const textarea = container?.querySelector('textarea');
    await userEvent.fill(textarea as HTMLTextAreaElement, 'hello?');
    await userEvent.keyboard('{Enter}');
    await waitFor(
      () =>
        addToastSpy.mock.calls.some((call) =>
          String((call[0] as { message?: string }).message).includes(
            'steward generation failed',
          ),
        ),
      'stream error never surfaced as a toast',
    );
  });
});


it('retains the draft and conversation when the shared controller changes presentation', async () => {
  const onChat = vi.fn();
  stubFetch({ status: READY, sseEvents: [delta({ content: 'Fixture reply.' })], onChat });
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  const instances: InstanceCardData[] = [];
  const present = async (key: string) => {
    await act(async () => root?.render(<Provider store={store}><ThemeProvider theme={darkTheme}>
      <StewardControllerProvider readyInstances={instances}><StewardChatView key={key} /></StewardControllerProvider>
    </ThemeProvider></Provider>));
    await act(async () => { await vi.waitFor(() => expect(container?.querySelector('textarea')).not.toBeNull()); });
  };
  await present('drawer');
  await act(async () => { await userEvent.fill(container!.querySelector('textarea')!, 'Keep this draft'); });
  await present('page');
  expect(container.querySelector('textarea')?.value).toBe('Keep this draft');
  expect(onChat).not.toHaveBeenCalled();
  await act(async () => { await userEvent.click(container!.querySelector('button[aria-label="Send message"]')!); });
  await act(async () => { await vi.waitFor(() => expect(container?.textContent).toContain('Fixture reply.')); });
  await present('virtual-model');
  expect(container.textContent).toContain('Keep this draft');
  expect(container.textContent).toContain('Fixture reply.');
  expect(onChat).toHaveBeenCalledOnce();
});

it('keeps an active generation across presentation changes and cancels it from the new view', async () => {
  let requestSignal: AbortSignal | null | undefined;
  const onChat = vi.fn();
  stubFetch({ status: READY, onChat, openStream: signal => {
    requestSignal = signal;
    return new ReadableStream({ start(controller) {
      controller.enqueue(new TextEncoder().encode(`data: ${delta({ content: 'Working on it' })}\n\n`));
      signal?.addEventListener('abort', () => controller.error(new DOMException('Aborted', 'AbortError')), { once: true });
    } });
  } });
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  const instances: InstanceCardData[] = [];
  const present = async (key: string) => {
    await act(async () => root?.render(<Provider store={store}><ThemeProvider theme={darkTheme}>
      <StewardControllerProvider readyInstances={instances}><StewardChatView key={key} /></StewardControllerProvider>
    </ThemeProvider></Provider>));
  };
  await present('drawer');
  await waitFor(() => container?.querySelector('textarea') !== null, 'composer did not mount');
  await userEvent.fill(container.querySelector('textarea')!, 'Inspect the cluster');
  await userEvent.click(container.querySelector('button[aria-label="Send message"]')!);
  await waitFor(() => container?.textContent?.includes('Working on it') ?? false, 'stream did not start');
  await present('page');
  expect(requestSignal?.aborted).toBe(false);
  expect(container.textContent).toContain('Working on it');
  await userEvent.click(container.querySelector('button[aria-label="Cancel generation"]')!);
  await waitFor(() => requestSignal?.aborted === true, 'stream was not aborted');
  expect(onChat).toHaveBeenCalledOnce();
});

it('saves drawer messages without selecting another chat and restores them after remount', async () => {
  const ordinary = store.dispatch(chatActions.newConversation('example/ordinary')).payload.id;
  stubFetch({ status: READY, sseEvents: [delta({ content: 'Saved reply.' })] });
  await renderPage();
  await waitFor(() => !!container?.querySelector('textarea'), 'composer did not mount');
  await userEvent.fill(container!.querySelector('textarea')!, 'Remember this turn');
  await userEvent.keyboard('{Enter}');
  await waitFor(() => container?.textContent?.includes('Saved reply.') ?? false, 'reply missing');
  const id = store.getState().chat.modelToConversationId['skulk/steward'];
  expect(store.getState().chat.activeConversationId).toBe(ordinary);
  expect(store.getState().chat.conversations[id].messages).toHaveLength(2);
  expect(JSON.parse(localStorage.getItem('skulk-chat')!).state.conversations[id].messages).toHaveLength(2);
  await act(async () => root?.unmount());
  container?.remove();
  await renderPage();
  await waitFor(() => container?.textContent?.includes('Saved reply.') ?? false, 'saved history missing after remount');
  await act(async () => { store.dispatch(chatActions.selectModel('skulk/steward')); });
  expect(store.getState().chat.activeConversationId).toBe(id);
  await act(async () => { store.dispatch(chatActions.renameConversation({ conversationId: id, name: 'Cluster inspection' })); });
  expect(store.getState().chat.conversations[id].name).toBe('Cluster inspection');
  store.dispatch(chatActions.deleteConversation(ordinary));
});

it.each(['new', 'delete'] as const)('cancels the owning request on %s and never redirects its reply', async action => {
  let signal: AbortSignal | null | undefined;
  stubFetch({ status: READY, openStream: requestSignal => {
    signal = requestSignal;
    return new ReadableStream({ start(controller) {
      controller.enqueue(new TextEncoder().encode(`data: ${delta({ content: 'Old response' })}\n\n`));
      signal?.addEventListener('abort', () => controller.error(new DOMException('Aborted', 'AbortError')), { once: true });
    } });
  } });
  await renderPage();
  await waitFor(() => !!container?.querySelector('textarea'), 'composer missing');
  await userEvent.fill(container!.querySelector('textarea')!, 'Old question');
  await userEvent.keyboard('{Enter}');
  await waitFor(() => container?.textContent?.includes('Old response') ?? false, 'stream missing');
  const oldId = store.getState().chat.modelToConversationId['skulk/steward'];
  await act(async () => {
    if (action === 'new') store.dispatch(chatActions.newConversation('skulk/steward'));
    else store.dispatch(chatActions.deleteConversation(oldId));
  });
  await waitFor(() => signal?.aborted === true, 'request not cancelled');
  await waitFor(() => !container?.querySelector('button[aria-label="Cancel generation"]'), 'request did not settle');
  expect(container?.textContent).not.toContain('Old response');
  if (action === 'new') {
    const nextId = store.getState().chat.modelToConversationId['skulk/steward'];
    expect(nextId).not.toBe(oldId);
    expect(store.getState().chat.conversations[nextId].messages).toEqual([]);
    expect(store.getState().chat.conversations[oldId].messages).toHaveLength(1);
    await act(async () => { store.dispatch(chatActions.selectConversation(oldId)); });
    expect(container?.textContent).toContain('Old question');
  } else expect(store.getState().chat.conversations[oldId]).toBeUndefined();
});

it('keeps an unsent drawer draft when selecting the virtual model creates its first conversation', async () => {
  const onChat = vi.fn();
  stubFetch({ status: READY, onChat });
  await renderPage();
  await waitFor(() => !!container?.querySelector('textarea'), 'composer missing');
  await userEvent.fill(container!.querySelector('textarea')!, 'Not submitted yet');
  await act(async () => { store.dispatch(chatActions.selectModel('skulk/steward')); });
  expect(container!.querySelector('textarea')!.value).toBe('Not submitted yet');
  expect(onChat).not.toHaveBeenCalled();
});

it.each(['enter', 'button'])('submits the cluster draft once with %s and retains it across presentations', async method => {
  const onChat = vi.fn();
  const onOpen = vi.fn();
  stubFetch({ status: READY, onChat, openStream: signal => new ReadableStream({ start(controller) {
    controller.enqueue(new TextEncoder().encode(`data: ${delta({ content: 'Working on it' })}\n\n`));
    signal?.addEventListener('abort', () => controller.error(new DOMException('Aborted', 'AbortError')), { once: true });
  } }) });
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  const present = async (chat = false) => {
    await act(async () => root?.render(<Provider store={store}><ThemeProvider theme={darkTheme}>
      <StewardControllerProvider><StewardClusterPrompt onOpen={onOpen} />{chat && <StewardChatView />}</StewardControllerProvider>
    </ThemeProvider></Provider>));
  };
  await present();
  const input = container.querySelector('input')!;
  await userEvent.fill(input, 'How is the cluster?');
  await waitFor(() => !container!.querySelector<HTMLButtonElement>('button')!.disabled, 'Steward did not become ready');
  expect(onOpen).not.toHaveBeenCalled();
  expect(onChat).not.toHaveBeenCalled();
  const composition = new KeyboardEvent('keydown', { key: 'Enter', isComposing: true, bubbles: true, cancelable: true });
  input.dispatchEvent(composition);
  expect(composition.defaultPrevented).toBe(true);
  expect(onChat).not.toHaveBeenCalled();
  await present(true);
  expect(container.querySelector('textarea')?.value).toBe('How is the cluster?');
  await present();
  await userEvent.fill(input, '  ');
  await userEvent.keyboard('{Enter}');
  expect(onChat).not.toHaveBeenCalled();
  await userEvent.fill(input, 'How is the cluster?');
  await act(async () => {
    if (method === 'enter') await userEvent.keyboard('{Enter}');
    else await userEvent.click(container!.querySelector('button')!);
  });
  await waitFor(() => onChat.mock.calls.length === 1, 'cluster prompt did not send');
  expect(onOpen).toHaveBeenCalledOnce();
  expect(input.value).toBe('');
  expect(JSON.parse(String(onChat.mock.calls[0][0].body)).messages.at(-1).content).toBe('How is the cluster?');
  await userEvent.fill(input, 'A follow-up draft');
  await userEvent.keyboard('{Enter}');
  expect(onChat).toHaveBeenCalledOnce();
  expect(input.value).toBe('A follow-up draft');
  await present(true);
  expect(container.textContent).toContain('How is the cluster?');
  expect(container.querySelector('textarea')?.value).toBe('A follow-up draft');
});
