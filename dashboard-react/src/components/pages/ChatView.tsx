import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import styled from 'styled-components';
import { ChatMessages } from '../chat/ChatMessages';
import { ChatForm } from '../chat/ChatForm';
import type {
  AudioResponseFormat,
  ChatMessage,
  ChatSpeechModelOption,
  ChatUploadedFile,
  ChatVoiceOption,
} from '../../types/chat';
import { modelSupportsTextChat, type ModelInfo } from '../../types/models';
import type { InstanceCardData } from '../layout/InstancePanel';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import { uiActions } from '../../store/slices/uiSlice';
import { chatActions } from '../../store/slices/chatSlice';
import { store } from '../../store';
import { tolgee, useSkulkTranslation } from '../../i18n/tolgee';
import {
  canUseStreamingSpeechPlayback,
  resyncVisibleSpeech,
  SPEECH_PAUSE_MARKER,
  SPEECH_PAUSE_SECONDS,
  SpeechSentenceQueue,
  speechTextFromMarkdown,
  splitCompleteSpeechSentences,
  StreamingSpeechPlayback,
} from '../../audio/streamingSpeechPlayback';
import { CodeNarrator, FenceProbe } from '../../audio/codeNarration';
import {
  createPinnedSpeechVoiceSelector,
  fetchSpeechVoiceCatalog,
} from '../../audio/speechVoiceSelection';
import {
  batchSpeechMaxTokens,
  buildSpeechSynthesisRequest,
  DASHBOARD_SPEECH_SEED,
  speechLanguageForDashboardLocale,
} from '../../audio/speechSynthesisRequest';
import { buildApiMessages, type ApiMessagePayload } from './chatApiPayload';

/* ── Types ────────────────────────────────────────────── */

export interface ChatViewProps {
  /** Ready instances the user can chat with. */
  readyInstances: InstanceCardData[];
  /** Whether this API node currently advertises the realtime STT provider. */
  realtimeTranscriptionAvailable?: boolean;
  className?: string;
}

/* ── AI Summary ───────────────────────────────────────── */

/* ── Styles ───────────────────────────────────────────── */

const Container = styled.div`
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  overflow: hidden;
`;

const MessagesScroll = styled.div`
  flex: 1;
  overflow-y: auto;
  min-height: 0;
`;

const InputArea = styled.div`
  flex-shrink: 0;
  padding: 12px 24px 16px;
`;

const NoModels = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  gap: 12px;
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: ${({ theme }) => theme.fontSizes.sm};
`;

const ModelSelect = styled.select`
  appearance: none;
  background: transparent;
  border: none;
  color: ${({ theme }) => theme.colors.gold};
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  cursor: pointer;
  outline: none;
  padding-right: 4px;

  option {
    background: ${({ theme }) => theme.colors.surface};
    color: ${({ theme }) => theme.colors.text};
  }
`;

const EMPTY_MESSAGES: ChatMessage[] = [];
const INITIAL_STREAM_STALL_TIMEOUT_MS = 10 * 60_000;
const ACTIVE_STREAM_STALL_TIMEOUT_MS = 60_000;
const THINK_TAG_START = '<think>';
const THINK_TAG_END = '</think>';
const GEMMA_THINK_START = '<|channel>thought\n';
const GEMMA_THINK_END = '<channel|>';
// gpt-oss "harmony" channel markers (distinct from Gemma's `<|channel>thought`).
const HARMONY_CHANNEL_MARK = '<|channel|>';
const HARMONY_CONTROL_TOKENS = [
  '<|start|>',
  '<|end|>',
  '<|return|>',
  '<|message|>',
  '<|channel|>',
  '<|call|>',
  '<|constrain|>',
];
const MAX_TOOL_ROUNDS = 4;
const AUDIO_RESPONSE_FORMATS: readonly AudioResponseFormat[] = [
  'mp3',
  'wav',
  'flac',
  'ogg',
  'opus',
  'pcm',
];

type StreamingSpeechResponseSession = {
  playback: StreamingSpeechPlayback;
  useEncodedFallback: boolean;
};
const GPT_OSS_BROWSER_TOOLS = {
  web_search: {
    type: 'function',
    function: {
      name: 'web_search',
      description: 'Search the public web and return structured results with titles, URLs, and snippets.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Natural-language search query.',
          },
          top_k: {
            type: 'integer',
            description: 'Maximum number of results to return.',
            minimum: 1,
            maximum: 10,
          },
        },
        required: ['query'],
        additionalProperties: false,
      },
    },
  },
  open_url: {
    type: 'function',
    function: {
      name: 'open_url',
      description: 'Open one HTTP or HTTPS URL, follow redirects, and inspect page metadata such as title and content type.',
      parameters: {
        type: 'object',
        properties: {
          url: {
            type: 'string',
            description: 'HTTP or HTTPS URL to inspect.',
          },
        },
        required: ['url'],
        additionalProperties: false,
      },
    },
  },
  extract_page: {
    type: 'function',
    function: {
      name: 'extract_page',
      description: 'Fetch one HTTP or HTTPS URL and return bounded readable text extracted from the page body.',
      parameters: {
        type: 'object',
        properties: {
          url: {
            type: 'string',
            description: 'HTTP or HTTPS URL to extract readable text from.',
          },
          max_chars: {
            type: 'integer',
            description: 'Maximum number of characters of extracted text to return.',
            minimum: 500,
            maximum: 50000,
          },
        },
        required: ['url'],
        additionalProperties: false,
      },
    },
  },
} as const;

type BuiltinBrowserToolName = keyof typeof GPT_OSS_BROWSER_TOOLS;

function isAudioResponseFormat(value: string | null | undefined): value is AudioResponseFormat {
  return AUDIO_RESPONSE_FORMATS.includes(value as AudioResponseFormat);
}

function shortModelLabel(modelId: string): string {
  const parts = modelId.split('/');
  return parts[parts.length - 1] || modelId;
}

function speechModelOption(modelId: string, model: ModelInfo | undefined): ChatSpeechModelOption {
  const resolved = model?.resolved_capabilities;
  const responseFormats = (
    resolved?.audio_response_formats
      ?? model?.audio?.response_formats
      ?? []
  ).filter(isAudioResponseFormat);
  const resolvedDefault = resolved?.default_audio_response_format ?? null;
  const cardDefault = model?.audio?.default_response_format ?? null;
  const defaultResponseFormat: AudioResponseFormat = isAudioResponseFormat(resolvedDefault)
    ? resolvedDefault
    : isAudioResponseFormat(cardDefault)
      ? cardDefault
      : responseFormats[0] ?? 'mp3';
  const formats = responseFormats.length > 0 ? responseFormats : [defaultResponseFormat];
  return {
    modelId,
    label: shortModelLabel(modelId),
    defaultResponseFormat,
    responseFormats: formats,
    supportsVoiceListing: model?.audio?.supports_voice_listing ?? false,
    defaultVoice: model?.audio?.default_voice ?? null,
    supportsStreaming: model?.audio?.supports_streaming ?? false,
    supportsReferenceAudio: model?.audio?.supports_reference_audio ?? false,
    supportsRealtime: Boolean(
      resolved?.supports_realtime_audio
        && model?.audio?.supports_streaming
        && model.audio.supports_realtime,
    ),
  };
}

async function responseErrorMessage(response: Response): Promise<string> {
  const body = await response.text().catch(() => '');
  if (!body) return `HTTP ${response.status}`;
  try {
    const parsed = JSON.parse(body) as { detail?: unknown; error?: { message?: unknown } };
    if (typeof parsed.detail === 'string') return parsed.detail;
    if (typeof parsed.error?.message === 'string') return parsed.error.message;
  } catch {
    // Fall through to the raw response body.
  }
  return body;
}

interface StreamToolCall {
  id: string;
  type?: string;
  function?: {
    name?: string;
    arguments?: string;
  };
}

interface WebSearchToolResponse {
  query: string;
  provider: string;
  results: Array<{
    title: string;
    url: string;
    snippet: string;
  }>;
}

interface OpenUrlToolResponse {
  url: string;
  final_url: string;
  title?: string | null;
  status_code: number;
  content_type?: string | null;
  provider: string;
}

interface ExtractPageToolResponse {
  url: string;
  final_url: string;
  title?: string | null;
  text: string;
  truncated: boolean;
  provider: string;
}

// Split a gpt-oss harmony stream/string into final content vs analysis thinking,
// stripping every control marker. Mirrors the server-side parser so already-stored
// conversations (captured before output channel-parsing) render cleanly, and the
// raw scaffolding never leaks into a message bubble.
function splitHarmonyChannels(raw: string): { content: string; thinking: string } {
  let content = '';
  let thinking = '';
  let inBody = true; // start in content passthrough (text before any channel marker)
  let channel: string | null = null;
  let header = '';
  let i = 0;

  while (i < raw.length) {
    let matched: string | null = null;
    for (const token of HARMONY_CONTROL_TOKENS) {
      if (raw.startsWith(token, i)) {
        matched = token;
        break;
      }
    }
    if (matched) {
      if (matched === '<|channel|>') {
        inBody = false;
        header = '';
      } else if (matched === '<|message|>') {
        const name = header.trim().split(/\s+/)[0];
        channel = name && name.length > 0 ? name : null;
        header = '';
        inBody = true;
      } else if (matched !== '<|constrain|>') {
        // <|end|> / <|start|> / <|return|> / <|call|>: end of body.
        inBody = false;
        channel = null;
        header = '';
      }
      i += matched.length;
      continue;
    }
    const char = raw[i];
    if (inBody) {
      if (channel === 'analysis') thinking += char;
      else content += char;
    } else {
      header += char;
    }
    i += 1;
  }

  return { content, thinking };
}

function splitReasoningDecoratedContent(raw: string): { content: string; thinking: string } {
  if (raw.includes(HARMONY_CHANNEL_MARK)) {
    return splitHarmonyChannels(raw);
  }
  let content = '';
  let thinking = '';
  let i = 0;
  let activeEndTag: string | null = null;

  while (i < raw.length) {
    if (activeEndTag === null) {
      if (raw.startsWith(THINK_TAG_START, i)) {
        activeEndTag = THINK_TAG_END;
        i += THINK_TAG_START.length;
        continue;
      }
      if (raw.startsWith(GEMMA_THINK_START, i)) {
        activeEndTag = GEMMA_THINK_END;
        i += GEMMA_THINK_START.length;
        continue;
      }
      if (raw.startsWith(THINK_TAG_END, i)) {
        i += THINK_TAG_END.length;
        continue;
      }
      if (raw.startsWith(GEMMA_THINK_END, i)) {
        i += GEMMA_THINK_END.length;
        continue;
      }
      content += raw[i];
      i++;
      continue;
    }

    if (raw.startsWith(activeEndTag, i)) {
      i += activeEndTag.length;
      activeEndTag = null;
      continue;
    }

    thinking += raw[i];
    i++;
  }

  return { content, thinking };
}

function mergeThinkingContent(existing: string, incoming: string): string {
  if (!incoming) return existing;
  if (!existing) return incoming;
  if (incoming.startsWith(existing)) {
    return existing + incoming.slice(existing.length);
  }
  if (existing.includes(incoming)) {
    return existing;
  }
  return existing + incoming;
}

function normalizeToolArguments(rawArguments: string | undefined): { query: string; top_k?: number } {
  if (!rawArguments) {
    throw new Error('Tool call arguments were empty.');
  }
  const parsed = JSON.parse(rawArguments) as { query?: unknown; top_k?: unknown };
  if (typeof parsed.query !== 'string' || parsed.query.trim() === '') {
    throw new Error('Tool call did not provide a valid search query.');
  }
  const topK = typeof parsed.top_k === 'number' && Number.isFinite(parsed.top_k)
    ? Math.max(1, Math.min(10, Math.trunc(parsed.top_k)))
    : undefined;
  return {
    query: parsed.query,
    ...(topK !== undefined ? { top_k: topK } : {}),
  };
}

function normalizeUrlToolArguments(
  rawArguments: string | undefined,
): { url: string } {
  if (!rawArguments) {
    throw new Error('Tool call arguments were empty.');
  }
  const parsed = JSON.parse(rawArguments) as { url?: unknown };
  if (typeof parsed.url !== 'string' || parsed.url.trim() === '') {
    throw new Error('Tool call did not provide a valid URL.');
  }
  return { url: parsed.url.trim() };
}

function normalizeExtractPageArguments(
  rawArguments: string | undefined,
): { url: string; max_chars?: number } {
  if (!rawArguments) {
    throw new Error('Tool call arguments were empty.');
  }
  const parsed = JSON.parse(rawArguments) as { url?: unknown; max_chars?: unknown };
  if (typeof parsed.url !== 'string' || parsed.url.trim() === '') {
    throw new Error('Tool call did not provide a valid URL.');
  }
  const maxChars = typeof parsed.max_chars === 'number' && Number.isFinite(parsed.max_chars)
    ? Math.max(500, Math.min(50000, Math.trunc(parsed.max_chars)))
    : undefined;
  return {
    url: parsed.url.trim(),
    ...(maxChars !== undefined ? { max_chars: maxChars } : {}),
  };
}

function builtinToolDefinitions(toolNames: string[]): ReadonlyArray<(typeof GPT_OSS_BROWSER_TOOLS)[BuiltinBrowserToolName]> {
  return toolNames
    .filter((toolName): toolName is BuiltinBrowserToolName => toolName in GPT_OSS_BROWSER_TOOLS)
    .map((toolName) => GPT_OSS_BROWSER_TOOLS[toolName]);
}

async function executeBuiltinToolCall(toolCall: StreamToolCall): Promise<string> {
  const functionCall = toolCall.function;
  if (!functionCall?.name) {
    throw new Error('Tool call did not include a function name.');
  }

  if (functionCall.name === 'web_search') {
    const payload = normalizeToolArguments(functionCall.arguments);
    const res = await fetch('/v1/tools/web_search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const body = await res.json().catch(() => ({})) as WebSearchToolResponse | { detail?: string };
    if (!res.ok) {
      const detail = 'detail' in body && typeof body.detail === 'string'
        ? body.detail
        : `HTTP ${res.status}`;
      throw new Error(detail);
    }
    return JSON.stringify(body);
  }

  if (functionCall.name === 'open_url') {
    const payload = normalizeUrlToolArguments(functionCall.arguments);
    const res = await fetch('/v1/tools/open_url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const body = await res.json().catch(() => ({})) as OpenUrlToolResponse | { detail?: string };
    if (!res.ok) {
      const detail = 'detail' in body && typeof body.detail === 'string'
        ? body.detail
        : `HTTP ${res.status}`;
      throw new Error(detail);
    }
    return JSON.stringify(body);
  }

  if (functionCall.name === 'extract_page') {
    const payload = normalizeExtractPageArguments(functionCall.arguments);
    const res = await fetch('/v1/tools/extract_page', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const body = await res.json().catch(() => ({})) as ExtractPageToolResponse | { detail?: string };
    if (!res.ok) {
      const detail = 'detail' in body && typeof body.detail === 'string'
        ? body.detail
        : `HTTP ${res.status}`;
      throw new Error(detail);
    }
    return JSON.stringify(body);
  }

  {
    throw new Error(`Unsupported tool call: ${functionCall?.name ?? 'unknown'}`);
  }
}

async function readUploadedImageAsDataUrl(file: ChatUploadedFile): Promise<string> {
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result);
        return;
      }
      reject(new Error(`Failed to read attachment ${file.name} as a data URL.`));
    };
    reader.onerror = () => {
      reject(new Error(`Failed to read attachment ${file.name}.`));
    };

    if (file.file) {
      reader.readAsDataURL(file.file);
      return;
    }

    if (!file.preview) {
      reject(new Error(`Attachment ${file.name} has no file contents.`));
      return;
    }

    // Fallback for any older in-memory attachment shape that only has an object URL.
    fetch(file.preview)
      .then((resp) => {
        if (!resp.ok) {
          throw new Error(`Failed to read attachment preview for ${file.name} (HTTP ${resp.status}).`);
        }
        return resp.blob();
      })
      .then((blob) => reader.readAsDataURL(blob))
      .catch((error: unknown) => {
        reject(error instanceof Error ? error : new Error(`Failed to read attachment ${file.name}.`));
      });
  });
}

/* ── Component ────────────────────────────────────────── */

export function ChatView({
  readyInstances,
  realtimeTranscriptionAvailable = false,
  className,
}: ChatViewProps) {
  const { t } = useSkulkTranslation();
  const speechLanguage = speechLanguageForDashboardLocale(tolgee.getLanguage());
  // Store state
  const selectedModelId = useAppSelector((s) => s.chat.selectedModelId);
  const selectedTranscriptionModelId = useAppSelector((s) => s.chat.selectedTranscriptionModelId);
  const selectedSpeechModelId = useAppSelector((s) => s.chat.selectedSpeechModelId);
  const selectedVoice = useAppSelector((s) => s.chat.selectedVoice);
  const autoSpeakAssistant = useAppSelector((s) => s.chat.autoSpeakAssistant);
  const realtimeVoiceEnabled = useAppSelector((s) => s.chat.realtimeVoiceEnabled);
  const autoSubmitVoice = useAppSelector((s) => s.chat.autoSubmitVoice);
  const activeConversationId = useAppSelector((s) => s.chat.activeConversationId);
  const messages = useAppSelector((s) =>
    s.chat.activeConversationId
      ? s.chat.conversations[s.chat.activeConversationId]?.messages ?? EMPTY_MESSAGES
      : EMPTY_MESSAGES,
  );
  const selectModel = (modelId: string) => dispatch(chatActions.selectModel(modelId));
  const selectTranscriptionModel = (modelId: string | null) =>
    dispatch(chatActions.selectTranscriptionModel(modelId));
  const selectSpeechModel = (modelId: string | null) =>
    dispatch(chatActions.selectSpeechModel(modelId));
  const setSelectedVoice = (voice: string | null) =>
    dispatch(chatActions.setSelectedVoice(voice));
  const setAutoSpeakAssistant = (enabled: boolean) =>
    dispatch(chatActions.setAutoSpeakAssistant(enabled));
  const narrateCodeBlocks = useAppSelector((s) => s.chat.narrateCodeBlocks);
  const setNarrateCodeBlocks = (enabled: boolean) =>
    dispatch(chatActions.setNarrateCodeBlocks(enabled));
  const setRealtimeVoiceEnabled = (enabled: boolean) =>
    dispatch(chatActions.setRealtimeVoiceEnabled(enabled));
  const setAutoSubmitVoice = (enabled: boolean) =>
    dispatch(chatActions.setAutoSubmitVoice(enabled));
  const addMessage = (msg: ChatMessage) => dispatch(chatActions.addMessage(msg));
  const deleteMessageAction = (id: string) => dispatch(chatActions.deleteMessage(id));
  const editMessageAction = (messageId: string, content: string) =>
    dispatch(chatActions.editMessage({ messageId, content }));
  const removeLastAssistantMessages = () => dispatch(chatActions.removeLastAssistantMessages());

  // Local transient state
  const [streamingContent, setStreamingContent] = useState<string | null>(null);
  const [streamingThinking, setStreamingThinking] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [thinkingEnabled, setThinkingEnabled] = useState(false);
  const [ttftMs, setTtftMs] = useState<number | null>(null);
  const [tps, setTps] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const activeCommandIdRef = useRef<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [modelThinkingToggleSupport, setModelThinkingToggleSupport] = useState<Record<string, boolean>>({});
  const [modelImageInputSupport, setModelImageInputSupport] = useState<Record<string, boolean>>({});
  const [modelBuiltinTools, setModelBuiltinTools] = useState<Record<string, string[]>>({});
  const [modelContextLengths, setModelContextLengths] = useState<Record<string, number>>({});
  const [modelCatalogById, setModelCatalogById] = useState<Record<string, ModelInfo>>({});
  const [speechError, setSpeechError] = useState<string | null>(null);
  const [voiceCatalogError, setVoiceCatalogError] = useState<string | null>(null);
  const [voiceOptions, setVoiceOptions] = useState<ChatVoiceOption[]>([]);
  const [isVoiceCatalogLoading, setIsVoiceCatalogLoading] = useState(false);
  const [referenceAudioFile, setReferenceAudioFile] = useState<File | null>(null);
  const [referenceAudioText, setReferenceAudioText] = useState('');
  const [speakingMessageId, setSpeakingMessageId] = useState<string | null>(null);
  const [isAutoSpeaking, setIsAutoSpeaking] = useState(false);
  const audioPlaybackRef = useRef<HTMLAudioElement | null>(null);
  const audioObjectUrlRef = useRef<string | null>(null);
  const streamingPlaybackRef = useRef<StreamingSpeechPlayback | null>(null);
  const speechSentenceQueueRef = useRef<SpeechSentenceQueue | null>(null);

  // Restore scroll position after store hydration + DOM render
  const dispatch = useAppDispatch();
  const chatScrollTop = useAppSelector((s) => s.ui.chatScrollTop);
  const setChatScrollTop = (pos: number) => dispatch(uiActions.setChatScrollTop(pos));
  const scrollRestored = useRef(false);
  useEffect(() => {
    if (scrollRestored.current) return;
    if (chatScrollTop <= 0) {
      // Nothing to restore, but latch anyway: the store hydrates synchronously,
      // so a non-positive value on first run is final. Leaving the guard
      // unlatched lets later chatScrollTop updates (echoes of the user's own
      // scrolling, saved by handleScroll) re-enter this effect and assign a
      // stale scrollTop — which cancels in-flight smooth scrolling and made
      // the scroll-to-bottom button a no-op in fresh sessions.
      scrollRestored.current = true;
      return;
    }

    // Wait for store to hydrate and DOM to render messages
    const tryRestore = () => {
      const el = scrollRef.current;
      if (!el) return;
      // Only restore once the scroll container has enough content
      if (el.scrollHeight > el.clientHeight) {
        scrollRestored.current = true;
        el.scrollTop = chatScrollTop;
      }
    };

    // Poll briefly — store hydration + DOM render may take a few frames
    const attempts = [0, 50, 100, 200, 500];
    const timers = attempts.map((ms) => setTimeout(tryRestore, ms));
    return () => timers.forEach(clearTimeout);
  }, [messages.length, chatScrollTop]);

  // Save scroll position on scroll (throttled to avoid jank)
  const scrollRaf = useRef<number>(0);
  const handleScroll = useCallback(() => {
    cancelAnimationFrame(scrollRaf.current);
    scrollRaf.current = requestAnimationFrame(() => {
      if (scrollRef.current) {
        setChatScrollTop(scrollRef.current.scrollTop);
      }
    });
  }, [setChatScrollTop]);

  // Fetch model capabilities and context lengths
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch('/models');
        if (!res.ok) return;
        const data = await res.json() as { data?: ModelInfo[] };
        const toggleSupport: Record<string, boolean> = {};
        const imageSupport: Record<string, boolean> = {};
        const builtinTools: Record<string, string[]> = {};
        const ctxLens: Record<string, number> = {};
        const catalogById: Record<string, ModelInfo> = {};
        for (const m of data.data ?? []) {
          if (m.id) {
            catalogById[m.id] = m;
            toggleSupport[m.id] = m.resolved_capabilities?.supports_thinking_toggle ?? false;
            imageSupport[m.id] = m.resolved_capabilities?.supports_image_input ?? false;
            builtinTools[m.id] = m.tooling?.builtin_tools
              ?? m.resolved_capabilities?.builtin_tools
              ?? [];
          }
          if (m.id && m.context_length) ctxLens[m.id] = m.context_length;
        }
        setModelThinkingToggleSupport(toggleSupport);
        setModelImageInputSupport(imageSupport);
        setModelBuiltinTools(builtinTools);
        setModelContextLengths(ctxLens);
        setModelCatalogById(catalogById);
      } catch { /* ignore */ }
    })();
  }, []);

  const contextLength = selectedModelId ? modelContextLengths[selectedModelId] ?? 0 : 0;

  const supportsThinking = selectedModelId
    ? (modelThinkingToggleSupport[selectedModelId] ?? false)
    : false;
  const supportsImageAttachments = selectedModelId
    ? (modelImageInputSupport[selectedModelId] ?? false)
    : false;
  const selectedBuiltinTools = selectedModelId
    ? (modelBuiltinTools[selectedModelId] ?? [])
    : [];

  useEffect(() => {
    if (!supportsThinking && thinkingEnabled) {
      setThinkingEnabled(false);
    }
  }, [supportsThinking, thinkingEnabled]);

  const readyRunnableInstances = useMemo(
    () => readyInstances.filter((i) => i.status === 'ready' || i.status === 'running'),
    [readyInstances],
  );

  // Ready models
  const readyModels = useMemo(
    () => readyRunnableInstances.filter((i) => !i.isEmbedding && modelSupportsTextChat(modelCatalogById[i.modelId])),
    [modelCatalogById, readyRunnableInstances],
  );

  const readyTranscriptionModels = useMemo(() => {
    const seen = new Set<string>();
    const options: ChatSpeechModelOption[] = [];
    for (const instance of readyRunnableInstances) {
      const model = modelCatalogById[instance.modelId];
      if (
        !model?.resolved_capabilities?.supports_transcription
        && !model?.resolved_capabilities?.supports_speech_translation
      ) {
        continue;
      }
      if (seen.has(instance.modelId)) continue;
      seen.add(instance.modelId);
      options.push(speechModelOption(instance.modelId, model));
    }
    return options;
  }, [modelCatalogById, readyRunnableInstances]);

  const readySpeechModels = useMemo(() => {
    const seen = new Set<string>();
    const options: ChatSpeechModelOption[] = [];
    for (const instance of readyRunnableInstances) {
      const model = modelCatalogById[instance.modelId];
      if (!model?.resolved_capabilities?.supports_speech_synthesis) {
        continue;
      }
      if (seen.has(instance.modelId)) continue;
      seen.add(instance.modelId);
      options.push(speechModelOption(instance.modelId, model));
    }
    return options;
  }, [modelCatalogById, readyRunnableInstances]);

  // Auto-select first ready model if none selected
  useEffect(() => {
    if (
      readyModels.length > 0
      && (!selectedModelId || !readyModels.some((model) => model.modelId === selectedModelId))
    ) {
      selectModel(readyModels[0].modelId);
    }
  }, [selectedModelId, readyModels, selectModel]);

  useEffect(() => {
    if (readyTranscriptionModels.length === 0) {
      if (selectedTranscriptionModelId) selectTranscriptionModel(null);
      return;
    }
    if (
      !selectedTranscriptionModelId
      || !readyTranscriptionModels.some((model) => model.modelId === selectedTranscriptionModelId)
    ) {
      selectTranscriptionModel(readyTranscriptionModels[0].modelId);
    }
  }, [readyTranscriptionModels, selectedTranscriptionModelId, selectTranscriptionModel]);

  useEffect(() => {
    if (readySpeechModels.length === 0) {
      if (selectedSpeechModelId) selectSpeechModel(null);
      return;
    }
    if (
      !selectedSpeechModelId
      || !readySpeechModels.some((model) => model.modelId === selectedSpeechModelId)
    ) {
      selectSpeechModel(readySpeechModels[0].modelId);
    }
  }, [readySpeechModels, selectedSpeechModelId, selectSpeechModel]);

  const selectedLabel = useMemo(() => {
    if (!selectedModelId) return undefined;
    const parts = selectedModelId.split('/');
    return parts[parts.length - 1];
  }, [selectedModelId]);

  const selectedSpeechOption = useMemo(
    () => readySpeechModels.find((model) => model.modelId === selectedSpeechModelId) ?? null,
    [readySpeechModels, selectedSpeechModelId],
  );

  useEffect(() => {
    setReferenceAudioFile(null);
    setReferenceAudioText('');
  }, [selectedSpeechModelId, selectedSpeechOption?.supportsReferenceAudio]);

  useEffect(() => {
    setVoiceOptions([]);
    setVoiceCatalogError(null);
    if (!selectedSpeechModelId || !selectedSpeechOption?.supportsVoiceListing) {
      setIsVoiceCatalogLoading(false);
      return;
    }

    const controller = new AbortController();
    setIsVoiceCatalogLoading(true);
    void fetchSpeechVoiceCatalog(selectedSpeechModelId, controller.signal)
      .then((voices) => {
        if (!controller.signal.aborted) setVoiceOptions(voices);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setVoiceCatalogError(error instanceof Error
          ? error.message
          : t('chat.view.errors.voiceDiscoveryFailed', 'Voice discovery failed.'));
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsVoiceCatalogLoading(false);
      });
    return () => controller.abort();
  }, [selectedSpeechModelId, selectedSpeechOption?.supportsVoiceListing, t]);

  const effectiveSelectedVoice = selectedSpeechOption?.supportsVoiceListing
    && selectedVoice
    && voiceOptions.some((voice) => voice.id === selectedVoice)
    ? selectedVoice
    : null;
  const voiceCatalogReady = !selectedSpeechOption?.supportsVoiceListing
    || voiceOptions.length > 0;

  const canSendMessages = Boolean(
    selectedModelId && readyModels.some((model) => model.modelId === selectedModelId),
  );

  const stopSpeechPlayback = useCallback(() => {
    speechSentenceQueueRef.current?.stop();
    speechSentenceQueueRef.current = null;
    streamingPlaybackRef.current?.stop();
    streamingPlaybackRef.current = null;
    audioPlaybackRef.current?.pause();
    audioPlaybackRef.current = null;
    if (audioObjectUrlRef.current) {
      URL.revokeObjectURL(audioObjectUrlRef.current);
      audioObjectUrlRef.current = null;
    }
    setSpeakingMessageId(null);
    setIsAutoSpeaking(false);
  }, []);

  useEffect(() => {
    if (!autoSpeakAssistant) stopSpeechPlayback();
  }, [autoSpeakAssistant, stopSpeechPlayback]);

  const playSpeechSegment = useCallback(async (
    text: string,
    messageId: string | null,
    signal: AbortSignal,
    voice: string | null,
    responseSession: StreamingSpeechResponseSession | null = null,
  ) => {
    const input = text.trim();
    if (!input || !selectedSpeechModelId) return;
    if (input === SPEECH_PAUSE_MARKER) {
      // Structural break (a horizontal rule): schedule silence on the shared
      // streaming timeline instead of synthesizing text. Encoded-fallback
      // playback has no shared timeline, so the pause is skipped there and
      // the surrounding sentence boundaries still provide separation.
      if (responseSession && !responseSession.useEncodedFallback) {
        await responseSession.playback.appendSilence(SPEECH_PAUSE_SECONDS, signal);
      }
      return;
    }
    if (selectedSpeechOption?.supportsVoiceListing && !voice) {
      throw new Error(t(
        'chat.view.errors.noDiscoveredVoice',
        'No discovered voice is available for this speech model.',
      ));
    }
    const responseFormat = selectedSpeechOption?.defaultResponseFormat ?? 'mp3';
    const useStreamingPcm = Boolean(
      selectedSpeechOption?.supportsStreaming
      && selectedSpeechOption.responseFormats.includes('pcm')
      && canUseStreamingSpeechPlayback()
      && !responseSession?.useEncodedFallback
    );
    const encodedFallbackFormat = responseFormat !== 'pcm'
      ? responseFormat
      : (
          selectedSpeechOption?.responseFormats.includes('mp3')
            ? 'mp3'
            : selectedSpeechOption?.responseFormats.find((format) => format !== 'pcm')
        );
    if (!useStreamingPcm && !encodedFallbackFormat) {
      throw new Error(t(
        'chat.view.errors.pcmStreamingRequired',
        'This speech model requires Web Audio streaming playback in this browser.',
      ));
    }
    const speechRequest = (format: AudioResponseFormat, stream: boolean) => fetch(
      '/v1/audio/speech',
      buildSpeechSynthesisRequest({
        model: selectedSpeechModelId,
        input,
        language: speechLanguage,
        responseFormat: format,
        stream,
        maxTokens: stream ? null : batchSpeechMaxTokens(input),
        seed: DASHBOARD_SPEECH_SEED,
        voice,
        referenceAudio: referenceAudioFile,
        referenceText: referenceAudioText,
        signal,
      }),
    );
    const initialFormat: AudioResponseFormat = useStreamingPcm
      ? 'pcm'
      : encodedFallbackFormat!;
    let response = await speechRequest(initialFormat, useStreamingPcm);
    let streamingResponse = useStreamingPcm;
    if (response.status === 503 && useStreamingPcm && encodedFallbackFormat) {
      if (responseSession) {
        // Once a response-scoped timeline has PCM queued, drain it before
        // changing playback mechanisms so encoded audio cannot overlap it.
        await responseSession.playback.finish();
        responseSession.useEncodedFallback = true;
      }
      response = await speechRequest(encodedFallbackFormat, false);
      streamingResponse = false;
    }
    if (!response.ok) throw new Error(await responseErrorMessage(response));

    setSpeakingMessageId(messageId);
    setIsAutoSpeaking(messageId === null);
    if (streamingResponse) {
      const playback = responseSession?.playback ?? new StreamingSpeechPlayback();
      streamingPlaybackRef.current = playback;
      try {
        if (responseSession) {
          await playback.append(response, signal);
        } else {
          await playback.play(response, signal);
        }
      } finally {
        if (!responseSession && streamingPlaybackRef.current === playback) {
          streamingPlaybackRef.current = null;
        }
      }
      return;
    }

    const audioBlob = await response.blob();
    const objectUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(objectUrl);
    audioPlaybackRef.current = audio;
    audioObjectUrlRef.current = objectUrl;
    try {
      await new Promise<void>((resolve, reject) => {
        const cleanup = () => signal.removeEventListener('abort', onAbort);
        const onAbort = () => {
          cleanup();
          reject(new DOMException('cancelled', 'AbortError'));
        };
        audio.onended = () => {
          cleanup();
          resolve();
        };
        audio.onerror = () => {
          cleanup();
          reject(new Error(
            t('chat.view.errors.speechPlaybackFailed', 'Speech playback failed.'),
          ));
        };
        signal.addEventListener('abort', onAbort, { once: true });
        void audio.play().catch((error: unknown) => {
          cleanup();
          reject(error);
        });
      });
    } finally {
      audio.pause();
      if (audioPlaybackRef.current === audio) audioPlaybackRef.current = null;
      if (audioObjectUrlRef.current === objectUrl) audioObjectUrlRef.current = null;
      URL.revokeObjectURL(objectUrl);
    }
  }, [
    referenceAudioFile,
    referenceAudioText,
    selectedSpeechModelId,
    selectedSpeechOption,
    speechLanguage,
    t,
  ]);

  const createSpeechSentenceQueue = useCallback((
    messageId: string | null,
    selectVoice: (text: string) => string | null,
  ): SpeechSentenceQueue => {
    const useContinuousStreaming = Boolean(
      selectedSpeechOption?.supportsStreaming
      && selectedSpeechOption.responseFormats.includes('pcm')
      && canUseStreamingSpeechPlayback()
    );
    const responseSession: StreamingSpeechResponseSession | null = useContinuousStreaming
      ? {
          playback: new StreamingSpeechPlayback(),
          useEncodedFallback: false,
        }
      : null;
    if (responseSession) streamingPlaybackRef.current = responseSession.playback;

    const queue = new SpeechSentenceQueue(
      (sentence, signal) => playSpeechSegment(
        sentence,
        messageId,
        signal,
        // The pause sentinel is not prose: keep it away from response-scoped
        // automatic voice selection so its Latin token cannot pin a voice.
        sentence === SPEECH_PAUSE_MARKER ? null : selectVoice(sentence),
        responseSession,
      ),
      (error) => {
        const message = error instanceof Error
          ? error.message
          : t('chat.view.errors.speechSynthesisFailed', 'Speech synthesis failed.');
        setSpeechError(message);
      },
      () => {
        const isCurrentQueue = speechSentenceQueueRef.current === queue;
        if (isCurrentQueue) {
          speechSentenceQueueRef.current = null;
          setSpeakingMessageId(null);
          setIsAutoSpeaking(false);
        }
        if (
          responseSession
          && streamingPlaybackRef.current === responseSession.playback
        ) {
          streamingPlaybackRef.current = null;
        }
      },
      responseSession ? () => responseSession.playback.finish() : undefined,
      responseSession ? () => responseSession.playback.stop() : undefined,
    );
    speechSentenceQueueRef.current = queue;
    setSpeakingMessageId(messageId);
    setIsAutoSpeaking(messageId === null);
    return queue;
  }, [playSpeechSegment, selectedSpeechOption, t]);

  const speakText = useCallback((text: string, messageId: string | null = 'draft') => {
    const input = text.trim();
    if (!input || !selectedSpeechModelId) return;
    setSpeechError(null);
    stopSpeechPlayback();
    const selectVoice = createPinnedSpeechVoiceSelector(
      voiceOptions,
      effectiveSelectedVoice,
      selectedSpeechOption?.defaultVoice ?? null,
    );
    const useSentenceStreaming = Boolean(
      selectedSpeechOption?.supportsStreaming
      && selectedSpeechOption.responseFormats.includes('pcm')
      && canUseStreamingSpeechPlayback()
    );
    let segments = [input];
    if (useSentenceStreaming) {
      const split = splitCompleteSpeechSentences(input);
      segments = [...split.sentences];
      if (split.remainder.trim()) segments.push(split.remainder.trim());
    }
    const queue = createSpeechSentenceQueue(messageId, selectVoice);
    queue.enqueue(segments);
    queue.finish();
  }, [
    createSpeechSentenceQueue,
    effectiveSelectedVoice,
    selectedSpeechModelId,
    selectedSpeechOption?.defaultVoice,
    selectedSpeechOption?.responseFormats,
    selectedSpeechOption?.supportsStreaming,
    stopSpeechPlayback,
    voiceOptions,
  ]);

  const transcribeAudio = useCallback(async (audio: Blob): Promise<string> => {
    if (!selectedTranscriptionModelId) {
      throw new Error(t('chat.view.errors.noTranscriptionModel', 'No transcription model is ready.'));
    }

    setSpeechError(null);
    const formData = new FormData();
    const extension = audio.type.includes('mp4')
      ? 'mp4'
      : audio.type.includes('ogg')
        ? 'ogg'
        : 'webm';
    formData.append('file', audio, `recording.${extension}`);
    formData.append('model', selectedTranscriptionModelId);
    formData.append('response_format', 'json');

    const response = await fetch('/v1/audio/transcriptions', {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      throw new Error(await responseErrorMessage(response));
    }
    const body = await response.json() as { text?: unknown };
    return typeof body.text === 'string' ? body.text : '';
  }, [selectedTranscriptionModelId, t]);

  useEffect(() => stopSpeechPlayback, [stopSpeechPlayback]);

  const handleSend = useCallback(async (text: string, files: ChatUploadedFile[]) => {
    if (!selectedModelId || !canSendMessages || isLoading) return;
    stopSpeechPlayback();

    // Convert image files to base64 data URLs for the API and message history
    const imageAttachments: { dataUrl: string; file: ChatUploadedFile }[] = [];
    for (const f of files) {
      if (f.type.startsWith('image/') && (f.file || f.preview)) {
        const dataUrl = await readUploadedImageAsDataUrl(f);
        imageAttachments.push({ dataUrl, file: f });
      }
    }

    const userMsg: ChatMessage = {
      id: uuidv4(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
      attachments: imageAttachments.length > 0
        ? imageAttachments.map(({ dataUrl, file: f }) => ({
            id: f.id,
            name: f.name,
            type: f.type,
            size: f.size,
            preview: dataUrl,
          }))
        : undefined,
    };

    addMessage(userMsg);
    setIsLoading(true);
    setStreamingContent('');
    setStreamingThinking(null);
    setTtftMs(null);
    setTps(null);

    // Read messages from store (includes the user message we just added)
    const chatState = store.getState().chat;
    const activeConvo = chatState.activeConversationId
      ? chatState.conversations[chatState.activeConversationId]
      : undefined;
    if (!activeConvo) {
      setIsLoading(false);
      setStreamingContent(null);
      return;
    }
    const allMessages = activeConvo.messages;

    const controller = new AbortController();
    abortRef.current = controller;
    activeCommandIdRef.current = null;
    let stallTimer: number | null = null;
    let requestTimedOut = false;
    let lastStallTimeoutMs = INITIAL_STREAM_STALL_TIMEOUT_MS;

    const resetStallTimer = () => {
      if (stallTimer !== null) {
        window.clearTimeout(stallTimer);
      }
      lastStallTimeoutMs = firstTokenTime === null
        ? INITIAL_STREAM_STALL_TIMEOUT_MS
        : ACTIVE_STREAM_STALL_TIMEOUT_MS;
      stallTimer = window.setTimeout(() => {
        requestTimedOut = true;
        controller.abort();
      }, lastStallTimeoutMs);
    };

    const startTime = performance.now();
    let firstTokenTime: number | null = null;
    let tokenCount = 0;
    let finalRawContent = '';
    let fullThinking = '';
    let lastTps: number | undefined;
    let toolLoopLimitHit = false;
    let speechTail = '';
    let speechVisibleContent = '';
    const selectResponseVoice = createPinnedSpeechVoiceSelector(
      voiceOptions,
      effectiveSelectedVoice,
      selectedSpeechOption?.defaultVoice ?? null,
    );
    const sentenceQueue = (
      autoSpeakAssistant
      && selectedSpeechModelId
      && voiceCatalogReady
      && selectedSpeechOption?.supportsStreaming
      && selectedSpeechOption.responseFormats.includes('pcm')
      && canUseStreamingSpeechPlayback()
    )
      ? (() => {
          return createSpeechSentenceQueue(null, selectResponseVoice);
        })()
      : null;
    speechSentenceQueueRef.current = sentenceQueue;
    // Code narration (#769): short spoken interjections while a fenced code
    // block streams. Live generation only (replays never build a narrator),
    // and fillers fire only when the voice has run out of things to say.
    const codeNarrator = sentenceQueue && narrateCodeBlocks
      ? new CodeNarrator({
          openers: [
            t('chat.narration.opener1', "Let's write this out."),
            t('chat.narration.opener2', 'Time for some code.'),
            t('chat.narration.opener3', 'Writing it up now.'),
          ],
          fillers: [
            t('chat.narration.filler1', 'Okay, next part.'),
            t('chat.narration.filler2', 'Very good, now this.'),
            t('chat.narration.filler3', 'Still writing.'),
            t('chat.narration.filler4', 'Almost there.'),
            t('chat.narration.filler5', 'And a little more.'),
          ],
          closers: [
            t('chat.narration.closer1', 'Done!'),
            t('chat.narration.closer2', 'And that does it.'),
          ],
        })
      : null;
    const fenceProbe = codeNarrator ? new FenceProbe() : null;
    // Two-phase narration around each delta's enqueue: a finished block's
    // closer settles BEFORE the prose that follows it, while an opener for
    // a fence the delta introduces plays AFTER the introductory sentence.
    // Only sentences that will actually be spoken count as prose (completed
    // fences and rules produce no speech and must not defeat the debounce).
    const narrateAround = (sentences: readonly string[]) => {
      if (!codeNarrator || !sentenceQueue || !fenceProbe) {
        sentenceQueue?.enqueue(sentences);
        return;
      }
      const proseFollowing = sentences.some(
        (sentence) => speechTextFromMarkdown(sentence).length > 0,
      );
      const closer = codeNarrator.settlePendingClose({
        fenceNowOpen: fenceProbe.isOpen(),
        proseFollowing,
      });
      if (closer) sentenceQueue.enqueue([closer]);
      sentenceQueue.enqueue(sentences);
      const utterance = codeNarrator.observe({
        fenceOpen: fenceProbe.isOpen(),
        queueStarved: sentenceQueue.isStarved(),
        bufferedSeconds: streamingPlaybackRef.current?.bufferedSeconds() ?? 0,
      });
      if (utterance) sentenceQueue.enqueue([utterance]);
    };

    try {
      const apiMessages: ApiMessagePayload[] = buildApiMessages(allMessages);
      const requestTools = selectedBuiltinTools.length > 0
        ? builtinToolDefinitions(selectedBuiltinTools)
        : undefined;

      for (let toolRound = 0; toolRound < MAX_TOOL_ROUNDS; toolRound++) {
        resetStallTimer();
        speechVisibleContent = '';
        speechTail = '';
        const roundSpeechSentences: string[] = [];

        let iterationRawContent = '';
        let iterationThinking = '';
        const iterationToolCalls: StreamToolCall[] = [];

        const res = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: selectedModelId,
            messages: apiMessages,
            stream: true,
            ...(requestTools ? { tools: requestTools } : {}),
            ...(supportsThinking ? { enable_thinking: thinkingEnabled } : {}),
          }),
          signal: controller.signal,
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error((err as Record<string, string>).detail ?? `HTTP ${res.status}`);
        }

        const reader = res.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) throw new Error('No response body');

        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          resetStallTimer();
          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split('\n');
          buffer = lines.pop() ?? '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith(': command_id ')) {
              activeCommandIdRef.current = trimmed.slice(': command_id '.length).trim();
              continue;
            }
            if (!trimmed || trimmed.startsWith(':')) continue;
            if (!trimmed.startsWith('data: ')) continue;
            const data = trimmed.slice(6);
            if (data === '[DONE]') continue;

            try {
              const parsed = JSON.parse(data) as {
                id?: string;
                choices?: Array<{
                  delta?: {
                    content?: string;
                    reasoning_content?: string;
                    tool_calls?: StreamToolCall[];
                  };
                }>;
              };
              if (parsed.id) {
                activeCommandIdRef.current = parsed.id;
              }
              const delta = parsed.choices?.[0]?.delta;
              const hasToken = delta?.content || delta?.reasoning_content;

              if (hasToken && firstTokenTime === null) {
                firstTokenTime = performance.now();
                setTtftMs(firstTokenTime - startTime);
              }

              if (delta?.reasoning_content) {
                iterationThinking += delta.reasoning_content;
                const combinedThinking = mergeThinkingContent(fullThinking, iterationThinking);
                setStreamingThinking(combinedThinking);
              }

              if (delta?.content) {
                iterationRawContent += delta.content;
                const separated = splitReasoningDecoratedContent(iterationRawContent);
                if (separated.thinking) {
                  iterationThinking = mergeThinkingContent(iterationThinking, separated.thinking);
                  const combinedThinking = mergeThinkingContent(fullThinking, iterationThinking);
                  setStreamingThinking(combinedThinking);
                }
                setStreamingContent(separated.content || null);
                if (sentenceQueue && separated.content.startsWith(speechVisibleContent)) {
                  const visibleDelta = separated.content.slice(speechVisibleContent.length);
                  speechVisibleContent = separated.content;
                  const split = splitCompleteSpeechSentences(speechTail + visibleDelta);
                  speechTail = split.remainder;
                  if (requestTools) {
                    roundSpeechSentences.push(...split.sentences);
                  } else {
                    fenceProbe?.feed(visibleDelta);
                    narrateAround(split.sentences);
                  }
                } else if (sentenceQueue) {
                  const split = resyncVisibleSpeech(
                    speechVisibleContent,
                    speechTail,
                    separated.content,
                  );
                  speechVisibleContent = separated.content;
                  speechTail = split.remainder;
                  if (requestTools) {
                    roundSpeechSentences.push(...split.sentences);
                  } else {
                    fenceProbe?.reset();
                    fenceProbe?.feed(separated.content);
                    narrateAround(split.sentences);
                  }
                }
              }

              if (delta?.tool_calls?.length) {
                iterationToolCalls.push(...delta.tool_calls);
              }

              if (hasToken) {
                tokenCount++;
                if (firstTokenTime !== null && tokenCount > 1) {
                  const elapsed = (performance.now() - firstTokenTime) / 1000;
                  if (elapsed > 0) {
                    lastTps = tokenCount / elapsed;
                    setTps(lastTps);
                  }
                }
              }
            } catch {
              // skip malformed JSON
            }
          }
        }

        const separatedContent = splitReasoningDecoratedContent(iterationRawContent);
        if (separatedContent.thinking) {
          iterationThinking = mergeThinkingContent(iterationThinking, separatedContent.thinking);
        }

        if (iterationToolCalls.length === 0) {
          if (sentenceQueue && requestTools) {
            sentenceQueue.enqueue(roundSpeechSentences);
          }
          fullThinking = mergeThinkingContent(fullThinking, iterationThinking);
          finalRawContent = separatedContent.content;
          break;
        }

        fullThinking = mergeThinkingContent(fullThinking, iterationThinking);

        for (const toolCall of iterationToolCalls) {
          const toolName = toolCall.function?.name ?? 'unknown';
          const toolCallId = toolCall.id || uuidv4();
          let toolOutput: string;

          try {
            toolOutput = await executeBuiltinToolCall(toolCall);
          } catch (error) {
            const message = error instanceof Error
              ? error.message
              : t('chat.view.errors.toolExecutionFailed', 'Tool execution failed.');
            toolOutput = JSON.stringify({ error: message });
          }

          apiMessages.push({
            role: 'assistant',
            content: '',
            tool_calls: [{
              id: toolCallId,
              type: toolCall.type ?? 'function',
              function: {
                name: toolName,
                arguments: toolCall.function?.arguments ?? '{}',
              },
            }],
          });
          apiMessages.push({
            role: 'tool',
            tool_call_id: toolCallId,
            content: toolOutput,
          });
        }

        setStreamingContent(null);
        if (toolRound === MAX_TOOL_ROUNDS - 1) {
          toolLoopLimitHit = true;
        }
      }

      if (!finalRawContent && toolLoopLimitHit) {
        finalRawContent = t(
          'chat.view.errors.toolLoopLimit',
          'Error: web search tool loop exceeded the safety limit.',
        );
      }
    } catch (err) {
      sentenceQueue?.stop();
      if (speechSentenceQueueRef.current === sentenceQueue) {
        speechSentenceQueueRef.current = null;
      }
      setIsAutoSpeaking(false);
      if ((err as Error).name === 'AbortError') {
        // User cancelled
        if (requestTimedOut) {
          finalRawContent = t(
            'chat.view.errors.generationStalled',
            'Error: generation stalled for more than {seconds} seconds.',
            { seconds: Math.round(lastStallTimeoutMs / 1000) },
          );
        }
      } else {
        finalRawContent = finalRawContent || t(
          'chat.view.errors.generic',
          'Error: {message}',
          { message: (err as Error).message },
        );
      }
    } finally {
      if (stallTimer !== null) {
        window.clearTimeout(stallTimer);
      }
    }

    // Finalize assistant message
    const separatedContent = splitReasoningDecoratedContent(finalRawContent);
    if (separatedContent.thinking) {
      fullThinking = mergeThinkingContent(fullThinking, separatedContent.thinking);
    }
    const finalAssistantContent = separatedContent.content.trim();

    if (finalAssistantContent || fullThinking) {
      const assistantMsg: ChatMessage = {
        id: uuidv4(),
        role: 'assistant',
        content: finalAssistantContent,
        timestamp: Date.now(),
        ttftMs: firstTokenTime ? firstTokenTime - startTime : undefined,
        tps: lastTps,
        thinkingContent: fullThinking || undefined,
      };

      addMessage(assistantMsg);
      if (sentenceQueue) {
        // Settle any pending code closer ahead of the trailing prose so the
        // acknowledgement precedes the final explanation.
        if (codeNarrator && speechTail.trim()) {
          const closer = codeNarrator.settlePendingClose({
            fenceNowOpen: fenceProbe?.isOpen() ?? false,
            proseFollowing: speechTextFromMarkdown(speechTail).length > 0,
          });
          if (closer) sentenceQueue.enqueue([closer]);
        }
        if (speechTail.trim()) sentenceQueue.enqueue([speechTail.trim()]);
        speechTail = '';
      } else if (autoSpeakAssistant && selectedSpeechModelId && finalAssistantContent) {
        void speakText(finalAssistantContent, assistantMsg.id);
      }
    }
    if (codeNarrator && sentenceQueue) {
      const closer = codeNarrator.finish();
      if (closer) sentenceQueue.enqueue([closer]);
    }
    sentenceQueue?.finish();
    setStreamingContent(null);
    setStreamingThinking(null);
    setIsLoading(false);
    abortRef.current = null;
    activeCommandIdRef.current = null;

  }, [
    addMessage,
    autoSpeakAssistant,
    canSendMessages,
    createSpeechSentenceQueue,
    effectiveSelectedVoice,
    isLoading,
    selectedBuiltinTools,
    selectedModelId,
    selectedSpeechModelId,
    selectedSpeechOption,
    speakText,
    stopSpeechPlayback,
    supportsThinking,
    thinkingEnabled,
    t,
    voiceCatalogReady,
    voiceOptions,
  ]);

  const handleCancel = useCallback(async () => {
    const commandId = activeCommandIdRef.current;
    activeCommandIdRef.current = null;

    abortRef.current?.abort();
    stopSpeechPlayback();

    if (!commandId) return;

    try {
      await fetch(`/v1/cancel/${encodeURIComponent(commandId)}`, {
        method: 'POST',
      });
    } catch {
      // The UI should still stop immediately even if the cancel request races with teardown.
    }
  }, [stopSpeechPlayback]);

  const handleDelete = useCallback((id: string) => {
    deleteMessageAction(id);
  }, [deleteMessageAction]);

  const handleEdit = useCallback((id: string, content: string) => {
    editMessageAction(id, content);
  }, [editMessageAction]);

  const handleRegenerate = useCallback(() => {
    removeLastAssistantMessages();
    // Re-send last user message on next tick after store updates
    setTimeout(() => {
      const chatState = store.getState().chat;
      const convo = chatState.activeConversationId
        ? chatState.conversations[chatState.activeConversationId]
        : undefined;
      if (!convo) return;
      const lastUser = convo.messages.filter((m) => m.role === 'user').pop();
      if (lastUser) {
        handleSend(lastUser.content, []);
      }
    }, 50);
  }, [handleSend, removeLastAssistantMessages]);

  // Thinking expansion state — persisted per conversation in session store
  const expandedThinkingMap = useAppSelector((s) => s.ui.expandedThinking);
  const setExpandedThinking = (conversationId: string, messageIds: string[]) =>
    dispatch(uiActions.setExpandedThinking({ conversationId, messageIds }));
  const expandedThinkingIds = useMemo(
    () => new Set(activeConversationId ? expandedThinkingMap[activeConversationId] ?? [] : []),
    [expandedThinkingMap, activeConversationId],
  );
  const handleToggleThinking = useCallback((messageId: string) => {
    if (!activeConversationId) return;
    const current = expandedThinkingMap[activeConversationId] ?? [];
    const next = current.includes(messageId)
      ? current.filter((id) => id !== messageId)
      : [...current, messageId];
    setExpandedThinking(activeConversationId, next);
  }, [activeConversationId, expandedThinkingMap, setExpandedThinking]);

  const handleRealtimeTranscript = useCallback((text: string, final: boolean) => {
    if (!final || !autoSubmitVoice || !canSendMessages || !text.trim()) return;
    void handleSend(text.trim(), []);
  }, [autoSubmitVoice, canSendMessages, handleSend]);

  if (readyModels.length === 0 && readyTranscriptionModels.length === 0 && readySpeechModels.length === 0) {
    return (
      <NoModels>
        {t(
          'chat.view.noReadyModels',
          'No models are ready. Launch a model from the Model Store to start chatting.',
        )}
      </NoModels>
    );
  }

  const modelSelector = readyModels.length > 1 ? (
    <ModelSelect
      value={selectedModelId ?? ''}
      onChange={(e) => selectModel(e.target.value)}
      aria-label={t('chat.view.selectModel', 'Select chat model')}
    >
      {readyModels.map((m) => (
        <option key={m.instanceId} value={m.modelId}>
          {m.modelId.split('/').pop()}
        </option>
      ))}
    </ModelSelect>
  ) : undefined;

  return (
    <Container className={className}>
      <MessagesScroll ref={scrollRef} onScroll={handleScroll}>
        <ChatMessages
          messages={messages}
          streamingContent={streamingContent}
          streamingThinking={streamingThinking}
          isLoading={isLoading}
          onDelete={handleDelete}
          onEdit={handleEdit}
          onRegenerate={handleRegenerate}
          isSpeechPlaybackAvailable={readySpeechModels.length > 0}
          speakingMessageId={speakingMessageId}
          onSpeakMessage={(messageId, content) => { void speakText(content, messageId); }}
          onStopSpeaking={stopSpeechPlayback}
          expandedThinkingIds={expandedThinkingIds}
          onToggleThinking={handleToggleThinking}
        />
      </MessagesScroll>
      <InputArea>
        <ChatForm
          onSend={handleSend}
          onCancel={handleCancel}
          isLoading={isLoading}
          modelLabel={canSendMessages ? selectedLabel : undefined}
          modelSelector={modelSelector}
          ttftMs={ttftMs}
          tps={tps}
          contextLength={canSendMessages ? contextLength : 0}
          showThinkingToggle={canSendMessages && supportsThinking}
          thinkingEnabled={thinkingEnabled}
          onToggleThinking={() => setThinkingEnabled((v) => !v)}
          supportsImageAttachments={canSendMessages && supportsImageAttachments}
          canSendMessages={canSendMessages}
          transcriptionModels={readyTranscriptionModels}
          realtimeTranscriptionAvailable={realtimeTranscriptionAvailable}
          speechModels={readySpeechModels}
          selectedTranscriptionModelId={selectedTranscriptionModelId}
          selectedSpeechModelId={selectedSpeechModelId}
          selectedVoice={selectedVoice}
          voiceOptions={voiceOptions}
          isVoiceCatalogLoading={isVoiceCatalogLoading}
          referenceAudioFile={referenceAudioFile}
          referenceAudioText={referenceAudioText}
          autoSpeakAssistant={autoSpeakAssistant}
          realtimeVoiceEnabled={realtimeVoiceEnabled}
          autoSubmitVoice={autoSubmitVoice}
          isSpeaking={speakingMessageId !== null || isAutoSpeaking}
          voiceError={voiceCatalogError ?? speechError}
          onSelectTranscriptionModel={selectTranscriptionModel}
          onSelectSpeechModel={selectSpeechModel}
          onSelectedVoiceChange={setSelectedVoice}
          onReferenceAudioChange={setReferenceAudioFile}
          onReferenceAudioTextChange={setReferenceAudioText}
          onAutoSpeakAssistantChange={setAutoSpeakAssistant}
          narrateCodeBlocks={narrateCodeBlocks}
          onNarrateCodeBlocksChange={setNarrateCodeBlocks}
          onRealtimeVoiceEnabledChange={setRealtimeVoiceEnabled}
          onAutoSubmitVoiceChange={setAutoSubmitVoice}
          onRealtimeTranscript={handleRealtimeTranscript}
          onTranscribeAudio={transcribeAudio}
          onSpeakText={(text) => { void speakText(text, 'draft'); }}
          onStopSpeaking={stopSpeechPlayback}
          placeholder={canSendMessages && selectedModelId
            ? t('chat.view.messagePlaceholder', 'Message {model}...', {
                model: selectedLabel ?? t('chat.view.selectedModel', 'selected model'),
              })
            : t('chat.view.voiceDraftPlaceholder', 'Type text for speech or transcription')}
        />
      </InputArea>
    </Container>
  );
}
