import { useCallback, useEffect, useRef, useState } from 'react';
import styled from 'styled-components';
import { MdAutoAwesome } from 'react-icons/md';
import { ChatMessages } from '../chat/ChatMessages';
import { ChatForm } from '../chat/ChatForm';
import type { InstanceCardData } from '../layout/InstancePanel';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { tolgee } from '../../i18n/tolgee';
import { addToast } from '../../hooks/useToast';
import { useGetStewardStatusQuery } from '../../store/endpoints/steward';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import { chatActions } from '../../store/slices/chatSlice';
import type { ChatMessage, ChatSpeechModelOption, ChatVoiceOption } from '../../types/chat';
import type { ModelInfo } from '../../types/models';
import {
  canUseStreamingSpeechPlayback,
  SPEECH_PAUSE_MARKER,
  SPEECH_PAUSE_SECONDS,
  SpeechSentenceQueue,
  splitCompleteSpeechSentences,
  StreamingSpeechPlayback,
} from '../../audio/streamingSpeechPlayback';
import { fetchSpeechVoiceCatalog } from '../../audio/speechVoiceSelection';
import {
  speechLanguageForDashboardLocale,
} from '../../audio/speechSynthesisRequest';
import { buildSkulkSpeechSynthesisRequest } from '../../audio/fabricSpeechRequest';
import { discoverSkulkSpeechSelection } from '../../audio/fabricSpeechDiscovery';

/**
 * Skulk's fabric chat surface: talk to the intelligent fabric itself.
 *
 * Conversation rides the standard streaming chat-completions endpoint with
 * the reserved virtual model id: tool steps arrive live as
 * `reasoning_content` deltas while the steward investigates (rendered
 * through the shared thinking affordances), and the answer follows as
 * `content`. The page holds its placing state until the steward is actually
 * ready to serve, not merely placed.
 */

/** Reserved chat-completions model id selecting model-plus-harness. */
export const STEWARD_MODEL_ID = 'skulk/steward';
/** Inputs from current runtime truth needed to expose fabric speech safely. */
export interface StewardChatViewProps {
  readyInstances?: InstanceCardData[];
}

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

const CenterState = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: ${({ theme }) => theme.colors.textSecondary};
  padding: 24px;
  text-align: center;
`;

const CenterTitle = styled.div`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: ${({ theme }) => theme.colors.textMuted};
  display: flex;
  align-items: center;
  gap: 8px;
`;

const CenterBody = styled.div`
  max-width: 460px;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  line-height: 1.5;
`;

const ModelTag = styled.div`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  padding: 4px 24px 0;
  text-align: center;
`;

interface StreamDelta {
  content?: string;
  reasoning_content?: string;
}

/** Extract a mid-stream error envelope's message, if this payload is one. */
function parseStreamError(payload: string): string | null {
  try {
    const parsed: unknown = JSON.parse(payload);
    if (typeof parsed !== 'object' || parsed === null) return null;
    const error = (parsed as { error?: unknown }).error;
    if (typeof error === 'string') return error;
    if (typeof error === 'object' && error !== null) {
      const message = (error as { message?: unknown }).message;
      if (typeof message === 'string') return message;
    }
    return null;
  } catch {
    return null;
  }
}

/** Parse one SSE `data:` payload into its delta, ignoring non-JSON lines. */
function parseDelta(payload: string): StreamDelta | null {
  try {
    const parsed: unknown = JSON.parse(payload);
    if (typeof parsed !== 'object' || parsed === null) return null;
    const choices = (parsed as { choices?: unknown }).choices;
    if (!Array.isArray(choices) || choices.length === 0) return null;
    const delta = (choices[0] as { delta?: unknown }).delta;
    if (typeof delta !== 'object' || delta === null) return null;
    return delta as StreamDelta;
  } catch {
    return null;
  }
}

export function StewardChatView({ readyInstances = [] }: StewardChatViewProps) {
  const { t } = useSkulkTranslation();
  const dispatch = useAppDispatch();
  const autoSpeakAssistant = useAppSelector((state) => state.chat.autoSpeakAssistant);
  const speechLanguage = speechLanguageForDashboardLocale(tolgee.getLanguage());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState<string | null>(null);
  const [streamingThinking, setStreamingThinking] = useState<string | null>(null);
  const [speechModel, setSpeechModel] = useState<ChatSpeechModelOption | null>(null);
  const [speechVoice, setSpeechVoice] = useState<ChatVoiceOption | null>(null);
  const [speechError, setSpeechError] = useState<string | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const speechQueueRef = useRef<SpeechSentenceQueue | null>(null);
  const speechPlaybackRef = useRef<StreamingSpeechPlayback | null>(null);
  const { data: status, refetch } = useGetStewardStatusQuery(undefined, {
    pollingInterval: 15000,
  });

  const stopSpeechPlayback = useCallback(() => {
    speechQueueRef.current?.stop();
    speechQueueRef.current = null;
    speechPlaybackRef.current?.stop();
    speechPlaybackRef.current = null;
    setIsSpeaking(false);
  }, []);

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
    stopSpeechPlayback();
  }, [stopSpeechPlayback]);

  useEffect(() => {
    const controller = new AbortController();
    setSpeechError(null);
    const readyModelIds = new Set(
      readyInstances
        .filter((instance) => instance.status === 'ready' || instance.status === 'running')
        .map((instance) => instance.modelId),
    );
    void fetch('/models', { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Model discovery failed with HTTP ${response.status}.`);
        return await response.json() as { data?: ModelInfo[] };
      })
      .then(async ({ data = [] }) => {
        const selection = await discoverSkulkSpeechSelection(
          data,
          readyModelIds,
          (modelId) => fetchSpeechVoiceCatalog(modelId, controller.signal),
          canUseStreamingSpeechPlayback(),
        );
        if (controller.signal.aborted) return;
        if (selection) {
          setSpeechModel(selection.model);
          setSpeechVoice(selection.voice);
          return;
        }
        setSpeechModel(null);
        setSpeechVoice(null);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setSpeechModel(null);
        setSpeechVoice(null);
        setSpeechError(
          error instanceof Error
            ? error.message
            : t('stewardChat.errors.voiceDiscoveryFailed', 'Skulk voice discovery failed.'),
        );
      });
    return () => controller.abort();
  }, [readyInstances, t]);

  const createSpeechQueue = useCallback((): SpeechSentenceQueue | null => {
    if (!speechModel || !speechVoice) return null;
    stopSpeechPlayback();
    setSpeechError(null);
    setIsSpeaking(true);
    const playback = new StreamingSpeechPlayback();
    speechPlaybackRef.current = playback;
    const queue = new SpeechSentenceQueue(
      async (text, signal) => {
        if (text === SPEECH_PAUSE_MARKER) {
          await playback.appendSilence(SPEECH_PAUSE_SECONDS, signal);
          return;
        }
        const response = await fetch(
          '/v1/audio/speech',
          buildSkulkSpeechSynthesisRequest(
            speechModel.modelId,
            text,
            speechLanguage,
            signal,
          ),
        );
        if (!response.ok) {
          const detail = await response.text().catch(() => '');
          throw new Error(detail || `Speech synthesis failed with HTTP ${response.status}.`);
        }
        await playback.append(response, signal);
      },
      (error) => {
        setSpeechError(
          error instanceof Error
            ? error.message
            : t('stewardChat.errors.speechFailed', 'Skulk speech playback failed.'),
        );
      },
      () => {
        if (speechQueueRef.current !== queue) return;
        speechQueueRef.current = null;
        if (speechPlaybackRef.current === playback) speechPlaybackRef.current = null;
        setIsSpeaking(false);
      },
      () => playback.finish(),
      () => playback.stop(),
    );
    speechQueueRef.current = queue;
    return queue;
  }, [speechLanguage, speechModel, speechVoice, stopSpeechPlayback, t]);

  // Navigating away must not leave the steward generating for a stream
  // nobody is reading; the server cancels the turn on disconnect.
  useEffect(() => () => {
    abortRef.current?.abort();
    stopSpeechPlayback();
  }, [stopSpeechPlayback]);

  useEffect(() => {
    if (!autoSpeakAssistant) stopSpeechPlayback();
  }, [autoSpeakAssistant, stopSpeechPlayback]);

  const handleSend = useCallback(
    async (content: string) => {
      const trimmed = content.trim();
      if (!trimmed || isLoading) return;
      const userMessage: ChatMessage = {
        id: `steward-u-${Date.now()}`,
        role: 'user',
        content: trimmed,
        timestamp: Date.now(),
      };
      const history = [...messages, userMessage];
      setMessages(history);
      setIsLoading(true);
      setStreamingContent(null);
      setStreamingThinking(null);
      const controller = new AbortController();
      abortRef.current = controller;
      let reply = '';
      let thinking = '';
      let speechTail = '';
      const speechQueue = autoSpeakAssistant ? createSpeechQueue() : null;
      try {
        const res = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
          body: JSON.stringify({
            model: STEWARD_MODEL_ID,
            stream: true,
            messages: history.map(({ role, content: text }) => ({ role, content: text })),
          }),
        });
        if (!res.ok || !res.body) {
          let detail = t('stewardChat.errors.requestFailed', 'Skulk could not answer');
          try {
            const body: unknown = await res.json();
            const parsed = (body as { detail?: unknown }).detail;
            if (typeof parsed === 'string') {
              detail = parsed;
            } else if (typeof parsed === 'object' && parsed !== null) {
              // The not-ready 503 answers with the steward status object
              // plus a human message; surface the message rather than
              // falling back to the generic failure text.
              const message = (parsed as { message?: unknown }).message;
              if (typeof message === 'string') detail = message;
            }
          } catch {
            /* keep fallback */
          }
          throw new Error(detail);
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() ?? '';
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const payload = line.slice(6).trim();
            if (payload === '[DONE]') continue;
            const streamError = parseStreamError(payload);
            if (streamError) throw new Error(streamError);
            const delta = parseDelta(payload);
            if (!delta) continue;
            if (delta.reasoning_content) {
              thinking += delta.reasoning_content;
              setStreamingThinking(thinking);
            }
            if (delta.content) {
              reply += delta.content;
              setStreamingContent(reply);
              if (speechQueue) {
                const split = splitCompleteSpeechSentences(speechTail + delta.content);
                speechQueue.enqueue(split.sentences);
                speechTail = split.remainder;
              }
            }
          }
        }
        if (reply) {
          setMessages((prev) => [
            ...prev,
            {
              id: `steward-a-${Date.now()}`,
              role: 'assistant',
              content: reply,
              timestamp: Date.now(),
              thinkingContent: thinking || undefined,
            },
          ]);
        }
        if (speechQueue) {
          if (speechTail.trim()) speechQueue.enqueue([speechTail.trim()]);
          speechQueue.finish();
        }
      } catch (error: unknown) {
        stopSpeechPlayback();
        if (!(error instanceof DOMException && error.name === 'AbortError')) {
          const message =
            error instanceof Error
              ? error.message
              : t('stewardChat.errors.requestFailed', 'Skulk could not answer');
          addToast({ type: 'error', message });
          void refetch();
        }
      } finally {
        setIsLoading(false);
        setStreamingContent(null);
        setStreamingThinking(null);
        abortRef.current = null;
      }
    },
    [
      autoSpeakAssistant,
      createSpeechQueue,
      messages,
      isLoading,
      refetch,
      stopSpeechPlayback,
      t,
    ],
  );

  const speakDraft = useCallback((text: string) => {
    const queue = createSpeechQueue();
    if (!queue) return;
    const split = splitCompleteSpeechSentences(text);
    queue.enqueue([...split.sentences, ...(split.remainder.trim() ? [split.remainder.trim()] : [])]);
    queue.finish();
  }, [createSpeechQueue]);

  if (!status) {
    // Gate on the first status response so the chat surface cannot render
    // (or accept input) before the page knows the steward's state.
    return (
      <CenterState>
        <CenterTitle>
          <MdAutoAwesome size={16} />
          {t('stewardChat.loading.title', 'Connecting to Skulk')}
        </CenterTitle>
      </CenterState>
    );
  }

  if (!status.enabled) {
    return (
      <CenterState>
        <CenterTitle>
          <MdAutoAwesome size={16} />
          {t('stewardChat.disabled.title', 'Intelligent Fabric is off')}
        </CenterTitle>
        <CenterBody>
          {t(
            'stewardChat.disabled.body',
            'Enable Intelligent Fabric in Settings to make Skulk available here. You can then ask the fabric about its health, models, and diagnostics.',
          )}
        </CenterBody>
      </CenterState>
    );
  }

  if (!status.present || !status.ready) {
    return (
      <CenterState>
        <CenterTitle>
          <MdAutoAwesome size={16} />
          {t('stewardChat.placing.title', 'Skulk is getting ready')}
        </CenterTitle>
        <CenterBody>
          {t(
            'stewardChat.placing.body',
            'The fabric is preparing its resident intelligence. This page will become available automatically; the first start may take a few minutes while the model downloads.',
          )}
        </CenterBody>
        <CenterBody>
          {t('stewardChat.placing.state', 'Status: {state}', { state: status.state })}
        </CenterBody>
      </CenterState>
    );
  }

  return (
    <Container>
      {status.steward_model && (
        <ModelTag>
          {t('stewardChat.servedBy', 'fabric cognition: {model}', {
            model: status.steward_model,
          })}
        </ModelTag>
      )}
      <MessagesScroll>
        {messages.length === 0 && !isLoading ? (
          <CenterState>
            <CenterTitle>
              <MdAutoAwesome size={16} />
              {t('stewardChat.empty.title', 'Ask the cluster')}
            </CenterTitle>
            <CenterBody>
              {t(
                'stewardChat.empty.body',
                'Ask Skulk about its health, why a download failed, what a node is doing, or whether something looks slow. I investigate before answering; this interface observes and advises but cannot change the cluster.',
              )}
            </CenterBody>
          </CenterState>
        ) : (
          <ChatMessages
            messages={messages}
            isLoading={isLoading && !streamingContent && !streamingThinking}
            streamingContent={streamingContent}
            streamingThinking={streamingThinking}
          />
        )}
      </MessagesScroll>
      <InputArea>
        <ChatForm
          onSend={handleSend}
          onCancel={handleCancel}
          isLoading={isLoading}
          canSendMessages
          speechModels={speechModel ? [speechModel] : []}
          selectedSpeechModelId={speechModel?.modelId ?? null}
          selectedVoice={speechVoice?.id ?? null}
          voiceOptions={speechVoice ? [speechVoice] : []}
          fixedSpeechVoiceLabel="Skulk"
          autoSpeakAssistant={autoSpeakAssistant}
          onAutoSpeakAssistantChange={(enabled) => {
            dispatch(chatActions.setAutoSpeakAssistant(enabled));
            if (!enabled) stopSpeechPlayback();
          }}
          onSpeakText={speakDraft}
          onStopSpeaking={stopSpeechPlayback}
          isSpeaking={isSpeaking}
          voiceError={speechError}
          placeholder={t('stewardChat.placeholder', 'Ask about the cluster...')}
        />
      </InputArea>
    </Container>
  );
}
