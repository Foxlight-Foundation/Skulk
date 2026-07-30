import { useCallback, useRef, useState } from 'react';
import styled from 'styled-components';
import { MdAutoAwesome } from 'react-icons/md';
import { ChatMessages } from '../chat/ChatMessages';
import { ChatForm } from '../chat/ChatForm';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { addToast } from '../../hooks/useToast';
import { useGetStewardStatusQuery } from '../../store/endpoints/steward';
import type { ChatMessage } from '../../types/chat';

/**
 * The steward chat surface: talk to the cluster's resident assistant.
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

export function StewardChatView() {
  const { t } = useSkulkTranslation();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState<string | null>(null);
  const [streamingThinking, setStreamingThinking] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const { data: status, refetch } = useGetStewardStatusQuery(undefined, {
    pollingInterval: 15000,
  });

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

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
          let detail = t('stewardChat.errors.requestFailed', 'The steward request failed');
          try {
            const body: unknown = await res.json();
            const parsed = (body as { detail?: unknown }).detail;
            if (typeof parsed === 'string') detail = parsed;
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
            const delta = parseDelta(payload);
            if (!delta) continue;
            if (delta.reasoning_content) {
              thinking += delta.reasoning_content;
              setStreamingThinking(thinking);
            }
            if (delta.content) {
              reply += delta.content;
              setStreamingContent(reply);
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
      } catch (error: unknown) {
        if (!(error instanceof DOMException && error.name === 'AbortError')) {
          const message =
            error instanceof Error
              ? error.message
              : t('stewardChat.errors.requestFailed', 'The steward request failed');
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
    [messages, isLoading, refetch, t],
  );

  if (!status) {
    // Gate on the first status response so the chat surface cannot render
    // (or accept input) before the page knows the steward's state.
    return (
      <CenterState>
        <CenterTitle>
          <MdAutoAwesome size={16} />
          {t('stewardChat.loading.title', 'Checking steward status')}
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
            'Enable Intelligent Fabric in Settings and the cluster will place its resident steward. You can then ask it about cluster health, models, and diagnostics right here.',
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
          {t('stewardChat.placing.title', 'Steward is being placed')}
        </CenterTitle>
        <CenterBody>
          {t(
            'stewardChat.placing.body',
            'The fabric is placing the steward and preparing its model. This page will become available automatically; the first start may take a few minutes while the model downloads.',
          )}
        </CenterBody>
      </CenterState>
    );
  }

  return (
    <Container>
      {status.steward_model && (
        <ModelTag>
          {t('stewardChat.servedBy', 'steward: {model}', {
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
                'The steward investigates before answering: cluster health, why a download failed, what a node is doing, whether things look slow. It observes and advises; it cannot change the cluster.',
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
          placeholder={t('stewardChat.placeholder', 'Ask about the cluster...')}
        />
      </InputArea>
    </Container>
  );
}
