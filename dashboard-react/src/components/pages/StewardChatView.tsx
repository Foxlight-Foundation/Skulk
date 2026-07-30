import { useCallback, useState } from 'react';
import styled from 'styled-components';
import { MdAutoAwesome } from 'react-icons/md';
import { ChatMessages } from '../chat/ChatMessages';
import { ChatForm } from '../chat/ChatForm';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { addToast } from '../../hooks/useToast';
import {
  useGetStewardStatusQuery,
  useStewardChatMutation,
  type StewardToolStep,
} from '../../store/endpoints/steward';
import type { ChatMessage } from '../../types/chat';

/**
 * The steward chat surface: talk to the cluster's resident assistant.
 *
 * Unlike ChatView this page has no model picker, conversations, or
 * streaming: the steward is a single fabric-maintained placement, each turn
 * is a bounded server-side investigation, and the reply arrives whole. The
 * investigation trace (which tools the steward consulted) is rendered
 * through the existing collapsible thinking block on each reply.
 */

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

function traceToThinking(steps: StewardToolStep[]): string | undefined {
  if (steps.length === 0) return undefined;
  return steps
    .map((step) => {
      const args = Object.keys(step.arguments).length
        ? ` ${JSON.stringify(step.arguments)}`
        : '';
      return `${step.tool}${args}`;
    })
    .join('\n');
}

export function StewardChatView() {
  const { t } = useSkulkTranslation();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const { data: status, refetch } = useGetStewardStatusQuery(undefined, {
    pollingInterval: 15000,
  });
  const [sendChat, { isLoading }] = useStewardChatMutation();

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
      try {
        const response = await sendChat(
          history.map(({ role, content: text }) => ({ role, content: text })),
        ).unwrap();
        setMessages((prev) => [
          ...prev,
          {
            id: `steward-a-${Date.now()}`,
            role: 'assistant',
            content: response.reply,
            timestamp: Date.now(),
            thinkingContent: traceToThinking(response.steps),
          },
        ]);
      } catch (error: unknown) {
        // A 409 means the mode flipped off or the placement is mid-repair;
        // refresh status so the page swaps to the matching empty state.
        const detail =
          typeof error === 'object' && error !== null && 'data' in error
            ? String(
                (error as { data?: { detail?: string } }).data?.detail ??
                  t('stewardChat.errors.requestFailed', 'The steward request failed'),
              )
            : t('stewardChat.errors.requestFailed', 'The steward request failed');
        addToast({ type: 'error', message: detail });
        void refetch();
      }
    },
    [messages, isLoading, sendChat, refetch, t],
  );

  if (status && !status.enabled) {
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

  if (status && status.enabled && !status.present) {
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
      {status?.steward_model && (
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
          <ChatMessages messages={messages} isLoading={isLoading} />
        )}
      </MessagesScroll>
      <InputArea>
        <ChatForm
          onSend={handleSend}
          isLoading={isLoading}
          canSendMessages
          placeholder={t(
            'stewardChat.placeholder',
            'Ask about the cluster...',
          )}
        />
      </InputArea>
    </Container>
  );
}
