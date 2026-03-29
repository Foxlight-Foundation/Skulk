import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';
import { ChatMessages } from '../chat/ChatMessages';
import { ChatForm } from '../chat/ChatForm';
import type { ChatMessage } from '../../types/chat';
import type { ChatUploadedFile } from '../../types/chat';
import type { InstanceCardData } from '../layout/InstancePanel';
import { useChatStore } from '../../stores/chatStore';
import { useUIStore } from '../../stores/uiStore';

/* ── Types ────────────────────────────────────────────── */

export interface ChatViewProps {
  /** Ready instances the user can chat with. */
  readyInstances: InstanceCardData[];
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
  color: #FFD700;
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

/* ── Component ────────────────────────────────────────── */

export function ChatView({ readyInstances, className }: ChatViewProps) {
  // Store state
  const selectedModelId = useChatStore((s) => s.selectedModelId);
  const activeConversationId = useChatStore((s) => s.activeConversationId);
  const messages = useChatStore((s) =>
    s.activeConversationId ? s.conversations[s.activeConversationId]?.messages ?? EMPTY_MESSAGES : EMPTY_MESSAGES,
  );
  const selectModel = useChatStore((s) => s.selectModel);
  const addMessage = useChatStore((s) => s.addMessage);
  const deleteMessageAction = useChatStore((s) => s.deleteMessage);
  const editMessageAction = useChatStore((s) => s.editMessage);
  const removeLastAssistantMessages = useChatStore((s) => s.removeLastAssistantMessages);

  // Local transient state
  const [streamingContent, setStreamingContent] = useState<string | null>(null);
  const [streamingThinking, setStreamingThinking] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [thinkingEnabled, setThinkingEnabled] = useState(false);
  const [ttftMs, setTtftMs] = useState<number | null>(null);
  const [tps, setTps] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [modelCapabilities, setModelCapabilities] = useState<Record<string, string[]>>({});
  const [modelContextLengths, setModelContextLengths] = useState<Record<string, number>>({});

  // Restore scroll position after store hydration + DOM render
  const chatScrollTop = useUIStore((s) => s.chatScrollTop);
  const setChatScrollTop = useUIStore((s) => s.setChatScrollTop);
  const scrollRestored = useRef(false);
  useEffect(() => {
    if (scrollRestored.current || chatScrollTop <= 0) return;

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
        const data = await res.json();
        const caps: Record<string, string[]> = {};
        const ctxLens: Record<string, number> = {};
        for (const m of data.data ?? []) {
          if (m.id && m.capabilities) caps[m.id] = m.capabilities;
          if (m.id && m.context_length) ctxLens[m.id] = m.context_length;
        }
        setModelCapabilities(caps);
        setModelContextLengths(ctxLens);
      } catch { /* ignore */ }
    })();
  }, []);

  const contextLength = selectedModelId ? modelContextLengths[selectedModelId] ?? 0 : 0;

  const supportsThinking = selectedModelId
    ? (modelCapabilities[selectedModelId]?.includes('thinking_toggle') ?? false)
    : false;

  // Ready models
  const readyModels = useMemo(
    () => readyInstances.filter((i) => i.status === 'ready' || i.status === 'running'),
    [readyInstances],
  );

  // Auto-select first ready model if none selected
  useEffect(() => {
    if (!selectedModelId && readyModels.length > 0) {
      selectModel(readyModels[0].modelId);
    }
  }, [selectedModelId, readyModels, selectModel]);

  const selectedLabel = useMemo(() => {
    if (!selectedModelId) return undefined;
    const parts = selectedModelId.split('/');
    return parts[parts.length - 1];
  }, [selectedModelId]);

  const handleSend = useCallback(async (text: string, _files: ChatUploadedFile[]) => {
    if (!selectedModelId || isLoading) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };

    addMessage(userMsg);
    setIsLoading(true);
    setStreamingContent('');
    setStreamingThinking(null);
    setTtftMs(null);
    setTps(null);

    // Read messages from store (includes the user message we just added)
    const storeState = useChatStore.getState();
    const activeConvo = storeState.activeConversationId
      ? storeState.conversations[storeState.activeConversationId]
      : undefined;
    if (!activeConvo) {
      setIsLoading(false);
      setStreamingContent(null);
      return;
    }
    const allMessages = activeConvo.messages;

    const controller = new AbortController();
    abortRef.current = controller;

    const startTime = performance.now();
    let firstTokenTime: number | null = null;
    let tokenCount = 0;
    let rawContent = '';
    let fullThinking = '';
    let lastTps: number | undefined;

    try {
      const res = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModelId,
          messages: allMessages.map((m) => ({ role: m.role, content: m.content })),
          stream: true,
          ...(thinkingEnabled ? { enable_thinking: true } : {}),
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

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith(':')) continue;
          if (!trimmed.startsWith('data: ')) continue;
          const data = trimmed.slice(6);
          if (data === '[DONE]') continue;

          try {
            const parsed = JSON.parse(data);
            const delta = parsed.choices?.[0]?.delta;

            const hasToken = delta?.content || delta?.reasoning_content;

            // Record TTFT on first token of any kind
            if (hasToken && firstTokenTime === null) {
              firstTokenTime = performance.now();
              setTtftMs(firstTokenTime - startTime);
            }

            // Thinking via reasoning_content field
            if (delta?.reasoning_content) {
              fullThinking += delta.reasoning_content;
              setStreamingThinking(fullThinking);
            }

            // Content — parse out inline <think> tags
            if (delta?.content) {
              rawContent += delta.content;

              // Process <think> tags incrementally
              let displayContent = '';
              let thinkContent = '';
              let i = 0;
              let inTag = false;
              const raw = rawContent;

              while (i < raw.length) {
                if (!inTag && raw.startsWith('<think>', i)) {
                  inTag = true;
                  i += 7;
                } else if (inTag && raw.startsWith('</think>', i)) {
                  inTag = false;
                  i += 8;
                } else if (inTag) {
                  thinkContent += raw[i];
                  i++;
                } else {
                  displayContent += raw[i];
                  i++;
                }
              }

              // If we found think tags, route to thinking
              if (thinkContent) {
                fullThinking = thinkContent;
                setStreamingThinking(fullThinking);
              }
              setStreamingContent(displayContent || null);
            }

            // Update TPS on every token (thinking or content)
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
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        // User cancelled
      } else {
        rawContent = rawContent || `Error: ${(err as Error).message}`;
      }
    }

    // Finalize assistant message
    const assistantMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: rawContent.replace(/<think>[\s\S]*?<\/think>/gi, '').replace(/<think>[\s\S]*/i, '').trim(),
      timestamp: Date.now(),
      ttftMs: firstTokenTime ? firstTokenTime - startTime : undefined,
      tps: lastTps,
      thinkingContent: fullThinking || undefined,
    };

    addMessage(assistantMsg);
    setStreamingContent(null);
    setStreamingThinking(null);
    setIsLoading(false);
    abortRef.current = null;

  }, [selectedModelId, isLoading, thinkingEnabled, addMessage]);

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

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
      const state = useChatStore.getState();
      const convo = state.activeConversationId
        ? state.conversations[state.activeConversationId]
        : undefined;
      if (!convo) return;
      const lastUser = convo.messages.filter((m) => m.role === 'user').pop();
      if (lastUser) {
        handleSend(lastUser.content, []);
      }
    }, 50);
  }, [handleSend, removeLastAssistantMessages]);

  // Thinking expansion state — persisted per conversation in session store
  const expandedThinkingMap = useUIStore((s) => s.expandedThinking);
  const setExpandedThinking = useUIStore((s) => s.setExpandedThinking);
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

  if (readyModels.length === 0) {
    return (
      <NoModels>
        No models are ready. Launch a model from the Model Store to start chatting.
      </NoModels>
    );
  }

  const modelSelector = readyModels.length > 1 ? (
    <ModelSelect value={selectedModelId ?? ''} onChange={(e) => selectModel(e.target.value)}>
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
          expandedThinkingIds={expandedThinkingIds}
          onToggleThinking={handleToggleThinking}
        />
      </MessagesScroll>
      <InputArea>
        <ChatForm
          onSend={handleSend}
          onCancel={handleCancel}
          isLoading={isLoading}
          modelLabel={selectedLabel}
          modelSelector={modelSelector}
          ttftMs={ttftMs}
          tps={tps}
          contextLength={contextLength}
          showThinkingToggle={supportsThinking}
          thinkingEnabled={thinkingEnabled}
          onToggleThinking={() => setThinkingEnabled((v) => !v)}
          placeholder={selectedModelId ? `Message ${selectedLabel}…` : 'Select a model to chat'}
        />
      </InputArea>
    </Container>
  );
}
