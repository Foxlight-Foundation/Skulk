import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';
import { ChatMessages } from '../chat/ChatMessages';
import { ChatForm } from '../chat/ChatForm';
import type { ChatMessage } from '../../types/chat';
import type { ChatUploadedFile } from '../../types/chat';
import type { ModelInfo } from '../../types/models';
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
const MAX_TOOL_ROUNDS = 4;
const GPT_OSS_WEB_SEARCH_TOOL = {
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
} as const;

type ApiMessagePayload = Record<string, unknown>;

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

function splitReasoningDecoratedContent(raw: string): { content: string; thinking: string } {
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

function buildApiMessages(messages: ChatMessage[]): ApiMessagePayload[] {
  return messages.map((message) => {
    if (message.attachments?.some((attachment) => attachment.type.startsWith('image/') && attachment.preview)) {
      const parts: Array<{ type: string; text?: string; image_url?: { url: string } }> = [];
      for (const attachment of message.attachments) {
        if (attachment.type.startsWith('image/') && attachment.preview) {
          parts.push({ type: 'image_url', image_url: { url: attachment.preview } });
        }
      }
      if (message.content) {
        parts.push({ type: 'text', text: message.content });
      }
      return { role: message.role, content: parts };
    }
    return { role: message.role, content: message.content };
  });
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

async function executeWebSearchToolCall(toolCall: StreamToolCall): Promise<string> {
  const functionCall = toolCall.function;
  if (!functionCall?.name || functionCall.name !== 'web_search') {
    throw new Error(`Unsupported tool call: ${functionCall?.name ?? 'unknown'}`);
  }
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
  const [modelThinkingToggleSupport, setModelThinkingToggleSupport] = useState<Record<string, boolean>>({});
  const [modelImageInputSupport, setModelImageInputSupport] = useState<Record<string, boolean>>({});
  const [modelWebSearchToolSupport, setModelWebSearchToolSupport] = useState<Record<string, boolean>>({});
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
        const data = await res.json() as { data?: ModelInfo[] };
        const toggleSupport: Record<string, boolean> = {};
        const imageSupport: Record<string, boolean> = {};
        const webSearchSupport: Record<string, boolean> = {};
        const ctxLens: Record<string, number> = {};
        for (const m of data.data ?? []) {
          if (m.id) {
            toggleSupport[m.id] = m.resolved_capabilities?.supports_thinking_toggle ?? false;
            imageSupport[m.id] = m.resolved_capabilities?.supports_image_input ?? false;
            webSearchSupport[m.id] = Boolean(
              m.tooling?.builtin_tools?.includes('web_search')
              || m.resolved_capabilities?.builtin_tools?.includes('web_search'),
            );
          }
          if (m.id && m.context_length) ctxLens[m.id] = m.context_length;
        }
        setModelThinkingToggleSupport(toggleSupport);
        setModelImageInputSupport(imageSupport);
        setModelWebSearchToolSupport(webSearchSupport);
        setModelContextLengths(ctxLens);
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
  const supportsBuiltinWebSearch = selectedModelId
    ? (modelWebSearchToolSupport[selectedModelId] ?? false)
    : false;

  useEffect(() => {
    if (!supportsThinking && thinkingEnabled) {
      setThinkingEnabled(false);
    }
  }, [supportsThinking, thinkingEnabled]);

  // Ready models
  const readyModels = useMemo(
    () => readyInstances.filter((i) => (i.status === 'ready' || i.status === 'running') && !i.isEmbedding),
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

  const handleSend = useCallback(async (text: string, files: ChatUploadedFile[]) => {
    if (!selectedModelId || isLoading) return;

    // Convert image files to base64 data URLs for the API and message history
    const imageAttachments: { dataUrl: string; file: ChatUploadedFile }[] = [];
    for (const f of files) {
      if (f.type.startsWith('image/') && (f.file || f.preview)) {
        const dataUrl = await readUploadedImageAsDataUrl(f);
        imageAttachments.push({ dataUrl, file: f });
      }
    }

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
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

    try {
      const apiMessages: ApiMessagePayload[] = buildApiMessages(allMessages);
      const requestTools = supportsBuiltinWebSearch ? [GPT_OSS_WEB_SEARCH_TOOL] : undefined;

      for (let toolRound = 0; toolRound < MAX_TOOL_ROUNDS; toolRound++) {
        resetStallTimer();

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
            if (!trimmed || trimmed.startsWith(':')) continue;
            if (!trimmed.startsWith('data: ')) continue;
            const data = trimmed.slice(6);
            if (data === '[DONE]') continue;

            try {
              const parsed = JSON.parse(data) as {
                choices?: Array<{
                  delta?: {
                    content?: string;
                    reasoning_content?: string;
                    tool_calls?: StreamToolCall[];
                  };
                }>;
              };
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
          fullThinking = mergeThinkingContent(fullThinking, iterationThinking);
          finalRawContent = separatedContent.content;
          break;
        }

        fullThinking = mergeThinkingContent(fullThinking, iterationThinking);

        for (const toolCall of iterationToolCalls) {
          const toolName = toolCall.function?.name ?? 'unknown';
          const toolCallId = toolCall.id || crypto.randomUUID();
          let toolOutput: string;

          try {
            toolOutput = await executeWebSearchToolCall(toolCall);
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Tool execution failed.';
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
        finalRawContent = 'Error: web search tool loop exceeded the safety limit.';
      }
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        // User cancelled
        if (requestTimedOut) {
          finalRawContent = `Error: generation stalled for more than ${Math.round(lastStallTimeoutMs / 1000)} seconds.`;
        }
      } else {
        finalRawContent = finalRawContent || `Error: ${(err as Error).message}`;
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
        id: crypto.randomUUID(),
        role: 'assistant',
        content: finalAssistantContent,
        timestamp: Date.now(),
        ttftMs: firstTokenTime ? firstTokenTime - startTime : undefined,
        tps: lastTps,
        thinkingContent: fullThinking || undefined,
      };

      addMessage(assistantMsg);
    }
    setStreamingContent(null);
    setStreamingThinking(null);
    setIsLoading(false);
    abortRef.current = null;

  }, [selectedModelId, isLoading, thinkingEnabled, supportsThinking, supportsBuiltinWebSearch, addMessage]);

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
          supportsImageAttachments={supportsImageAttachments}
          placeholder={selectedModelId ? `Message ${selectedLabel}…` : 'Select a model to chat'}
        />
      </InputArea>
    </Container>
  );
}
