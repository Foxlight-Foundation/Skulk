/**
 * Chat Store
 *
 * All conversation state: messages, active conversation, active model,
 * SSE streaming, image generation/editing, image generation params.
 * Replaces the chat slices of app.svelte.ts.
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { streamChatCompletion, generateImage, editImage } from '../api/streaming';
import type { ChatMessage, Conversation, MessageContent } from '../api/types';
import { useToastStore } from './toastStore';

function generateId(): string {
  return crypto.randomUUID();
}

function deriveTitle(content: MessageContent): string {
  const text = typeof content === 'string' ? content : content[0]
    ? (content[0] as { type: string; text?: string }).text ?? ''
    : '';
  return text.slice(0, 60).trim() || 'New conversation';
}

// ─── Image generation / edit mode ────────────────────────────────────────────

export type ChatMode = 'chat' | 'image-gen' | 'image-edit';

// ─── Image generation params ──────────────────────────────────────────────────

export type ImageSize =
  | 'auto' | '512x512' | '768x768' | '1024x1024'
  | '1024x768' | '768x1024' | '1024x1536' | '1536x1024';

export type ImageQuality = 'low' | 'medium' | 'high';
export type ImageOutputFormat = 'png' | 'jpeg';
export type ImageInputFidelity = 'low' | 'high';

export interface ImageGenerationParams {
  size: ImageSize;
  quality: ImageQuality;
  outputFormat: ImageOutputFormat;
  numImages: number;
  stream: boolean;
  partialImages: number;
  inputFidelity: ImageInputFidelity;
  seed: number | null;
  numInferenceSteps: number | null;
  guidance: number | null;
  negativePrompt: string | null;
  numSyncSteps: number | null;
}

export const DEFAULT_IMAGE_PARAMS: ImageGenerationParams = {
  size: 'auto',
  quality: 'medium',
  outputFormat: 'png',
  numImages: 1,
  stream: true,
  partialImages: 3,
  inputFidelity: 'high',
  seed: null,
  numInferenceSteps: null,
  guidance: null,
  negativePrompt: null,
  numSyncSteps: null,
};

// ─── Store shape ──────────────────────────────────────────────────────────────

interface ChatState {
  conversations: Conversation[];
  activeConversationId: string | null;
  selectedModelId: string | null;
  mode: ChatMode;
  isStreaming: boolean;
  editingImageBase64: string | null;
  imageGenerationParams: ImageGenerationParams;

  // Stream abort controller — not persisted
  _abortController: AbortController | null;

  // ── Selectors ──────────────────────────────────────────────────────────────
  getActiveConversation: () => Conversation | null;
  getMessages: () => ChatMessage[];
  hasStartedChat: () => boolean;
  getLastTtft: () => number | null;
  getLastTps: () => number | null;

  // ── Conversation management ────────────────────────────────────────────────
  createConversation: () => string;
  deleteConversation: (id: string) => void;
  clearAllConversations: () => void;
  setActiveConversation: (id: string) => void;
  renameConversation: (id: string, title: string) => void;

  // ── Model ──────────────────────────────────────────────────────────────────
  setSelectedModel: (modelId: string | null) => void;

  // ── Mode ───────────────────────────────────────────────────────────────────
  setMode: (mode: ChatMode) => void;
  setEditingImage: (base64: string | null) => void;

  // ── Thinking ──────────────────────────────────────────────────────────────
  setConversationThinking: (enabled: boolean) => void;

  // ── Image params ──────────────────────────────────────────────────────────
  setImageGenerationParams: (partial: Partial<ImageGenerationParams>) => void;
  resetImageGenerationParams: () => void;

  // ── Sending messages ───────────────────────────────────────────────────────
  sendMessage: (content: MessageContent) => Promise<void>;
  generateImageMessage: (prompt: string) => Promise<void>;
  editImageMessage: (prompt: string) => Promise<void>;
  stopStreaming: () => void;

  // ── Message editing ────────────────────────────────────────────────────────
  editMessage: (messageId: string, newContent: MessageContent) => void;
  deleteMessageAndAfter: (messageId: string) => void;
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      conversations: [],
      activeConversationId: null,
      selectedModelId: null,
      mode: 'chat',
      isStreaming: false,
      editingImageBase64: null,
      imageGenerationParams: DEFAULT_IMAGE_PARAMS,
      _abortController: null,

      // ── Selectors ────────────────────────────────────────────────────────────

      getActiveConversation: () => {
        const { conversations, activeConversationId } = get();
        return conversations.find((c) => c.id === activeConversationId) ?? null;
      },

      getMessages: () => get().getActiveConversation()?.messages ?? [],

      hasStartedChat: () => (get().getMessages().length ?? 0) > 0,

      getLastTtft: () => {
        const messages = get().getMessages();
        for (let i = messages.length - 1; i >= 0; i--) {
          const m = messages[i];
          if (m != null && m.role === 'assistant' && (m.ttft != null || m.ttftMs != null)) {
            return m.ttft ?? m.ttftMs ?? null;
          }
        }
        return null;
      },

      getLastTps: () => {
        const messages = get().getMessages();
        for (let i = messages.length - 1; i >= 0; i--) {
          const m = messages[i];
          if (m != null && m.role === 'assistant' && m.tps != null) return m.tps;
        }
        return null;
      },

      // ── Conversation management ───────────────────────────────────────────────

      createConversation: () => {
        const id = generateId();
        const conversation: Conversation = {
          id,
          title: 'New conversation',
          createdAt: Date.now(),
          updatedAt: Date.now(),
          messages: [],
          modelId: get().selectedModelId ?? undefined,
        };
        set((state) => ({
          conversations: [conversation, ...state.conversations],
          activeConversationId: id,
        }));
        return id;
      },

      deleteConversation: (id) => {
        set((state) => {
          const remaining = state.conversations.filter((c) => c.id !== id);
          const newActiveId =
            state.activeConversationId === id
              ? (remaining[0]?.id ?? null)
              : state.activeConversationId;
          return { conversations: remaining, activeConversationId: newActiveId };
        });
      },

      clearAllConversations: () => {
        set({ conversations: [], activeConversationId: null });
      },

      setActiveConversation: (id) => {
        set({ activeConversationId: id });
      },

      renameConversation: (id, title) => {
        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === id ? { ...c, title } : c,
          ),
        }));
      },

      // ── Model ─────────────────────────────────────────────────────────────────

      setSelectedModel: (modelId) => {
        set({ selectedModelId: modelId });
      },

      // ── Mode ──────────────────────────────────────────────────────────────────

      setMode: (mode) => set({ mode }),
      setEditingImage: (base64) => set({ editingImageBase64: base64 }),

      // ── Thinking ──────────────────────────────────────────────────────────────

      setConversationThinking: (enabled) => {
        const { activeConversationId } = get();
        if (!activeConversationId) return;
        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === activeConversationId ? { ...c, thinkingEnabled: enabled } : c,
          ),
        }));
      },

      // ── Image params ──────────────────────────────────────────────────────────

      setImageGenerationParams: (partial) => {
        set((state) => ({
          imageGenerationParams: { ...state.imageGenerationParams, ...partial },
        }));
      },

      resetImageGenerationParams: () => {
        set({ imageGenerationParams: DEFAULT_IMAGE_PARAMS });
      },

      // ── Message sending ───────────────────────────────────────────────────────

      sendMessage: async (content) => {
        const { selectedModelId, createConversation, activeConversationId } = get();
        if (!selectedModelId) {
          useToastStore.getState().addToast('No model selected', 'warning');
          return;
        }

        // Ensure there's an active conversation
        let convId = activeConversationId;
        if (!convId) {
          convId = createConversation();
        }

        // Add user message
        const userMessage: ChatMessage = {
          id: generateId(),
          role: 'user',
          content,
          createdAt: Date.now(),
        };

        // Add streaming assistant placeholder
        const assistantMessage: ChatMessage = {
          id: generateId(),
          role: 'assistant',
          content: '',
          createdAt: Date.now(),
          isStreaming: true,
          isPrefilling: true,
        };

        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === convId
              ? {
                  ...c,
                  messages: [...c.messages, userMessage, assistantMessage],
                  title:
                    c.messages.length === 0 ? deriveTitle(content) : c.title,
                  updatedAt: Date.now(),
                }
              : c,
          ),
          isStreaming: true,
        }));

        const conversationMessages = get()
          .getActiveConversation()
          ?.messages.slice(0, -1) // exclude the placeholder
          .map((m) => ({ role: m.role, content: m.content })) ?? [];

        const abortController = streamChatCompletion(
          {
            model: selectedModelId,
            messages: conversationMessages,
            stream: true,
          },
          {
            onToken: (token) => {
              set((state) => ({
                conversations: state.conversations.map((c) =>
                  c.id === convId
                    ? {
                        ...c,
                        messages: c.messages.map((m) =>
                          m.id === assistantMessage.id
                            ? {
                                ...m,
                                content:
                                  (typeof m.content === 'string'
                                    ? m.content
                                    : '') + token,
                                isPrefilling: false,
                              }
                            : m,
                        ),
                      }
                    : c,
                ),
              }));
            },
            onTtft: (ms) => {
              set((state) => ({
                conversations: state.conversations.map((c) =>
                  c.id === convId
                    ? {
                        ...c,
                        messages: c.messages.map((m) =>
                          m.id === assistantMessage.id ? { ...m, ttft: ms } : m,
                        ),
                      }
                    : c,
                ),
              }));
            },
            onComplete: (tps) => {
              set((state) => ({
                isStreaming: false,
                _abortController: null,
                conversations: state.conversations.map((c) =>
                  c.id === convId
                    ? {
                        ...c,
                        messages: c.messages.map((m) =>
                          m.id === assistantMessage.id
                            ? { ...m, isStreaming: false, tps }
                            : m,
                        ),
                      }
                    : c,
                ),
              }));
            },
            onError: (error) => {
              set((state) => ({
                isStreaming: false,
                _abortController: null,
                conversations: state.conversations.map((c) =>
                  c.id === convId
                    ? {
                        ...c,
                        messages: c.messages.map((m) =>
                          m.id === assistantMessage.id
                            ? {
                                ...m,
                                isStreaming: false,
                                content: `Error: ${error.message}`,
                              }
                            : m,
                        ),
                      }
                    : c,
                ),
              }));
            },
            onPrefillStart: () => {
              set((state) => ({
                conversations: state.conversations.map((c) =>
                  c.id === convId
                    ? {
                        ...c,
                        messages: c.messages.map((m) =>
                          m.id === assistantMessage.id
                            ? { ...m, isPrefilling: true }
                            : m,
                        ),
                      }
                    : c,
                ),
              }));
            },
            onPrefillEnd: () => {
              set((state) => ({
                conversations: state.conversations.map((c) =>
                  c.id === convId
                    ? {
                        ...c,
                        messages: c.messages.map((m) =>
                          m.id === assistantMessage.id
                            ? { ...m, isPrefilling: false }
                            : m,
                        ),
                      }
                    : c,
                ),
              }));
            },
          },
        );

        set({ _abortController: abortController });
      },

      generateImageMessage: async (prompt) => {
        const { selectedModelId, createConversation, activeConversationId } = get();
        if (!selectedModelId) {
          useToastStore.getState().addToast('No model selected', 'warning');
          return;
        }

        let convId = activeConversationId;
        if (!convId) convId = createConversation();

        const userMessage: ChatMessage = {
          id: generateId(),
          role: 'user',
          content: prompt,
          createdAt: Date.now(),
        };

        const assistantMessage: ChatMessage = {
          id: generateId(),
          role: 'assistant',
          content: '',
          createdAt: Date.now(),
          isStreaming: true,
        };

        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === convId
              ? { ...c, messages: [...c.messages, userMessage, assistantMessage] }
              : c,
          ),
          isStreaming: true,
        }));

        try {
          const url = await generateImage({ model: selectedModelId, prompt });
          set((state) => ({
            isStreaming: false,
            conversations: state.conversations.map((c) =>
              c.id === convId
                ? {
                    ...c,
                    messages: c.messages.map((m) =>
                      m.id === assistantMessage.id
                        ? { ...m, isStreaming: false, imageUrl: url, content: url }
                        : m,
                    ),
                  }
                : c,
            ),
          }));
        } catch (error) {
          set((state) => ({
            isStreaming: false,
            conversations: state.conversations.map((c) =>
              c.id === convId
                ? {
                    ...c,
                    messages: c.messages.map((m) =>
                      m.id === assistantMessage.id
                        ? {
                            ...m,
                            isStreaming: false,
                            content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
                          }
                        : m,
                    ),
                  }
                : c,
            ),
          }));
        }
      },

      editImageMessage: async (prompt) => {
        const { selectedModelId, editingImageBase64, createConversation, activeConversationId } = get();
        if (!selectedModelId || !editingImageBase64) {
          useToastStore.getState().addToast('No model or image selected', 'warning');
          return;
        }

        let convId = activeConversationId;
        if (!convId) convId = createConversation();

        const userMessage: ChatMessage = {
          id: generateId(),
          role: 'user',
          content: prompt,
          createdAt: Date.now(),
        };

        const assistantMessage: ChatMessage = {
          id: generateId(),
          role: 'assistant',
          content: '',
          createdAt: Date.now(),
          isStreaming: true,
        };

        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === convId
              ? { ...c, messages: [...c.messages, userMessage, assistantMessage] }
              : c,
          ),
          isStreaming: true,
        }));

        try {
          const url = await editImage({
            model: selectedModelId,
            image: editingImageBase64,
            prompt,
          });
          set((state) => ({
            isStreaming: false,
            conversations: state.conversations.map((c) =>
              c.id === convId
                ? {
                    ...c,
                    messages: c.messages.map((m) =>
                      m.id === assistantMessage.id
                        ? { ...m, isStreaming: false, imageUrl: url, content: url }
                        : m,
                    ),
                  }
                : c,
            ),
          }));
        } catch (error) {
          set((state) => ({
            isStreaming: false,
            conversations: state.conversations.map((c) =>
              c.id === convId
                ? {
                    ...c,
                    messages: c.messages.map((m) =>
                      m.id === assistantMessage.id
                        ? {
                            ...m,
                            isStreaming: false,
                            content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
                          }
                        : m,
                    ),
                  }
                : c,
            ),
          }));
        }
      },

      stopStreaming: () => {
        const { _abortController } = get();
        _abortController?.abort();
        set({ isStreaming: false, _abortController: null });
      },

      // ── Message editing ───────────────────────────────────────────────────────

      editMessage: (messageId, newContent) => {
        const { activeConversationId } = get();
        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === activeConversationId
              ? {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === messageId ? { ...m, content: newContent } : m,
                  ),
                }
              : c,
          ),
        }));
      },

      deleteMessageAndAfter: (messageId) => {
        const { activeConversationId } = get();
        set((state) => ({
          conversations: state.conversations.map((c) => {
            if (c.id !== activeConversationId) return c;
            const idx = c.messages.findIndex((m) => m.id === messageId);
            if (idx === -1) return c;
            return { ...c, messages: c.messages.slice(0, idx) };
          }),
        }));
      },
    }),
    {
      name: 'exo-chat',
      // Exclude non-serialisable / ephemeral state from persistence
      partialize: (state) => ({
        conversations: state.conversations,
        activeConversationId: state.activeConversationId,
        selectedModelId: state.selectedModelId,
        imageGenerationParams: state.imageGenerationParams,
      }),
    },
  ),
);
