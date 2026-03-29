import { create } from 'zustand';
import { devtools, persist, createJSONStorage } from 'zustand/middleware';
import type { ChatMessage, Conversation } from '../types/chat';

/* ── Helpers ──────────────────────────────────────────── */

function uuid(): string {
  return crypto.randomUUID();
}

function stripTransientFields(
  conversations: Record<string, Conversation>,
): Record<string, Conversation> {
  const stripped: Record<string, Conversation> = {};
  for (const [id, convo] of Object.entries(conversations)) {
    stripped[id] = {
      ...convo,
      messages: convo.messages.map((msg) => {
        const { tokens, generatedImages, ...rest } = msg;
        return rest;
      }),
    };
  }
  return stripped;
}

function autoName(content: string): string {
  const trimmed = content.trim();
  if (!trimmed) return 'New conversation';
  if (trimmed.length <= 50) return trimmed;
  return trimmed.slice(0, 50) + '...';
}

/* ── Store Interface ──────────────────────────────────── */

export interface ChatState {
  conversations: Record<string, Conversation>;
  activeConversationId: string | null;
  selectedModelId: string | null;
  modelToConversationId: Record<string, string>;

  // Actions
  selectModel: (modelId: string) => void;
  addMessage: (message: ChatMessage) => void;
  deleteMessage: (messageId: string) => void;
  editMessage: (messageId: string, content: string) => void;
  removeLastAssistantMessages: () => void;
  newConversation: (modelId: string) => string;
  selectConversation: (conversationId: string) => void;
  deleteConversation: (conversationId: string) => void;
  setSummary: (conversationId: string, summary: string) => void;
}

/* ── Selectors ────────────────────────────────────────── */

export const selectActiveConversation = (state: ChatState): Conversation | null =>
  state.activeConversationId
    ? state.conversations[state.activeConversationId] ?? null
    : null;

export const selectActiveMessages = (state: ChatState): ChatMessage[] =>
  selectActiveConversation(state)?.messages ?? [];

export const selectAllConversationsSorted = (state: ChatState): Conversation[] =>
  Object.values(state.conversations).sort((a, b) => b.updatedAt - a.updatedAt);

export const selectConversationsForModel = (modelId: string) => (state: ChatState): Conversation[] =>
  Object.values(state.conversations)
    .filter((c) => c.modelId === modelId)
    .sort((a, b) => b.updatedAt - a.updatedAt);

/* ── Store ────────────────────────────────────────────── */

export const useChatStore = create<ChatState>()(
  devtools(
  persist(
    (set, get) => ({
      conversations: {},
      activeConversationId: null,
      selectedModelId: null,
      modelToConversationId: {},

      selectModel: (modelId: string) => {
        const state = get();
        const now = Date.now();
        const conversations = { ...state.conversations };

        // Update timestamp on current conversation
        if (state.activeConversationId && conversations[state.activeConversationId]) {
          conversations[state.activeConversationId] = {
            ...conversations[state.activeConversationId],
            updatedAt: now,
          };
        }

        // If same model, no switch needed
        if (modelId === state.selectedModelId && state.activeConversationId) {
          set({ conversations });
          return;
        }

        // Find or create conversation for new model
        const existingId = state.modelToConversationId[modelId];
        if (existingId && conversations[existingId]) {
          set({
            conversations,
            activeConversationId: existingId,
            selectedModelId: modelId,
          });
        } else {
          const newId = uuid();
          conversations[newId] = {
            id: newId,
            name: 'New conversation',
            modelId,
            createdAt: now,
            updatedAt: now,
            messages: [],
          };
          set({
            conversations,
            activeConversationId: newId,
            selectedModelId: modelId,
            modelToConversationId: {
              ...state.modelToConversationId,
              [modelId]: newId,
            },
          });
        }
      },

      addMessage: (message: ChatMessage) => {
        const state = get();
        const convoId = state.activeConversationId;
        if (!convoId || !state.conversations[convoId]) return;

        const convo = state.conversations[convoId];
        const updatedMessages = [...convo.messages, message];

        // Auto-name from first user message
        let name = convo.name;
        if (name === 'New conversation' && message.role === 'user') {
          name = autoName(message.content);
        }

        set({
          conversations: {
            ...state.conversations,
            [convoId]: {
              ...convo,
              messages: updatedMessages,
              name,
              updatedAt: Date.now(),
            },
          },
        });
      },

      deleteMessage: (messageId: string) => {
        const state = get();
        const convoId = state.activeConversationId;
        if (!convoId || !state.conversations[convoId]) return;

        const convo = state.conversations[convoId];
        set({
          conversations: {
            ...state.conversations,
            [convoId]: {
              ...convo,
              messages: convo.messages.filter((m) => m.id !== messageId),
              updatedAt: Date.now(),
            },
          },
        });
      },

      editMessage: (messageId: string, content: string) => {
        const state = get();
        const convoId = state.activeConversationId;
        if (!convoId || !state.conversations[convoId]) return;

        const convo = state.conversations[convoId];
        set({
          conversations: {
            ...state.conversations,
            [convoId]: {
              ...convo,
              messages: convo.messages.map((m) =>
                m.id === messageId ? { ...m, content } : m,
              ),
              updatedAt: Date.now(),
            },
          },
        });
      },

      removeLastAssistantMessages: () => {
        const state = get();
        const convoId = state.activeConversationId;
        if (!convoId || !state.conversations[convoId]) return;

        const convo = state.conversations[convoId];
        const msgs = [...convo.messages];
        while (msgs.length > 0 && msgs[msgs.length - 1].role === 'assistant') {
          msgs.pop();
        }
        set({
          conversations: {
            ...state.conversations,
            [convoId]: { ...convo, messages: msgs, updatedAt: Date.now() },
          },
        });
      },

      newConversation: (modelId: string) => {
        const now = Date.now();
        const newId = uuid();
        const newConvo: Conversation = {
          id: newId,
          name: 'New conversation',
          modelId,
          createdAt: now,
          updatedAt: now,
          messages: [],
        };

        set((state) => ({
          conversations: { ...state.conversations, [newId]: newConvo },
          activeConversationId: newId,
          selectedModelId: modelId,
          modelToConversationId: {
            ...state.modelToConversationId,
            [modelId]: newId,
          },
        }));

        return newId;
      },

      selectConversation: (conversationId: string) => {
        const state = get();
        const convo = state.conversations[conversationId];
        if (!convo) return;

        set({
          activeConversationId: conversationId,
          selectedModelId: convo.modelId,
          modelToConversationId: {
            ...state.modelToConversationId,
            [convo.modelId]: conversationId,
          },
        });
      },

      deleteConversation: (conversationId: string) => {
        const state = get();
        const convo = state.conversations[conversationId];
        if (!convo) return;

        const { [conversationId]: _, ...rest } = state.conversations;
        const modelMap = { ...state.modelToConversationId };
        if (modelMap[convo.modelId] === conversationId) {
          delete modelMap[convo.modelId];
        }

        set({
          conversations: rest,
          modelToConversationId: modelMap,
          activeConversationId:
            state.activeConversationId === conversationId ? null : state.activeConversationId,
          selectedModelId:
            state.activeConversationId === conversationId ? null : state.selectedModelId,
        });
      },

      setSummary: (conversationId: string, summary: string) => {
        const state = get();
        const convo = state.conversations[conversationId];
        if (!convo) return;

        // Update name if still default or auto-generated
        const name = convo.name === 'New conversation' || convo.name.endsWith('...')
          ? summary
          : convo.name;

        set({
          conversations: {
            ...state.conversations,
            [conversationId]: { ...convo, summary, name, updatedAt: Date.now() },
          },
        });
      },
    }),
    {
      name: 'skulk-chat',
      version: 1,
      storage: createJSONStorage(() => ({
        getItem: (name: string) => {
          const durable = localStorage.getItem(name);
          const session = sessionStorage.getItem(name + '-session');
          const d = durable ? JSON.parse(durable) : {};
          const s = session ? JSON.parse(session) : {};
          return JSON.stringify({
            state: { ...(d.state ?? {}), ...(s.state ?? {}) },
            version: d.version ?? s.version ?? 1,
          });
        },
        setItem: (name: string, value: string) => {
          const parsed = JSON.parse(value);
          const state = parsed.state ?? {};
          const { activeConversationId, selectedModelId, ...rest } = state;
          localStorage.setItem(name, JSON.stringify({
            state: {
              conversations: rest.conversations,
              modelToConversationId: rest.modelToConversationId,
            },
            version: parsed.version,
          }));
          sessionStorage.setItem(name + '-session', JSON.stringify({
            state: { activeConversationId, selectedModelId },
            version: parsed.version,
          }));
        },
        removeItem: (name: string) => {
          localStorage.removeItem(name);
          sessionStorage.removeItem(name + '-session');
        },
      })),
      partialize: (state) => ({
        conversations: stripTransientFields(state.conversations),
        modelToConversationId: state.modelToConversationId,
        activeConversationId: state.activeConversationId,
        selectedModelId: state.selectedModelId,
      }),
    },
  ),
  { name: 'ChatStore' },
  ),
);
