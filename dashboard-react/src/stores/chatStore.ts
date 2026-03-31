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

/** Persisted chat state keyed by conversation id and selected model. */
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

/** Return the currently active conversation, if one is selected. */
export const selectActiveConversation = (state: ChatState): Conversation | null =>
  state.activeConversationId
    ? state.conversations[state.activeConversationId] ?? null
    : null;

/** Return the messages for the active conversation. */
export const selectActiveMessages = (state: ChatState): ChatMessage[] =>
  selectActiveConversation(state)?.messages ?? [];

/** Return all conversations sorted by last update time, newest first. */
export const selectAllConversationsSorted = (state: ChatState): Conversation[] =>
  Object.values(state.conversations).sort((a, b) => b.updatedAt - a.updatedAt);

/** Build a selector that returns conversations for one model, newest first. */
export const selectConversationsForModel = (modelId: string) => (state: ChatState): Conversation[] =>
  Object.values(state.conversations)
    .filter((c) => c.modelId === modelId)
    .sort((a, b) => b.updatedAt - a.updatedAt);

/* ── Store ────────────────────────────────────────────── */

/** Persisted Zustand store that backs the dashboard chat experience. */
export const useChatStore = create<ChatState>()(
  persist(devtools(
    (set, get) => ({
      conversations: {},
      activeConversationId: null,
      selectedModelId: null,
      modelToConversationId: {},

      selectModel: (modelId: string) => {
        const state = get();
        const now = Date.now();
        const conversations = { ...state.conversations };
        const currentConvo = state.activeConversationId
          ? conversations[state.activeConversationId]
          : null;

        // If same model, no switch needed
        if (modelId === state.selectedModelId && state.activeConversationId) {
          return;
        }

        // If current conversation is empty, just re-assign it to the new model
        // instead of creating a new one
        if (currentConvo && currentConvo.messages.length === 0) {
          const updatedConvo = { ...currentConvo, modelId, updatedAt: now };
          conversations[currentConvo.id] = updatedConvo;

          // Update model mapping
          const modelMap = { ...state.modelToConversationId };
          // Remove old model's pointer if it pointed here
          if (modelMap[currentConvo.modelId] === currentConvo.id) {
            delete modelMap[currentConvo.modelId];
          }
          modelMap[modelId] = currentConvo.id;

          set({
            conversations,
            selectedModelId: modelId,
            modelToConversationId: modelMap,
          });
          return;
        }

        // Current conversation has messages — save it and switch
        if (currentConvo) {
          conversations[currentConvo.id] = {
            ...currentConvo,
            updatedAt: now,
          };
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

        set({
          conversations: {
            ...state.conversations,
            [conversationId]: { ...convo, summary, updatedAt: Date.now() },
          },
        });
      },
    }),
    { name: 'ChatStore', store: 'ChatStore', enabled: true },
  ),
  {
    name: 'skulk-chat',
    version: 1,
    storage: createJSONStorage(() => ({
      getItem: (name: string) => {
        const durable = localStorage.getItem(name);
        const session = sessionStorage.getItem(name + '-session');
        let d: Record<string, unknown> = {};
        let s: Record<string, unknown> = {};
        try { if (durable) d = JSON.parse(durable); } catch { /* corrupted — reset */ }
        try { if (session) s = JSON.parse(session); } catch { /* corrupted — reset */ }
        return JSON.stringify({
          state: { ...((d as Record<string, unknown>).state ?? {}), ...((s as Record<string, unknown>).state ?? {}) },
          version: (d as Record<string, unknown>).version ?? (s as Record<string, unknown>).version ?? 1,
        });
      },
      setItem: (name: string, value: string) => {
        let parsed: Record<string, unknown>;
        try { parsed = JSON.parse(value); } catch { return; }
        const state = (parsed.state ?? {}) as Record<string, unknown>;
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
);
