import { createSlice, nanoid, type PayloadAction } from '@reduxjs/toolkit';
import type { ChatMessage, Conversation } from '../../types/chat';

/* ── Helpers ──────────────────────────────────────────── */

function autoName(content: string): string {
  const trimmed = content.trim();
  if (!trimmed) return 'New conversation';
  if (trimmed.length <= 50) return trimmed;
  return trimmed.slice(0, 50) + '...';
}

/**
 * Drop transient per-message fields that should not survive a reload (token
 * counters, in-progress generated images). The persistence layer applies this
 * before writing to localStorage.
 */
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

/* ── State ────────────────────────────────────────────── */

const DURABLE_STORAGE_KEY = 'skulk-chat';
const SESSION_STORAGE_KEY = 'skulk-chat-session';

export interface ChatState {
  conversations: Record<string, Conversation>;
  activeConversationId: string | null;
  selectedModelId: string | null;
  modelToConversationId: Record<string, string>;
}

interface PersistedDurable {
  conversations?: Record<string, Conversation>;
  modelToConversationId?: Record<string, string>;
}

interface PersistedSession {
  activeConversationId?: string | null;
  selectedModelId?: string | null;
}

function loadDurable(): PersistedDurable {
  if (typeof window === 'undefined') return {};
  try {
    const raw = window.localStorage.getItem(DURABLE_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as { state?: PersistedDurable };
    return parsed.state ?? {};
  } catch {
    return {};
  }
}

function loadSession(): PersistedSession {
  if (typeof window === 'undefined') return {};
  try {
    const raw = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as { state?: PersistedSession };
    return parsed.state ?? {};
  } catch {
    return {};
  }
}

function initialState(): ChatState {
  const durable = loadDurable();
  const session = loadSession();
  return {
    conversations: durable.conversations ?? {},
    modelToConversationId: durable.modelToConversationId ?? {},
    activeConversationId: session.activeConversationId ?? null,
    selectedModelId: session.selectedModelId ?? null,
  };
}

/* ── Slice ────────────────────────────────────────────── */

const slice = createSlice({
  name: 'chat',
  initialState: initialState(),
  reducers: {
    selectModel(state, action: PayloadAction<string>) {
      const modelId = action.payload;
      const now = Date.now();

      const currentId = state.activeConversationId;
      const currentConvo = currentId ? state.conversations[currentId] : null;

      // Same model + active conversation: no-op.
      if (modelId === state.selectedModelId && currentId) return;

      // Empty current conversation — re-assign it to the new model rather
      // than leaving an unused conversation behind.
      if (currentConvo && currentConvo.messages.length === 0) {
        if (state.modelToConversationId[currentConvo.modelId] === currentConvo.id) {
          delete state.modelToConversationId[currentConvo.modelId];
        }
        currentConvo.modelId = modelId;
        currentConvo.updatedAt = now;
        state.modelToConversationId[modelId] = currentConvo.id;
        state.selectedModelId = modelId;
        return;
      }

      // Save the existing conversation's updatedAt, then switch.
      if (currentConvo) {
        currentConvo.updatedAt = now;
      }

      // Find or create the new model's conversation.
      const existingId = state.modelToConversationId[modelId];
      if (existingId && state.conversations[existingId]) {
        state.activeConversationId = existingId;
        state.selectedModelId = modelId;
      } else {
        const newId = nanoid();
        state.conversations[newId] = {
          id: newId,
          name: 'New conversation',
          modelId,
          createdAt: now,
          updatedAt: now,
          messages: [],
        };
        state.activeConversationId = newId;
        state.selectedModelId = modelId;
        state.modelToConversationId[modelId] = newId;
      }
    },

    addMessage(state, action: PayloadAction<ChatMessage>) {
      const convoId = state.activeConversationId;
      if (!convoId) return;
      const convo = state.conversations[convoId];
      if (!convo) return;
      convo.messages.push(action.payload);
      if (convo.name === 'New conversation' && action.payload.role === 'user') {
        convo.name = autoName(action.payload.content);
      }
      convo.updatedAt = Date.now();
    },

    deleteMessage(state, action: PayloadAction<string>) {
      const convoId = state.activeConversationId;
      if (!convoId) return;
      const convo = state.conversations[convoId];
      if (!convo) return;
      convo.messages = convo.messages.filter((m) => m.id !== action.payload);
      convo.updatedAt = Date.now();
    },

    editMessage(
      state,
      action: PayloadAction<{ messageId: string; content: string }>,
    ) {
      const convoId = state.activeConversationId;
      if (!convoId) return;
      const convo = state.conversations[convoId];
      if (!convo) return;
      const target = convo.messages.find((m) => m.id === action.payload.messageId);
      if (target) target.content = action.payload.content;
      convo.updatedAt = Date.now();
    },

    removeLastAssistantMessages(state) {
      const convoId = state.activeConversationId;
      if (!convoId) return;
      const convo = state.conversations[convoId];
      if (!convo) return;
      while (
        convo.messages.length > 0 &&
        convo.messages[convo.messages.length - 1].role === 'assistant'
      ) {
        convo.messages.pop();
      }
      convo.updatedAt = Date.now();
    },

    /**
     * Create a new conversation for `modelId`. The id is generated in the
     * `prepare` callback so the dispatcher's caller can recover it from the
     * dispatched action's payload (`dispatch(...).payload.id`) when needed —
     * matches the pre-RTK ergonomics where `newConversation(modelId)` returned
     * the id directly.
     */
    newConversation: {
      reducer(state, action: PayloadAction<{ id: string; modelId: string }>) {
        const { id, modelId } = action.payload;
        const now = Date.now();
        state.conversations[id] = {
          id,
          name: 'New conversation',
          modelId,
          createdAt: now,
          updatedAt: now,
          messages: [],
        };
        state.activeConversationId = id;
        state.selectedModelId = modelId;
        state.modelToConversationId[modelId] = id;
      },
      prepare(modelId: string) {
        return { payload: { id: nanoid(), modelId } };
      },
    },

    selectConversation(state, action: PayloadAction<string>) {
      const convo = state.conversations[action.payload];
      if (!convo) return;
      state.activeConversationId = action.payload;
      state.selectedModelId = convo.modelId;
      state.modelToConversationId[convo.modelId] = action.payload;
    },

    deleteConversation(state, action: PayloadAction<string>) {
      const convo = state.conversations[action.payload];
      if (!convo) return;
      delete state.conversations[action.payload];
      if (state.modelToConversationId[convo.modelId] === action.payload) {
        delete state.modelToConversationId[convo.modelId];
      }
      if (state.activeConversationId === action.payload) {
        state.activeConversationId = null;
        state.selectedModelId = null;
      }
    },

    setSummary(
      state,
      action: PayloadAction<{ conversationId: string; summary: string }>,
    ) {
      const convo = state.conversations[action.payload.conversationId];
      if (!convo) return;
      convo.summary = action.payload.summary;
      convo.updatedAt = Date.now();
    },
  },
});

export const chatSliceReducer = slice.reducer;
export const chatActions = slice.actions;

/* ── Selectors ────────────────────────────────────────── */

export interface ChatRootState {
  chat: ChatState;
}

export const selectActiveConversation = (state: ChatRootState): Conversation | null =>
  state.chat.activeConversationId
    ? state.chat.conversations[state.chat.activeConversationId] ?? null
    : null;

export const selectActiveMessages = (state: ChatRootState): ChatMessage[] =>
  selectActiveConversation(state)?.messages ?? [];

export const selectAllConversationsSorted = (state: ChatRootState): Conversation[] =>
  Object.values(state.chat.conversations).sort((a, b) => b.updatedAt - a.updatedAt);

export const selectConversationsForModel =
  (modelId: string) =>
  (state: ChatRootState): Conversation[] =>
    Object.values(state.chat.conversations)
      .filter((c) => c.modelId === modelId)
      .sort((a, b) => b.updatedAt - a.updatedAt);

/* ── Persistence ──────────────────────────────────────── */

/**
 * Wires localStorage (durable) and sessionStorage (per-tab) writes to the chat
 * slice. Strips transient message fields before writing to localStorage so
 * token counters and in-progress images don't bloat persisted state.
 *
 * Wraps writes with a dirty check so the subscriber doesn't hammer storage on
 * dispatches that don't affect chat state.
 */
export function subscribeChatPersistence(
  store: { subscribe: (listener: () => void) => () => void; getState: () => ChatRootState },
): () => void {
  if (typeof window === 'undefined') return () => {};
  let lastDurableJson: string | null = null;
  let lastSessionJson: string | null = null;

  return store.subscribe(() => {
    const chat = store.getState().chat;

    const durable = {
      state: {
        conversations: stripTransientFields(chat.conversations),
        modelToConversationId: chat.modelToConversationId,
      },
      version: 1,
    };
    const durableJson = JSON.stringify(durable);
    if (durableJson !== lastDurableJson) {
      lastDurableJson = durableJson;
      try {
        window.localStorage.setItem(DURABLE_STORAGE_KEY, durableJson);
      } catch {
        /* ignore */
      }
    }

    const session = {
      state: {
        activeConversationId: chat.activeConversationId,
        selectedModelId: chat.selectedModelId,
      },
      version: 1,
    };
    const sessionJson = JSON.stringify(session);
    if (sessionJson !== lastSessionJson) {
      lastSessionJson = sessionJson;
      try {
        window.sessionStorage.setItem(SESSION_STORAGE_KEY, sessionJson);
      } catch {
        /* ignore */
      }
    }
  });
}
