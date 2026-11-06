import { createSlice, nanoid, type PayloadAction } from '@reduxjs/toolkit';
import { DEFAULT_CONVERSATION_NAME, type ChatMessage, type Conversation } from '../../types/chat';
import { chatDocumentIds, removeChatDocuments } from '../chatDocuments';
import type { ChatContinuation } from '../chatContinuation';
import { chatStorageChanges, readChatStorage, readCommittedChatSnapshot, writeChatStorage, flushChatStorage, CHAT_STORAGE_KEY, ChatStorageUnavailable, hasUnsavedChatChanges, type ChatStorageObject } from '../chatCoordination';

/* ── Helpers ──────────────────────────────────────────── */

function autoName(content: string): string {
  const trimmed = content.trim();
  if (!trimmed) return DEFAULT_CONVERSATION_NAME;
  if (trimmed.length <= 50) return trimmed;
  return trimmed.slice(0, 50) + '...';
}

/**
 * Drop transient per-message fields that should not survive a reload (token
 * counters, in-progress generated images). The persistence layer applies this
 * before writing to durable browser storage.
 */
function stripTransientFields(
  conversations: Record<string, Conversation>,
): Record<string, Conversation> {
  const stripped: Record<string, Conversation> = {};
  for (const [id, convo] of Object.entries(conversations)) {
    stripped[id] = {
      ...convo,
      messages: convo.messages.map((msg) => {
        const rest = { ...msg };
        delete rest.tokens;
        delete rest.generatedImages;
        if (rest.attachments) rest.attachments = rest.attachments.map(attachment => {
          const metadata = { ...attachment };
          delete metadata.fileData;
          return metadata;
        });
        return rest;
      }),
    };
  }
  return stripped;
}

/* ── State ────────────────────────────────────────────── */

const SESSION_STORAGE_KEY = 'skulk-chat-session';

export interface ChatState {
  conversations: Record<string, Conversation>;
  /** Visible persistence conflict/failure; no successful save is implied. */
  persistenceError?: string;
  /** Per-tab explicit selection required after a foreign deletion or model reassignment. */
  selectionNeedsReview?: boolean;
  activeConversationId: string | null;
  selectedModelId: string | null;
  selectedTranscriptionModelId: string | null;
  selectedSpeechModelId: string | null;
  selectedVoice: string | null;
  autoSpeakAssistant: boolean;
  /** Speak short interjections while a code block streams (voice loop). */
  narrateCodeBlocks: boolean;
  realtimeVoiceEnabled: boolean;
  autoSubmitVoice: boolean;
  modelToConversationId: Record<string, string>;
}

interface PersistedDurable {
  conversations?: Record<string, Conversation>;
  modelToConversationId?: Record<string, string>;
  selectedVoice?: string | null;
  autoSpeakAssistant?: boolean;
  narrateCodeBlocks?: boolean;
  realtimeVoiceEnabled?: boolean;
  autoSubmitVoice?: boolean;
}

interface PersistedSession {
  selectionNeedsReview?: boolean;
  activeConversationId?: string | null;
  selectedModelId?: string | null;
  selectedTranscriptionModelId?: string | null;
  selectedSpeechModelId?: string | null;
}

function loadDurable(): PersistedDurable {
  if (typeof window === 'undefined') return {};
  try {
    return readChatStorage() as PersistedDurable;
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
    selectionNeedsReview: session.selectionNeedsReview ?? false,
    selectedModelId: session.selectedModelId ?? null,
    selectedTranscriptionModelId: session.selectedTranscriptionModelId ?? null,
    selectedSpeechModelId: session.selectedSpeechModelId ?? null,
    selectedVoice: durable.selectedVoice ?? null,
    autoSpeakAssistant: durable.autoSpeakAssistant ?? false,
    narrateCodeBlocks: durable.narrateCodeBlocks ?? true,
    realtimeVoiceEnabled: durable.realtimeVoiceEnabled ?? true,
    autoSubmitVoice: durable.autoSubmitVoice ?? false,
  };
}

/* ── Slice ────────────────────────────────────────────── */

const slice = createSlice({
  name: 'chat',
  initialState: initialState(),
  reducers: {
    /** Adopt committed browser data while retaining this tab's current selection. */
    refreshDurable(state, action: PayloadAction<PersistedDurable>) {
      const durable = action.payload;
      const conversations = durable.conversations ?? {};
      for (const id of Object.keys(state.conversations)) if (!conversations[id]) delete state.conversations[id];
      for (const [id, conversation] of Object.entries(conversations)) {
        const live = state.conversations[id];
        const durable = live ? stripTransientFields({ [id]: live })[id] : undefined;
        if (!live || JSON.stringify(durable) !== JSON.stringify(conversation)) {
          // Metadata adoption must not erase process-local image URLs, token
          // counters or staged file bytes for unchanged public messages.
          const previousMessages = new Map(live?.messages.map((message, index) => [message.id, { live: message, durable: durable!.messages[index] }]));
          state.conversations[id] = { ...conversation, messages: conversation.messages.map(message => {
            const previous = previousMessages.get(message.id);
            return previous && live?.modelId === conversation.modelId && JSON.stringify(previous.durable) === JSON.stringify(message) ? previous.live : message;
          }) };
        }
      }
      state.modelToConversationId = durable.modelToConversationId ?? {};
      state.selectedVoice = durable.selectedVoice ?? null;
      state.autoSpeakAssistant = durable.autoSpeakAssistant ?? false;
      state.narrateCodeBlocks = durable.narrateCodeBlocks ?? true;
      state.realtimeVoiceEnabled = durable.realtimeVoiceEnabled ?? true;
      state.autoSubmitVoice = durable.autoSubmitVoice ?? false;
      const active = state.activeConversationId ? state.conversations[state.activeConversationId] : undefined;
      if (state.activeConversationId && (!active || active.modelId !== state.selectedModelId)) {
        // Retain this tab's chosen processing destination. Silently following
        // another tab's reassignment could send its draft to a different model.
        state.activeConversationId = null;
        state.selectionNeedsReview = true;
      }
    },
    /** Surface failed custody without losing the saved browser snapshot. */
    setPersistenceError(state, action: PayloadAction<string | undefined>) { state.persistenceError = action.payload; },
    selectModel(state, action: PayloadAction<string>) {
      state.selectionNeedsReview = false;
      const modelId = action.payload;
      const now = Date.now();

      const currentId = state.activeConversationId;
      const currentConvo = currentId ? state.conversations[currentId] : null;

      // Same model + active conversation: no-op.
      if (modelId === state.selectedModelId && currentConvo?.modelId === modelId) return;

      // A conversation ID permanently identifies its processing destination,
      // even while empty: another tab may already be preparing its first send.

      // Save the existing conversation's updatedAt, then switch.
      if (currentConvo) {
        currentConvo.updatedAt = now;
      }

      // Find or create the new model's conversation.
      const existingId = state.modelToConversationId[modelId];
      if (existingId && state.conversations[existingId]?.modelId === modelId) {
        state.activeConversationId = existingId;
        state.selectedModelId = modelId;
      } else {
        const newId = nanoid();
        state.conversations[newId] = {
          id: newId,
          name: DEFAULT_CONVERSATION_NAME,
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

    selectTranscriptionModel(state, action: PayloadAction<string | null>) {
      state.selectedTranscriptionModelId = action.payload;
    },

    selectSpeechModel(state, action: PayloadAction<string | null>) {
      state.selectedSpeechModelId = action.payload;
    },

    setSelectedVoice(state, action: PayloadAction<string | null>) {
      const normalized = action.payload?.trim() || null;
      state.selectedVoice = normalized;
    },

    setAutoSpeakAssistant(state, action: PayloadAction<boolean>) {
      state.autoSpeakAssistant = action.payload;
    },

    setNarrateCodeBlocks(state, action: PayloadAction<boolean>) {
      state.narrateCodeBlocks = action.payload;
    },

    setRealtimeVoiceEnabled(state, action: PayloadAction<boolean>) {
      state.realtimeVoiceEnabled = action.payload;
    },

    setAutoSubmitVoice(state, action: PayloadAction<boolean>) {
      state.autoSubmitVoice = action.payload;
    },

    addMessage(state, action: PayloadAction<ChatMessage>) {
      const convoId = state.activeConversationId;
      if (!convoId) return;
      const convo = state.conversations[convoId];
      if (!convo) return;
      convo.messages.push(action.payload);
      if (convo.name === DEFAULT_CONVERSATION_NAME && action.payload.role === 'user') {
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
      if (target) {
        target.content = action.payload.content;
        delete target.continuationMessages;
      }
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
        state.selectionNeedsReview = false;
        const { id, modelId } = action.payload;
        const now = Date.now();
        state.conversations[id] = {
          id,
          name: DEFAULT_CONVERSATION_NAME,
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

    /** Create a model-owned conversation without moving another chat's selection. */
    createBackgroundConversation: {
      reducer(state, action: PayloadAction<{ id: string; modelId: string }>) {
        const { id, modelId } = action.payload;
        const now = Date.now();
        state.conversations[id] = { id, modelId, name: DEFAULT_CONVERSATION_NAME, createdAt: now, updatedAt: now, messages: [] };
        state.modelToConversationId[modelId] = id;
        if (state.selectedModelId === modelId) state.activeConversationId = id;
      },
      prepare(modelId: string) { return { payload: { id: nanoid(), modelId } }; },
    },

    /** Pin async replies to their originating conversation; deletion never resurrects it. */
    appendConversationMessage(state, action: PayloadAction<{ conversationId: string; message: ChatMessage }>) {
      const conversation = state.conversations[action.payload.conversationId];
      if (!conversation) return;
      conversation.messages.push(action.payload.message);
      if (conversation.name === DEFAULT_CONVERSATION_NAME && action.payload.message.role === 'user') {
        conversation.name = autoName(action.payload.message.content);
      }
      conversation.updatedAt = Date.now();
    },

    /** Pin continuation state to the originating chat, never the current selection. */
    setConversationContinuation(state, action: PayloadAction<{ conversationId: string; continuation?: ChatContinuation }>) {
      const conversation = state.conversations[action.payload.conversationId];
      if (conversation) conversation.continuation = action.payload.continuation;
    },

    /** Publish an acknowledged round and its receipt as one state transition. */
    saveContinuationMessage(state, action: PayloadAction<{ conversationId: string; continuation: ChatContinuation; message: ChatMessage }>) {
      const conversation = state.conversations[action.payload.conversationId];
      if (!conversation) return;
      const index = conversation.messages.findIndex(message => message.id === action.payload.message.id);
      if (index < 0) conversation.messages.push(action.payload.message);
      else conversation.messages[index] = action.payload.message;
      conversation.continuation = action.payload.continuation;
      conversation.updatedAt = action.payload.message.timestamp;
    },

    /** Rename a saved conversation without replacing its messages or selection. */
    renameConversation(state, action: PayloadAction<{ conversationId: string; name: string }>) {
      const conversation = state.conversations[action.payload.conversationId];
      const name = action.payload.name.trim();
      if (conversation && name) { conversation.name = name; conversation.updatedAt = Date.now(); }
    },

    selectConversation(state, action: PayloadAction<string>) {
      const convo = state.conversations[action.payload];
      if (!convo) return;
      state.selectionNeedsReview = false;
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

/** Normalize durable fields before comparison; transient data cannot cause a false conflict. */
function durableChatState(chat: ChatState): ChatStorageObject {
  return JSON.parse(JSON.stringify({
    conversations: stripTransientFields(chat.conversations),
    modelToConversationId: chat.modelToConversationId,
    selectedVoice: chat.selectedVoice,
    autoSpeakAssistant: chat.autoSpeakAssistant,
    narrateCodeBlocks: chat.narrateCodeBlocks,
    realtimeVoiceEnabled: chat.realtimeVoiceEnabled,
    autoSubmitVoice: chat.autoSubmitVoice,
  })) as ChatStorageObject;
}

/** Persist a new message before accepting its draft or starting inference. */
export async function persistDocumentMessage(chat: ChatState, conversationId: string, message: ChatMessage): Promise<void> {
  const conversation = chat.conversations[conversationId];
  if (!conversation) throw new Error('The conversation is no longer available.');
  const next: Conversation = {
    ...conversation, messages: [...conversation.messages, message],
    name: conversation.name === DEFAULT_CONVERSATION_NAME ? autoName(message.content) : conversation.name,
    updatedAt: message.timestamp,
  };
  await writeChatStorage(chatStorageChanges(durableChatState(chat), durableChatState({ ...chat, conversations: { ...chat.conversations, [conversationId]: next } })));
}

/** Atomically acknowledge continuation state and its exact public history. */
export async function persistChatContinuation(chat: ChatState, conversationId: string, continuation?: ChatContinuation, message?: ChatMessage): Promise<void> {
  const conversation = chat.conversations[conversationId];
  if (!conversation) throw new Error('The conversation is no longer available.');
  const messages = message
    ? conversation.messages.some(existing => existing.id === message.id)
      ? conversation.messages.map(existing => existing.id === message.id ? message : existing)
      : [...conversation.messages, message]
    : conversation.messages;
  const changes = chatStorageChanges(durableChatState(chat), durableChatState({ ...chat,
    conversations: { ...chat.conversations, [conversationId]: { ...conversation, continuation, messages } },
  }));
  // Receipt-only acknowledgment must still compare the verified transcript:
  // ordinary message edits can commit without the long-lived writer lock.
  if (continuation?.receipt && !message) changes.push({ path: ['conversations', conversationId, 'messages'], before: messages, after: messages });
  await writeChatStorage(changes);
}

let refreshCommittedChat: (() => Promise<void>) | undefined;
/** Flush queued writes and adopt the latest saved conversations before acquiring new work. */
export async function synchronizeChatPersistence(): Promise<void> {
  await flushChatStorage().catch(() => {});
  await refreshCommittedChat?.();
}

/** Coordinate every ordinary and acknowledged save, preserving unrelated tab changes. */
export function subscribeChatPersistence(
  store: { subscribe: (listener: () => void) => () => void; getState: () => ChatRootState; dispatch: (action: ReturnType<typeof chatActions.refreshDurable> | ReturnType<typeof chatActions.setPersistenceError>) => unknown },
): () => void {
  if (typeof window === 'undefined') return () => {};
  let previous = durableChatState(store.getState().chat);
  let lastSessionJson: string | null = null;
  let adopting = false;
  let localRevision = 0;
  let committedRevision = 0;
  const refresh = async () => {
    // A refresh queued before a newer local action cannot discard that action.
    // Recheck after draining so adoption and its revision share one JS turn.
    let revision: number;
    do {
      revision = localRevision;
      await flushChatStorage().catch(() => {});
    } while (revision !== localRevision);
    if (hasUnsavedChatChanges()) return;
    const revisionBeforeRead = localRevision;
    const snapshot = await readCommittedChatSnapshot();
    if (revisionBeforeRead !== localRevision) return refresh();
    if (snapshot.revision < committedRevision) return;
    committedRevision = snapshot.revision;
    const committed = snapshot.state;
    adopting = true;
    try {
      store.dispatch(chatActions.refreshDurable(committed as PersistedDurable));
      previous = durableChatState(store.getState().chat);
    } finally { adopting = false; }
  };
  refreshCommittedChat = refresh;
  const onStorage = (event: StorageEvent) => {
    if (event.key === CHAT_STORAGE_KEY) void refresh().catch(error => store.dispatch(chatActions.setPersistenceError(String(error))));
  };
  window.addEventListener('storage', onStorage);
  const channel = typeof BroadcastChannel === 'undefined' ? undefined : new BroadcastChannel('skulk-chat-writers-v2');
  if (channel) channel.onmessage = event => { if (event.data === 'committed') void refresh().catch(error => store.dispatch(chatActions.setPersistenceError(String(error)))); };
  void refresh().catch(error => store.dispatch(chatActions.setPersistenceError(String(error))));
  const unsubscribe = store.subscribe(() => {
    const chat = store.getState().chat;
    const current = durableChatState(chat);
    const before = previous;
    const changes = chatStorageChanges(before, current);
    previous = current;
    if (!adopting && changes.length) {
      localRevision++;
      void writeChatStorage(changes, true).then(committed => {
        // Only delete blobs dropped by this successful mutation, after checking
        // the merged snapshot. Never prune another tab's uncommitted uploads.
        const oldReferences = chatDocumentIds((before.conversations ?? {}) as Record<string, Conversation>);
        const retained = chatDocumentIds((committed.conversations ?? {}) as Record<string, Conversation>);
        return removeChatDocuments([...oldReferences].filter(id => !retained.has(id)));
      }).catch(async error => {
        store.dispatch(chatActions.setPersistenceError(error instanceof Error ? error.message : 'Conversation storage is unavailable.'));
        if (!(error instanceof ChatStorageUnavailable)) await refresh().catch(() => {});
      });
    }
    const sessionJson = JSON.stringify({ state: {
      activeConversationId: chat.activeConversationId, selectedModelId: chat.selectedModelId, selectionNeedsReview: chat.selectionNeedsReview,
      selectedTranscriptionModelId: chat.selectedTranscriptionModelId, selectedSpeechModelId: chat.selectedSpeechModelId,
    }, version: 1 });
    if (sessionJson !== lastSessionJson) {
      lastSessionJson = sessionJson;
      try { window.sessionStorage.setItem(SESSION_STORAGE_KEY, sessionJson); } catch { /* Per-tab selection is best effort. */ }
    }
  });
  return () => { unsubscribe(); channel?.close(); window.removeEventListener('storage', onStorage); if (refreshCommittedChat === refresh) refreshCommittedChat = undefined; };
}
