import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { applyChatStorageChanges, chatStorageChanges, writeChatStorage, readCommittedChatStorage, hasUnsavedChatChanges, acquireChatConversation, CHAT_STORAGE_KEY, type ChatStorageObject } from './chatCoordination';

const base: ChatStorageObject = {
  conversations: { a: { id: 'a', name: 'One', messages: [], continuation: { phase: 'ready', nonce: '1' }, updatedAt: 1 }, b: { id: 'b', name: 'Two', messages: [], updatedAt: 1 } },
  modelToConversationId: { model: 'a' }, selectedVoice: null,
};
const copy = () => structuredClone(base) as { conversations: Record<string, { id: string; name: string; messages: unknown[]; continuation?: { phase: string; nonce: string }; updatedAt: number }>; modelToConversationId: Record<string, string>; selectedVoice: string | null };

const openDatabase = IDBFactory.prototype.open, getStorageItem = Storage.prototype.getItem, setStorageItem = Storage.prototype.setItem;
const isolatedKey = (key: string) => key === CHAT_STORAGE_KEY || key === 'skulk-chat' ? 'coordination-test:' + key : key;
beforeAll(() => {
  // Browser suites share an origin. Opaque CAS fixtures are intentionally not
  // ChatState records and must never become another suite's startup history.
  IDBFactory.prototype.open = function (name, version) { return openDatabase.call(this, name === 'skulk-chat-history-v2' ? name + '-coordination-test' : name, version); };
  Storage.prototype.getItem = function (key) { return getStorageItem.call(this, isolatedKey(key)); };
  Storage.prototype.setItem = function (key, value) { setStorageItem.call(this, isolatedKey(key), value); };
});
afterAll(() => {
  IDBFactory.prototype.open = openDatabase;
  Storage.prototype.getItem = getStorageItem;
  Storage.prototype.setItem = setStorageItem;
});

describe('transactional chat snapshot merging', () => {
  it('preserves a newer nonce when an ordinary stale save changes another conversation', () => {
    const stale = copy(); stale.conversations.b.name = 'Edited elsewhere';
    const current = copy(); current.conversations.a.continuation = { phase: 'pending', nonce: '2' };
    const merged = applyChatStorageChanges(current, chatStorageChanges(base, stale));
    expect(merged.conversations).toEqual({ ...current.conversations, b: stale.conversations.b });
  });
  it('refuses conflicting public history and receipt as one transaction', () => {
    const stale = copy(); stale.conversations.a.messages = ['old answer']; stale.conversations.a.continuation = { phase: 'ready', nonce: 'old' };
    const current = copy(); current.conversations.a.messages = ['new answer']; current.conversations.a.continuation = { phase: 'pending', nonce: 'new' };
    expect(() => applyChatStorageChanges(current, chatStorageChanges(base, stale))).toThrow('changed in another tab');
    expect(current.conversations.a.messages).toEqual(['new answer']);
    expect(current.conversations.a.continuation?.nonce).toBe('new');
  });
  it('merges nonconflicting metadata without rolling back an acknowledged round', () => {
    const renamed = copy(); renamed.conversations.a.name = 'Renamed'; renamed.conversations.a.updatedAt = 4;
    const current = copy(); current.conversations.a.messages = ['committed answer']; current.conversations.a.updatedAt = 5;
    const result = applyChatStorageChanges(current, chatStorageChanges(base, renamed));
    expect(result.conversations).toEqual({ ...current.conversations, a: { ...current.conversations.a, name: 'Renamed' } });
  });
  it('accepts the ordinary subscriber acknowledging an already durable explicit write', () => {
    const next = copy(); next.conversations.a.continuation = { phase: 'pending', nonce: '2' };
    expect(applyChatStorageChanges(next, chatStorageChanges(base, next))).toEqual(next);
  });
  it('refuses a stale deletion after another tab has appended a committed answer', () => {
    const deleted = copy(); delete deleted.conversations.a;
    const current = copy(); current.conversations.a.messages = ['answer'];
    expect(() => applyChatStorageChanges(current, chatStorageChanges(base, deleted))).toThrow('changed in another tab');
  });
  it('never recreates a deleted conversation through a stale timestamp update', () => {
    const stale = copy(); stale.conversations.a.updatedAt = 10;
    const deleted = copy(); delete deleted.conversations.a;
    expect(() => applyChatStorageChanges(deleted, chatStorageChanges(base, stale))).toThrow('deleted in another tab');
  });
  it('preserves independently changed model associations and preferences', () => {
    const stale = copy(); stale.selectedVoice = 'voice';
    const current = copy(); current.modelToConversationId.other = 'b';
    const result = applyChatStorageChanges(current, chatStorageChanges(base, stale));
    expect(result.modelToConversationId).toEqual(current.modelToConversationId); expect(result.selectedVoice).toBe('voice');
  });
});


describe('authoritative browser transactions', () => {
  it('ignores a stale renderer mirror when an ordinary save follows a pending nonce', async () => {
    const id = crypto.randomUUID(), otherId = crypto.randomUUID();
    const before = await readCommittedChatStorage();
    const originalConversation = { id, messages: [], continuation: { phase: 'ready', nonce: 'old' } };
    const seed = { ...before, conversations: { ...(before.conversations as Record<string, unknown>), [id]: originalConversation, [otherId]: { id: otherId, name: 'Other' } } };
    await writeChatStorage(chatStorageChanges(before, seed));
    const pending = { ...seed, conversations: { ...seed.conversations, [id]: { ...originalConversation, continuation: { phase: 'pending', nonce: 'new' } } } };
    await writeChatStorage(chatStorageChanges(seed, pending));
    // Another renderer's startup mirror can lag the canonical transaction.
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify({ state: seed, version: 2 }));
    const staleEdit = { ...seed, conversations: { ...seed.conversations, [otherId]: { id: otherId, name: 'Renamed' } } };
    await writeChatStorage(chatStorageChanges(seed, staleEdit));
    const committed = await readCommittedChatStorage();
    expect((committed.conversations as Record<string, unknown>)[id]).toEqual(pending.conversations[id]);
    expect((committed.conversations as Record<string, unknown>)[otherId]).toEqual(staleEdit.conversations[otherId]);
  });

  it('acknowledges canonical custody when the optional startup mirror is full', async () => {
    const before = await readCommittedChatStorage(), id = crypto.randomUUID();
    const next = { ...before, conversations: { ...(before.conversations as Record<string, unknown>), [id]: { id, messages: ['saved in the transaction'] } } };
    const original = Storage.prototype.setItem;
    const mirror = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(function (this: Storage, key, value) {
      if (key === CHAT_STORAGE_KEY) throw new DOMException('Synthetic mirror quota', 'QuotaExceededError');
      original.call(this, key, value);
    });
    try {
      await writeChatStorage(chatStorageChanges(before, next));
      expect((await readCommittedChatStorage()).conversations).toEqual(next.conversations);
    } finally { mirror.mockRestore(); }
  });
  it('refuses a stale model reassignment when another tab enrolled the empty context', () => {
    const before = { conversations: { a: { id: 'a', modelId: 'model-a', messages: [] } } };
    const next = { conversations: { a: { ...before.conversations.a, modelId: 'model-b' } } };
    const enrolled = { conversations: { a: { ...before.conversations.a, continuation: { model: 'model-a', phase: 'ready' } } } };
    expect(() => applyChatStorageChanges(enrolled, chatStorageChanges(before, next))).toThrow('changed in another tab');
    expect(enrolled.conversations.a.modelId).toBe('model-a');
  });

  it('refuses a stale field append after the context was reassigned to another model', () => {
    const before = { conversations: { a: { id: 'a', modelId: 'model-a', messages: [], updatedAt: 1 } } };
    const appended = { conversations: { a: { ...before.conversations.a, messages: ['model-a prompt'], updatedAt: 3 } } };
    const reassigned = { conversations: { a: { ...before.conversations.a, modelId: 'model-b' } } };
    expect(() => applyChatStorageChanges(reassigned, chatStorageChanges(before, appended))).toThrow('changed in another tab');
    expect(reassigned.conversations.a.messages).toEqual([]);
    const timestampOnly = { conversations: { a: { ...before.conversations.a, updatedAt: 3 } } };
    expect(applyChatStorageChanges(reassigned, chatStorageChanges(before, timestampOnly)).conversations).toEqual({ a: { ...reassigned.conversations.a, updatedAt: 3 } });
  });

  it.each(['open', 'transaction'])('retains ordered ordinary edits across %s failures and commits them after recovery', async failureAt => {
    const before = await readCommittedChatStorage(), id = crypto.randomUUID();
    const created = { ...before, conversations: { ...(before.conversations as Record<string, unknown>), [id]: { id, name: 'New', messages: [] } } };
    const edited = { ...created, conversations: { ...created.conversations, [id]: { id, name: 'Named', messages: ['local history'] } } };
    const fail = () => { throw new DOMException('Synthetic database outage', 'InvalidStateError'); };
    const unavailable = failureAt === 'open' ? vi.spyOn(IDBFactory.prototype, 'open').mockImplementation(fail) : vi.spyOn(IDBDatabase.prototype, 'transaction').mockImplementation(fail);
    try {
      await expect(writeChatStorage(chatStorageChanges(before, created), true)).rejects.toThrow('Synthetic database outage');
      await expect(writeChatStorage(chatStorageChanges(created, edited), true)).rejects.toThrow('Synthetic database outage');
      expect(hasUnsavedChatChanges()).toBe(true);
    } finally { unavailable.mockRestore(); }
    await writeChatStorage([]);
    expect((await readCommittedChatStorage()).conversations).toEqual(edited.conversations);
    expect(hasUnsavedChatChanges()).toBe(false);
  });

  it('preserves ordinary transactional storage and local ownership without secure-context Web Locks', async () => {
    const original = Object.getOwnPropertyDescriptor(navigator, 'locks');
    Object.defineProperty(navigator, 'locks', { configurable: true, value: undefined });
    try {
      const before = await readCommittedChatStorage(), id = crypto.randomUUID();
      const next = { ...before, conversations: { ...(before.conversations as Record<string, unknown>), [id]: { id, messages: ['ordinary HTTP chat'] } } };
      const release = await acquireChatConversation(id);
      await writeChatStorage(chatStorageChanges(before, next));
      await release();
      expect((await readCommittedChatStorage()).conversations).toEqual(next.conversations);
    } finally {
      if (original) Object.defineProperty(navigator, 'locks', original);
      else Reflect.deleteProperty(navigator, 'locks');
    }
  });

});
