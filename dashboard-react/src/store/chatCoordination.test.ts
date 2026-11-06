import { describe, expect, it, vi } from 'vitest';
import { applyChatStorageChanges, chatStorageChanges, writeChatStorage, readCommittedChatStorage, CHAT_STORAGE_KEY, type ChatStorageObject } from './chatCoordination';

const base: ChatStorageObject = {
  conversations: { a: { id: 'a', name: 'One', messages: [], continuation: { phase: 'ready', nonce: '1' }, updatedAt: 1 }, b: { id: 'b', name: 'Two', messages: [], updatedAt: 1 } },
  modelToConversationId: { model: 'a' }, selectedVoice: null,
};
const copy = () => structuredClone(base) as { conversations: Record<string, { id: string; name: string; messages: unknown[]; continuation?: { phase: string; nonce: string }; updatedAt: number }>; modelToConversationId: Record<string, string>; selectedVoice: string | null };

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
});
