import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { chatStorageChanges, hasUnsavedChatChanges, readCommittedChatStorage, writeChatStorage, type ChatStorageObject } from './chatCoordination';
import { openChatDatabase, putChatSnapshot } from './chatDatabase';
import { readRetainedChatEdits, resolveRetainedChatEdit } from './chatRetainedEdits';
import { isolateChatStorage } from './chatStorageTestScope';
import { hydrateChatDocuments, pruneChatDocuments, storeChatDocuments } from './chatDocuments';

let restoreScope: () => void;
beforeEach(() => { restoreScope = isolateChatStorage('retained-tests:' + crypto.randomUUID()); });
afterEach(async () => { vi.restoreAllMocks(); await writeChatStorage([]).catch(() => {}); restoreScope(); });

function conversation(id: string, content: string) {
  return { id, name: id, modelId: 'fixture', createdAt: 1, updatedAt: 1,
    messages: [{ id: 'message', role: 'user' as const, content, timestamp: 1 }] };
}

async function seed(): Promise<ChatStorageObject> {
  const before = await readCommittedChatStorage();
  const state = { ...before, conversations: { a: conversation('a', 'Original'), b: conversation('b', 'Other') } };
  await writeChatStorage(chatStorageChanges(before, state));
  return state;
}

function changed(state: ChatStorageObject, content: string, other?: string): ChatStorageObject {
  return { ...state, conversations: { ...(state.conversations as Record<string, unknown>), a: conversation('a', content), ...(other ? { b: conversation('b', other) } : {}) } };
}

async function foreignCommit(state: ChatStorageObject): Promise<void> {
  const database = await openChatDatabase();
  try { await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(['snapshot', 'documents'], 'readwrite');
    const store = transaction.objectStore('snapshot'), request = store.get('current');
    request.onsuccess = () => {
      const saved = request.result as { state: ChatStorageObject; revision: number };
      putChatSnapshot(transaction, saved.state, state, saved.revision + 1, () => transaction.abort());
    };
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(transaction.error);
  }); } finally { database.close(); }
}

describe('retained conversation edits', () => {
  it('quarantines local intent while canonical history and an unrelated edit progress', async () => {
    const before = await seed(), foreign = changed(before, 'Saved in another tab'), local = changed(before, 'Local intent', 'Independent edit');
    await foreignCommit(foreign);
    await writeChatStorage(chatStorageChanges(before, local), true, local);
    const state = await readCommittedChatStorage();
    expect((state.conversations as Record<string, unknown>).a).toEqual((foreign.conversations as Record<string, unknown>).a);
    expect((state.conversations as Record<string, unknown>).b).toEqual((local.conversations as Record<string, unknown>).b);
    const retained = await readRetainedChatEdits();
    expect(retained).toHaveLength(1);
    expect(retained[0].conversation).toEqual((local.conversations as Record<string, unknown>).a);
    expect(hasUnsavedChatChanges()).toBe(false);
  });

  it.each(['open', 'transaction'])('retains edits through a %s outage followed by a foreign commit', async stage => {
    const before = await seed(), local = changed(before, 'Local after outage');
    const fail = () => { throw new DOMException('Synthetic storage failure', 'InvalidStateError'); };
    const outage = stage === 'open' ? vi.spyOn(IDBFactory.prototype, 'open').mockImplementation(fail) : vi.spyOn(IDBDatabase.prototype, 'transaction').mockImplementation(fail);
    await expect(writeChatStorage(chatStorageChanges(before, local), true, local)).rejects.toThrow('Synthetic storage failure');
    outage.mockRestore();
    const foreign = changed(before, 'Foreign acknowledged history');
    await foreignCommit(foreign);
    await writeChatStorage([]);
    expect((await readCommittedChatStorage()).conversations).toEqual(foreign.conversations);
    expect((await readRetainedChatEdits())[0].conversation).toEqual((local.conversations as Record<string, unknown>).a);
    expect(hasUnsavedChatChanges()).toBe(false);
  });

  it('keeps unsaved intent when quota refuses quarantine, then recovers after space returns', async () => {
    const before = await seed(), local = changed(before, 'Local quota intent'), foreign = changed(before, 'Foreign history');
    await foreignCommit(foreign);
    const add = IDBObjectStore.prototype.add;
    const quota = vi.spyOn(IDBObjectStore.prototype, 'add').mockImplementation(function (this: IDBObjectStore, value, key) {
      if (this.name === 'recoveries') throw new DOMException('Synthetic quota', 'QuotaExceededError');
      return add.call(this, value, key);
    });
    await expect(writeChatStorage(chatStorageChanges(before, local), true, local)).rejects.toThrow('Keep this tab open');
    expect(hasUnsavedChatChanges()).toBe(true);
    expect(await readRetainedChatEdits()).toEqual([]);
    expect((await readCommittedChatStorage()).conversations).toEqual(foreign.conversations);
    quota.mockRestore();
    await writeChatStorage([]);
    expect((await readRetainedChatEdits())[0].conversation).toEqual((local.conversations as Record<string, unknown>).a);
    expect(hasUnsavedChatChanges()).toBe(false);
  });

  it('forks once with a new identity and strips all attempt and affinity fields', async () => {
    const before = await seed(), local = changed(before, 'Preserve my edit');
    (local.conversations as Record<string, unknown>).a = { ...conversation('a', 'Preserve my edit'),
      originNodeId: 'old-owner', routeAffinity: { task: 'old-task' },
      continuation: { phase: 'pending', request_nonce: 'old-attempt', receipt: { handle: 'old-handle' } },
      messages: [{ ...conversation('a', 'Preserve my edit').messages[0], continuationMessages: [{ tool_call_id: 'old-affinity' }] }],
    };
    const foreign = changed(before, 'Canonical history');
    await foreignCommit(foreign);
    await writeChatStorage(chatStorageChanges(before, local), true, local);
    const [record] = await readRetainedChatEdits();
    const results = await Promise.allSettled([resolveRetainedChatEdit(record.id, 'fork'), resolveRetainedChatEdit(record.id, 'fork')]);
    expect(results.filter(result => result.status === 'fulfilled')).toHaveLength(1);
    const state = await readCommittedChatStorage(), conversations = state.conversations as Record<string, Record<string, unknown>>;
    const fork = Object.values(conversations).find(item => item.id !== 'a' && item.id !== 'b')!;
    expect(fork.id).toMatch(/^[a-f0-9]{32}$/);
    expect(fork.messages).toEqual(conversation('a', 'Preserve my edit').messages);
    expect(fork).not.toHaveProperty('continuation');
    expect(fork).not.toHaveProperty('routeAffinity');
    expect(fork).not.toHaveProperty('originNodeId');
    expect(conversations.a).toEqual((foreign.conversations as Record<string, unknown>).a);
    expect(await readRetainedChatEdits()).toEqual([]);
  });

  it('adopts canonical history explicitly without applying local deletion or edits', async () => {
    const before = await seed(), local = changed(before, 'Discard only after review'), foreign = changed(before, 'Keep saved');
    await foreignCommit(foreign);
    await writeChatStorage(chatStorageChanges(before, local), true, local);
    const [record] = await readRetainedChatEdits();
    await resolveRetainedChatEdit(record.id, 'adopt');
    expect((await readCommittedChatStorage()).conversations).toEqual(foreign.conversations);
    expect(await readRetainedChatEdits()).toEqual([]);
  });

  it('preserves both versions if quota refuses a requested fork', async () => {
    const before = await seed(), local = changed(before, 'Local fork'), foreign = changed(before, 'Canonical');
    await foreignCommit(foreign);
    await writeChatStorage(chatStorageChanges(before, local), true, local);
    const [record] = await readRetainedChatEdits();
    const put = IDBObjectStore.prototype.put;
    const quota = vi.spyOn(IDBObjectStore.prototype, 'put').mockImplementation(function (this: IDBObjectStore, value, key) {
      if (this.name === 'snapshot' && key === 'current') throw new DOMException('Synthetic fork quota', 'QuotaExceededError');
      return put.call(this, value, key);
    });
    await expect(resolveRetainedChatEdit(record.id, 'fork')).rejects.toThrow('Could not save conversation history');
    quota.mockRestore();
    expect((await readCommittedChatStorage()).conversations).toEqual(foreign.conversations);
    expect(await readRetainedChatEdits()).toEqual([record]);
  });

  it('keeps expired document bytes through quarantine and an acknowledged fork', async () => {
    const before = await seed();
    const file = new File(['RETAINED-DOCUMENT'], 'local.txt', { type: 'text/plain' });
    const clock = vi.spyOn(Date, 'now').mockReturnValue(Date.now() - 2 * 86400000);
    const [stored] = await storeChatDocuments([{ id: 'file', name: file.name, type: file.type, size: file.size, file }]);
    clock.mockRestore();
    const message = { ...conversation('a', 'My attached edit').messages[0], attachments: [{ id: stored.id, name: stored.name, type: stored.type, size: stored.size, documentId: stored.documentId }] };
    const local = { ...before, conversations: { ...(before.conversations as Record<string, unknown>), a: { ...conversation('a', 'My attached edit'), messages: [message] } } };
    await foreignCommit(changed(before, 'Canonical history'));
    await writeChatStorage(chatStorageChanges(before, local), true, local);
    await pruneChatDocuments();
    expect((await hydrateChatDocuments([message]))[0].attachments?.[0].fileData).toContain(btoa('RETAINED-DOCUMENT'));
    const [record] = await readRetainedChatEdits();
    const id = await resolveRetainedChatEdit(record.id, 'fork');
    await pruneChatDocuments();
    expect(JSON.stringify((await readCommittedChatStorage()).conversations)).toContain(stored.documentId);
    expect(id).not.toBe('a');
    expect((await hydrateChatDocuments([message]))[0].attachments?.[0].fileData).toContain(btoa('RETAINED-DOCUMENT'));
  });

  it('keeps local intent pending when document expiry cleanup wins before quarantine', async () => {
    const before = await seed();
    const file = new File(['EXPIRED-DOCUMENT'], 'local.txt', { type: 'text/plain' });
    const clock = vi.spyOn(Date, 'now').mockReturnValue(Date.now() - 2 * 86400000);
    const [stored] = await storeChatDocuments([{ id: 'expired', name: file.name, type: file.type, size: file.size, file }]);
    clock.mockRestore();
    const local = { ...before, conversations: { ...(before.conversations as Record<string, unknown>), a: {
      ...conversation('a', 'My unsaved document'), messages: [{ ...conversation('a', '').messages[0], attachments: [stored] }],
    } } };
    await pruneChatDocuments();
    await foreignCommit(changed(before, 'Canonical history'));
    await expect(writeChatStorage(chatStorageChanges(before, local), true, local)).rejects.toThrow('A local conversation document is missing');
    expect(hasUnsavedChatChanges()).toBe(true);
    expect(await readRetainedChatEdits()).toEqual([]);
    expect(JSON.stringify((await readCommittedChatStorage()).conversations)).toContain('Canonical history');
  });
});
