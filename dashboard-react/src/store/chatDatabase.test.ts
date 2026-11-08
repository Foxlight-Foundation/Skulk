import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { openChatDatabase } from './chatDatabase';
import { isolateChatStorage } from './chatStorageTestScope';

let restoreScope: () => void;
beforeEach(() => { restoreScope = isolateChatStorage('migration-tests:' + crypto.randomUUID()); });
afterEach(() => { vi.restoreAllMocks(); restoreScope(); });

async function legacyDocuments(): Promise<void> {
  const database = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open('skulk-chat-documents-v1', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('documents');
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  try { await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction('documents', 'readwrite');
    for (const id of ['a', 'b']) transaction.objectStore('documents').add({ blob: new Blob(['LEGACY-' + id], { type: 'text/plain' }), createdAt: 1 }, id.repeat(32));
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(transaction.error);
  }); } finally { database.close(); }
}

async function values(database: IDBDatabase, store: string): Promise<unknown[]> {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(store), request = transaction.objectStore(store).getAll();
    let result: unknown[];
    request.onsuccess = () => { result = request.result as unknown[]; };
    transaction.oncomplete = () => resolve(result);
    transaction.onabort = () => reject(transaction.error);
  });
}

describe('transactional chat document migration', () => {
  it('preserves canonical history and every legacy blob before retiring old storage', async () => {
    await legacyDocuments();
    const state = { conversations: { legacy: { id: 'legacy', modelId: 'fixture', messages: [{ attachments: [{ documentId: 'a'.repeat(32) }] }] } } };
    localStorage.setItem('skulk-chat', JSON.stringify({ state }));
    const database = await openChatDatabase();
    try {
      expect(await values(database, 'snapshot')).toContainEqual({ state, version: 2, revision: 0 });
      const records = await values(database, 'documents') as { blob: Blob; staged: { owner: string } }[];
      expect(await Promise.all(records.map(record => record.blob.text()))).toEqual(['LEGACY-a', 'LEGACY-b']);
      expect(records.every(record => record.staged.owner === 'legacy')).toBe(true);
    } finally { database.close(); }
  });

  it('resumes after quota refusal without deleting old data or restarting the cursor', async () => {
    await legacyDocuments();
    const add = IDBObjectStore.prototype.add;
    const quota = vi.spyOn(IDBObjectStore.prototype, 'add').mockImplementation(function (this: IDBObjectStore, value, key) {
      if (key === 'b'.repeat(32)) throw new DOMException('Synthetic quota', 'QuotaExceededError');
      return add.call(this, value, key);
    });
    await expect(openChatDatabase()).rejects.toThrow('Existing storage was preserved');
    quota.mockRestore();
    const old = await new Promise<IDBDatabase>(resolve => {
      const request = indexedDB.open('skulk-chat-documents-v1', 1);
      request.onsuccess = () => resolve(request.result);
    });
    try { expect(await values(old, 'documents')).toHaveLength(2); } finally { old.close(); }
    const database = await openChatDatabase();
    try { expect(await values(database, 'documents')).toHaveLength(2); } finally { database.close(); }
  });

  it('coordinates simultaneous migration and does not reimport a collected record', async () => {
    await legacyDocuments();
    const databases = await Promise.all([openChatDatabase(), openChatDatabase()]);
    try {
      expect(await values(databases[0], 'documents')).toHaveLength(2);
      await new Promise<void>((resolve, reject) => {
        const transaction = databases[0].transaction('documents', 'readwrite');
        transaction.objectStore('documents').delete('b'.repeat(32));
        transaction.oncomplete = () => resolve();
        transaction.onabort = () => reject(transaction.error);
      });
    } finally { databases.forEach(database => database.close()); }
    const reopened = await openChatDatabase();
    try { expect(await values(reopened, 'documents')).toHaveLength(1); } finally { reopened.close(); }
  });
});
