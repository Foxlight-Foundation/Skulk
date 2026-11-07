import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import type { ChatMessage, ChatUploadedFile } from '../types/chat';
import { hydrateChatDocuments, MAX_CHAT_DOCUMENT_BYTES, pruneChatDocuments, removeChatDocuments, storeChatDocuments } from './chatDocuments';
import { buildApiMessages } from '../components/pages/chatApiPayload';
import { chatStorageChanges, readCommittedChatStorage, writeChatStorage } from './chatCoordination';
import { openChatDatabase } from './chatDatabase';
import { isolateChatStorage } from './chatStorageTestScope';

const savedIds: string[] = [];
const conversations: string[] = [];
let restoreScope: () => void;
beforeAll(() => { restoreScope = isolateChatStorage('document-tests:' + crypto.randomUUID()); });
afterAll(() => restoreScope());

function upload(name: string, bytes: BlobPart, type = 'text/plain'): ChatUploadedFile {
  const file = new File([bytes], name, { type });
  return { id: crypto.randomUUID(), name, size: file.size, type, file };
}

async function save(files: ChatUploadedFile[]): Promise<ChatUploadedFile[]> {
  const stored = await storeChatDocuments(files);
  savedIds.push(...stored.flatMap(file => file.documentId ? [file.documentId] : []));
  return stored;
}

function history(files: ChatUploadedFile[]): ChatMessage[] {
  return [{ id: 'message', role: 'user', content: 'Read these.', timestamp: 1,
    attachments: files.map(({ id, name, type, size, documentId }) => ({ id, name, type, size, documentId })) }];
}

async function persist(files: ChatUploadedFile[]): Promise<void> {
  const before = await readCommittedChatStorage(), id = crypto.randomUUID();
  conversations.push(id);
  await writeChatStorage(chatStorageChanges(before, { ...before, conversations: {
    ...(before.conversations as Record<string, unknown>),
    [id]: { id, name: 'Documents', modelId: 'fixture', createdAt: 1, updatedAt: 1, messages: history(files) },
  } }));
}

afterEach(async () => {
  vi.restoreAllMocks(); vi.unstubAllGlobals();
  const before = await readCommittedChatStorage();
  const following = { ...(before.conversations as Record<string, unknown>) };
  for (const id of conversations.splice(0)) delete following[id];
  await writeChatStorage(chatStorageChanges(before, { ...before, conversations: following }));
  await removeChatDocuments(savedIds.splice(0));
});

describe('browser-owned chat documents', () => {
  it('persists an 8 MiB document outside metadata and rehydrates the exact bytes after serialization', async () => {
    const raw = new Uint8Array(MAX_CHAT_DOCUMENT_BYTES).fill(65);
    raw[raw.length - 1] = 90;
    const selected = await save([upload('large.txt', raw)]);
    const metadata = JSON.stringify(history(selected));
    expect(metadata.length).toBeLessThan(512);
    const restored = await hydrateChatDocuments(JSON.parse(metadata) as ChatMessage[]);
    const data = restored[0].attachments?.[0].fileData;
    expect(data?.startsWith('data:text/plain;base64,')).toBe(true);
    const decoded = atob(data!.split(',')[1]);
    expect(decoded.length).toBe(raw.length);
    expect(decoded[0]).toBe('A');
    expect(decoded.at(-1)).toBe('Z');
    expect(buildApiMessages(restored, { text: true, pdf: true })[0]).toMatchObject({ content: [
      { type: 'file', file: { filename: 'large.txt', file_data: data } }, { type: 'text', text: 'Read these.' },
    ] });
  });

  it('keeps UTF-8 text and PDF bytes and media types distinct', async () => {
    const stored = await save([upload('notes.TXT', '狐 MARIGOLD', ''), upload('notes.pdf', '%PDF-1.4\nJUNIPER', 'application/pdf')]);
    const hydrated = await hydrateChatDocuments(history(stored));
    expect(hydrated[0].attachments?.[0].fileData).toBe('data:text/plain;base64,' + btoa(String.fromCharCode(...new TextEncoder().encode('狐 MARIGOLD'))));
    expect(hydrated[0].attachments?.[1].fileData).toBe('data:application/pdf;base64,' + btoa('%PDF-1.4\nJUNIPER'));
  });

  it('saves documents when randomUUID is unavailable on an HTTP dashboard', async () => {
    const selected = upload('notes.txt', 'JUNIPER');
    vi.stubGlobal('crypto', { getRandomValues: crypto.getRandomValues.bind(crypto) });
    const stored = await save([selected]);
    expect(stored[0].documentId).toMatch(/^[a-f0-9]{32}$/);
    expect((await hydrateChatDocuments(history(stored)))[0].attachments?.[0].fileData).toBe('data:text/plain;base64,' + btoa('JUNIPER'));
  });

  it.each([
    ['empty.txt', '', /nonempty/],
    ['invalid.txt', new Uint8Array([255]), /UTF-8/],
    ['null.txt', 'a\0b', /null bytes/],
    ['invalid.pdf', 'not a PDF', /PDF header/],
    ['../notes.txt', 'text', /filename/],
  ])('refuses invalid content before saving: %s', async (name, raw, error) => {
    await expect(storeChatDocuments([upload(String(name), raw as BlobPart)])).rejects.toThrow(error as RegExp);
  });

  it('reports storage failure rather than falling back to unpersisted documents', async () => {
    vi.spyOn(IDBObjectStore.prototype, 'add').mockImplementation(() => { throw new DOMException('Quota exceeded', 'QuotaExceededError'); });
    await expect(storeChatDocuments([upload('notes.txt', 'MARIGOLD')])).rejects.toThrow('Could not save documents');
  });

  it('fails explicitly when a referenced document has been removed', async () => {
    const stored = await save([upload('notes.txt', 'MARIGOLD')]);
    await removeChatDocuments(stored.map(file => file.documentId!));
    await expect(hydrateChatDocuments(history(stored))).rejects.toThrow('missing from this browser');
  });

  it('prunes old abandoned submissions while retaining referenced documents', async () => {
    const clock = vi.spyOn(Date, 'now').mockReturnValue(Date.now() - 2 * 86400000);
    const stored = await save([upload('keep.txt', 'KEEP'), upload('abandoned.txt', 'REMOVE')]);
    clock.mockRestore();
    await persist([stored[0]]);
    await pruneChatDocuments();
    await expect(hydrateChatDocuments(history([stored[0]]))).resolves.toBeDefined();
    await expect(hydrateChatDocuments(history([stored[1]]))).rejects.toThrow('missing from this browser');
  });

  it('keeps an unexpired foreign staging lease without a history reference', async () => {
    const stored = await save([upload('pending.txt', 'FOREIGN')]);
    const database = await openChatDatabase();
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction('documents', 'readwrite');
      const store = transaction.objectStore('documents'), request = store.get(stored[0].documentId!);
      request.onsuccess = () => store.put({ ...request.result, staged: { owner: 'another-renderer', expiresAt: Date.now() + 10000 } }, stored[0].documentId!);
      transaction.oncomplete = () => resolve();
      transaction.onabort = () => reject(new Error('Could not prepare foreign lease'));
    });
    database.close();
    await pruneChatDocuments();
    await removeChatDocuments([stored[0].documentId!]);
    await expect(hydrateChatDocuments(history(stored))).resolves.toBeDefined();
  });

  it('never deletes a reference committed before explicit cleanup', async () => {
    const stored = await save([upload('shared.txt', 'KEEP')]);
    await persist(stored);
    await removeChatDocuments([stored[0].documentId!]);
    await expect(hydrateChatDocuments(history(stored))).resolves.toBeDefined();
  });

  it('refuses history publication if cleanup won the race for staged bytes', async () => {
    const clock = vi.spyOn(Date, 'now').mockReturnValue(Date.now() - 2 * 86400000);
    const stored = await save([upload('expired.txt', 'OLD')]);
    clock.mockRestore();
    await pruneChatDocuments();
    await expect(persist(stored)).rejects.toThrow('document is missing');
    const committed = JSON.stringify((await readCommittedChatStorage()).conversations);
    expect(committed).not.toContain(stored[0].documentId);
  });

  it('serializes competing reference publication and age-qualified collection', async () => {
    for (let attempt = 0; attempt < 8; attempt++) {
      const clock = vi.spyOn(Date, 'now').mockReturnValue(Date.now() - 2 * 86400000);
      const stored = await save([upload('race.txt', 'RACE')]);
      clock.mockRestore();
      const [publication] = await Promise.allSettled([persist(stored), pruneChatDocuments()]);
      const committed = JSON.stringify((await readCommittedChatStorage()).conversations);
      if (publication.status === 'fulfilled') {
        expect(committed).toContain(stored[0].documentId);
        await expect(hydrateChatDocuments(history(stored))).resolves.toBeDefined();
      } else {
        expect(committed).not.toContain(stored[0].documentId);
        expect(String(publication.reason)).toContain('document is missing');
      }
    }
  });

  it('preserves document bytes when canonical reference metadata cannot be interpreted', async () => {
    const clock = vi.spyOn(Date, 'now').mockReturnValue(Date.now() - 2 * 86400000);
    const stored = await save([upload('preserve.txt', 'PRESERVE')]);
    clock.mockRestore();
    const before = await readCommittedChatStorage(), id = crypto.randomUUID();
    conversations.push(id);
    await writeChatStorage(chatStorageChanges(before, { ...before, conversations: { ...(before.conversations as Record<string, unknown>), [id]: { messages: 'unreadable' } } }));
    await expect(pruneChatDocuments()).rejects.toThrow('Existing references were preserved');
    await expect(hydrateChatDocuments(history(stored))).resolves.toBeDefined();
  });
});
