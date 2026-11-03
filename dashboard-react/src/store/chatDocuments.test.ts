import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ChatMessage, ChatUploadedFile } from '../types/chat';
import { hydrateChatDocuments, MAX_CHAT_DOCUMENT_BYTES, pruneChatDocuments, removeChatDocuments, storeChatDocuments } from './chatDocuments';
import { buildApiMessages } from '../components/pages/chatApiPayload';

const savedIds: string[] = [];

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

afterEach(async () => { vi.restoreAllMocks(); await removeChatDocuments(savedIds.splice(0)); });

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
    vi.spyOn(IDBObjectStore.prototype, 'put').mockImplementation(() => { throw new DOMException('Quota exceeded', 'QuotaExceededError'); });
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
    await pruneChatDocuments(new Set([stored[0].documentId!]));
    await expect(hydrateChatDocuments(history([stored[0]]))).resolves.toBeDefined();
    await expect(hydrateChatDocuments(history([stored[1]]))).rejects.toThrow('missing from this browser');
  });
});
