import type { ChatMessage, ChatUploadedFile, Conversation } from '../types/chat';
import { randomHex32 } from '../utils/randomIds';

/** Transport limits shared by the versioned document chat contract. */
export const MAX_CHAT_DOCUMENT_BYTES = 8 * 1024 * 1024;
export const MAX_CHAT_DOCUMENTS = 8;
export const MAX_CHAT_REQUEST_BYTES = 16 * 1024 * 1024;
const DATABASE = 'skulk-chat-documents-v1';
const DOCUMENTS = 'documents';
let database: Promise<IDBDatabase> | undefined;

function openDocuments(): Promise<IDBDatabase> {
  if (!database) {
    database = new Promise<IDBDatabase>((resolve, reject) => {
      let failed = false;
      const request = indexedDB.open(DATABASE, 1);
      request.onupgradeneeded = () => { request.result.createObjectStore(DOCUMENTS); };
      request.onerror = () => { failed = true; reject(new Error('Document storage is unavailable in this browser.')); };
      request.onblocked = () => { failed = true; reject(new Error('Document storage is blocked by another browser tab.')); };
      request.onsuccess = () => {
        if (failed) { request.result.close(); return; }
        request.result.onversionchange = () => { request.result.close(); database = undefined; };
        resolve(request.result);
      };
    }).catch((error: unknown) => { database = undefined; throw error; });
  }
  return database;
}

/** Identify the supported document extension without treating MIME alone as proof. */
export function chatDocumentKind(name: string): 'text' | 'pdf' | null {
  if (name.toLowerCase().endsWith('.txt')) return 'text';
  if (name.toLowerCase().endsWith('.pdf')) return 'pdf';
  return null;
}

/** Refuse invalid names and transport sizes before reading browser file bytes. */
export function validateChatDocument(file: Pick<File, 'name' | 'size'>): void {
  if (!/^[^/\\\x00-\x1f\x7f]{1,128}$/.test(file.name) || !chatDocumentKind(file.name)) {
    throw new Error('Choose a .txt or PDF document with a filename of at most 128 characters.');
  }
  if (file.size < 1 || file.size > MAX_CHAT_DOCUMENT_BYTES) {
    throw new Error('Documents must be nonempty and at most 8 MiB each.');
  }
}

/** Validate and durably save only explicitly submitted documents, atomically. */
export async function storeChatDocuments(files: ChatUploadedFile[]): Promise<ChatUploadedFile[]> {
  const documents = files.filter(file => !file.type.startsWith('image/'));
  if (documents.length === 0) return files;
  if (documents.length > MAX_CHAT_DOCUMENTS || documents.reduce((total, file) => total + 4 * Math.ceil(file.size / 3), 0) > MAX_CHAT_REQUEST_BYTES) {
    throw new Error('Attachments exceed the chat request transport limits.');
  }
  const prepared = await Promise.all(documents.map(async (file) => {
    validateChatDocument(file);
    if (!file.file) throw new Error(`Document ${file.name} must be selected again.`);
    const bytes = await file.file.arrayBuffer();
    if (bytes.byteLength !== file.size) throw new Error(`Document ${file.name} could not be read completely.`);
    const kind = chatDocumentKind(file.name);
    if (kind === 'text') {
      let text: string;
      try { text = new TextDecoder('utf-8', { fatal: true }).decode(bytes); }
      catch { throw new Error('Text documents must contain valid UTF-8 text.'); }
      if (text.includes('\0')) throw new Error('Text documents must contain UTF-8 text without null bytes.');
    } else if (new TextDecoder().decode(bytes.slice(0, 5)) !== '%PDF-') {
      throw new Error('The selected PDF does not have a valid PDF header.');
    }
    const blob = new Blob([bytes], { type: kind === 'text' ? 'text/plain' : 'application/pdf' });
    return { file, id: randomHex32(), blob };
  }));
  const db = await openDocuments();
  await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(DOCUMENTS, 'readwrite');
    transaction.oncomplete = () => resolve();
    transaction.onabort = transaction.onerror = () => reject(new Error('Could not save documents in this browser. Free browser storage and try again.'));
    try {
      for (const { id, blob } of prepared) transaction.objectStore(DOCUMENTS).put({ blob, createdAt: Date.now() }, id);
    } catch {
      transaction.abort();
      reject(new Error('Could not save documents in this browser. Free browser storage and try again.'));
    }
  });
  return files.map(file => {
    const document = prepared.find(item => item.file === file);
    return document ? { ...file, documentId: document.id, type: document.blob.type } : file;
  });
}

async function documentData(id: string): Promise<string> {
  const db = await openDocuments();
  const blob = await new Promise<Blob>((resolve, reject) => {
    const transaction = db.transaction(DOCUMENTS, 'readonly');
    const request = transaction.objectStore(DOCUMENTS).get(id);
    request.onerror = transaction.onabort = () => reject(new Error('Could not read a conversation document.'));
    request.onsuccess = () => {
      const value: unknown = request.result;
      if (!value || typeof value !== 'object' || !('blob' in value) || !(value.blob instanceof Blob)) {
        reject(new Error('A conversation document is missing from this browser. Reattach it or remove its message.'));
      } else if (value.blob.size < 1 || value.blob.size > MAX_CHAT_DOCUMENT_BYTES || !['text/plain', 'application/pdf'].includes(value.blob.type)) {
        reject(new Error('A stored document exceeds the supported format or size.'));
      } else resolve(value.blob);
    };
  });
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = reader.onabort = () => reject(new Error('Could not encode a conversation document.'));
    reader.onload = () => typeof reader.result === 'string' ? resolve(reader.result) : reject(new Error('Invalid conversation document.'));
    reader.readAsDataURL(blob);
  });
}

/** Hydrate exact document bytes for one request; never persist the inline copies. */
export async function hydrateChatDocuments(messages: ChatMessage[]): Promise<ChatMessage[]> {
  const documents = messages.flatMap(message => message.attachments ?? []).filter(attachment => attachment.documentId);
  if (documents.length > MAX_CHAT_DOCUMENTS || documents.reduce((total, file) => total + 4 * Math.ceil((file.size ?? 0) / 3), 0) > MAX_CHAT_REQUEST_BYTES) {
    throw new Error('Conversation documents exceed the chat request transport limits. Remove older attachments or start a new conversation.');
  }
  return await Promise.all(messages.map(async message => ({
    ...message,
    attachments: message.attachments ? await Promise.all(message.attachments.map(async attachment => attachment.documentId
      ? { ...attachment, fileData: await documentData(attachment.documentId) }
      : attachment)) : undefined,
  })));
}

/** Collect referenced document handles without reading or retaining their bytes. */
export function chatDocumentIds(conversations: Record<string, Conversation>): Set<string> {
  return new Set(Object.values(conversations).flatMap(conversation => conversation.messages.flatMap(message =>
    (message.attachments ?? []).flatMap(attachment => attachment.documentId ? [attachment.documentId] : []))));
}

/** Remove deleted conversation documents or abandoned prepared submissions. */
export async function removeChatDocuments(ids: Iterable<string>): Promise<void> {
  const keys = [...ids];
  if (!keys.length) return;
  const db = await openDocuments();
  await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(DOCUMENTS, 'readwrite');
    keys.forEach(id => transaction.objectStore(DOCUMENTS).delete(id));
    transaction.oncomplete = () => resolve();
    transaction.onerror = transaction.onabort = () => reject(new Error('Could not remove local conversation documents.'));
  });
}

/** Expire crash-abandoned documents after a day, preserving all live references. */
export async function pruneChatDocuments(referenced: Set<string>): Promise<void> {
  const db = await openDocuments();
  await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(DOCUMENTS, 'readwrite');
    const cursor = transaction.objectStore(DOCUMENTS).openCursor();
    cursor.onsuccess = () => {
      const row = cursor.result;
      if (!row) return;
      const value: unknown = row.value;
      if (!referenced.has(String(row.key)) && value && typeof value === 'object' && 'createdAt' in value
        && typeof value.createdAt === 'number' && value.createdAt < Date.now() - 86400000) row.delete();
      row.continue();
    };
    transaction.oncomplete = () => resolve();
    transaction.onerror = transaction.onabort = () => reject(new Error('Could not clean abandoned conversation documents.'));
  });
}
