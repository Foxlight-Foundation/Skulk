import type { ChatMessage, ChatUploadedFile, Conversation } from '../types/chat';
import { randomHex32 } from '../utils/randomIds';
import { CHAT_DOCUMENT_STORE, chatStorageDocumentIds, openChatDatabase } from './chatDatabase';

/** Transport limits shared by the versioned document chat contract. */
export const MAX_CHAT_DOCUMENT_BYTES = 8 * 1024 * 1024;
export const MAX_CHAT_DOCUMENTS = 8;
export const MAX_CHAT_REQUEST_BYTES = 16 * 1024 * 1024;
const DOCUMENTS = CHAT_DOCUMENT_STORE;
const stagingOwner = randomHex32();
const stagingLifetime = 86400000;

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
  const db = await openChatDatabase();
  try { await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(DOCUMENTS, 'readwrite');
    transaction.oncomplete = () => resolve();
    transaction.onabort = transaction.onerror = () => reject(new Error('Could not save documents in this browser. Free browser storage and try again.'));
    try {
      for (const { id, blob } of prepared) transaction.objectStore(DOCUMENTS).add({ blob, createdAt: Date.now(), staged: { owner: stagingOwner, expiresAt: Date.now() + stagingLifetime } }, id);
    } catch {
      transaction.abort();
      reject(new Error('Could not save documents in this browser. Free browser storage and try again.'));
    }
  }); } finally { db.close(); }
  return files.map(file => {
    const document = prepared.find(item => item.file === file);
    return document ? { ...file, documentId: document.id, type: document.blob.type } : file;
  });
}

async function documentData(id: string): Promise<string> {
  const db = await openChatDatabase();
  let blob: Blob;
  try { blob = await new Promise<Blob>((resolve, reject) => {
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
  }); } finally { db.close(); }
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
  await collectDocuments(new Set(keys));
}

/** Expire abandoned staging leases while atomically preserving canonical references. */
export async function pruneChatDocuments(): Promise<void> {
  await collectDocuments();
}

async function collectDocuments(explicit?: Set<string>): Promise<void> {
  const db = await openChatDatabase();
  try { await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(['snapshot', DOCUMENTS, 'recoveries'], 'readwrite', { durability: 'strict' });
    const current = transaction.objectStore('snapshot').get('current');
    current.onsuccess = () => {
      const saved: unknown = current.result;
      if (!saved || typeof saved !== 'object' || !('state' in saved) || !saved.state || typeof saved.state !== 'object') { transaction.abort(); return; }
      let referenced: Set<string>;
      try { referenced = chatStorageDocumentIds(saved.state, true); }
      catch { transaction.abort(); return; }
      const recoveries = transaction.objectStore('recoveries').getAll(undefined, 65);
      recoveries.onsuccess = () => {
        const retained = recoveries.result as unknown[];
        if (retained.length > 64) { transaction.abort(); return; }
        for (const value of retained) {
          if (!value || typeof value !== 'object' || !('conversation' in value)) { transaction.abort(); return; }
          try {
            if (value.conversation !== null) for (const id of chatStorageDocumentIds({ conversations: { retained: value.conversation } }, true)) referenced.add(id);
          } catch { transaction.abort(); return; }
        }
      const cursor = transaction.objectStore(DOCUMENTS).openCursor();
      cursor.onsuccess = () => {
        const row = cursor.result;
        if (!row) return;
        const value: unknown = row.value;
        const staged = value && typeof value === 'object' && 'staged' in value ? value.staged : undefined;
        const expired = staged && typeof staged === 'object' && 'expiresAt' in staged && typeof staged.expiresAt === 'number' && Number.isFinite(staged.expiresAt) && staged.expiresAt <= Date.now();
        const removable = expired || (staged && typeof staged === 'object'
          && (('owner' in staged && (staged.owner === stagingOwner || staged.owner === 'legacy')) || ('claimed' in staged && staged.claimed === true)));
        if (!referenced.has(String(row.key)) && (explicit ? explicit.has(String(row.key)) && removable : expired)) row.delete();
        row.continue();
      };
      };
    };
    transaction.oncomplete = () => resolve();
    transaction.onerror = transaction.onabort = () => reject(new Error('Could not clean local conversation documents. Existing references were preserved.'));
  }); } finally { db.close(); }
}
