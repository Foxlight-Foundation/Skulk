import { chatStorageDocumentIds, openChatDatabase, putChatSnapshot } from './chatDatabase';
import { randomHex32 } from '../utils/randomIds';

/** Durable local intent kept separately from the canonical conversation. */
export interface RetainedChatEdit {
  id: string;
  conversationId: string;
  conversation: Record<string, unknown> | null;
  createdAt: number;
}

const maximumRetainedEdits = 64;
function object(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function edit(value: unknown): RetainedChatEdit {
  const record = object(value);
  if (typeof record.id !== 'string' || !/^[a-f0-9]{32}$/.test(record.id) || typeof record.conversationId !== 'string'
    || typeof record.createdAt !== 'number' || !Number.isFinite(record.createdAt)
    || (record.conversation !== null && (!record.conversation || typeof record.conversation !== 'object' || Array.isArray(record.conversation)))) throw new Error('Retained conversation storage is invalid.');
  return { id: record.id, conversationId: record.conversationId, createdAt: record.createdAt,
    conversation: record.conversation === null ? null : object(record.conversation) };
}

/** Preserve complete local conversation versions before releasing their failed writes. */
export async function retainChatEdits(intents: ReadonlyMap<string, Record<string, unknown> | undefined>): Promise<void> {
  const database = await openChatDatabase();
  try { await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(['recoveries', 'documents'], 'readwrite', { durability: 'strict' });
    const store = transaction.objectStore('recoveries'), count = store.count();
    let refusal: Error | undefined;
    count.onsuccess = () => {
      if (count.result + intents.size > maximumRetainedEdits) { refusal = new Error('Resolve retained conversation edits before saving more conflicts.'); transaction.abort(); return; }
      try {
        const add = () => {
          for (const [conversationId, conversation] of intents) {
            const record: RetainedChatEdit = { id: randomHex32(), conversationId, conversation: conversation ?? null, createdAt: Date.now() };
            store.add(record, record.id);
          }
        };
        const references = chatStorageDocumentIds({ conversations: Object.fromEntries([...intents].filter(([, value]) => value !== undefined)) });
        let remaining = references.size;
        if (!remaining) add();
        for (const id of references) {
          const request = transaction.objectStore('documents').get(id);
          request.onsuccess = () => {
            // If expiry cleanup won before quarantine, metadata alone cannot
            // claim custody. Keep the unsaved intent in its originating tab.
            if (!(object(request.result as unknown).blob instanceof Blob)) {
              refusal = new Error('A local conversation document is missing. Keep this tab open and reattach it before saving.');
              transaction.abort(); return;
            }
            try { if (--remaining === 0) add(); } catch { transaction.abort(); }
          };
        }
      } catch { transaction.abort(); }
    };
    transaction.oncomplete = () => resolve();
    transaction.onabort = transaction.onerror = () => reject(refusal ?? new Error('Could not retain local conversation edits. Keep this tab open and free browser storage.'));
  }); } finally { database.close(); }
}

/** Load bounded retained versions; a read never overwrites canonical history. */
export async function readRetainedChatEdits(): Promise<RetainedChatEdit[]> {
  const database = await openChatDatabase();
  try { return await new Promise((resolve, reject) => {
    const transaction = database.transaction('recoveries'), request = transaction.objectStore('recoveries').getAll(undefined, maximumRetainedEdits + 1);
    let records: RetainedChatEdit[] = [];
    let refusal: unknown;
    request.onsuccess = () => {
      try {
        const raw = request.result as unknown[];
        if (raw.length > maximumRetainedEdits) throw new Error('Retained conversation storage exceeds its supported capacity.');
        records = raw.map(edit).sort((a, b) => a.createdAt - b.createdAt || a.id.localeCompare(b.id));
      } catch (error) { refusal = error; transaction.abort(); }
    };
    transaction.oncomplete = () => resolve(records);
    transaction.onabort = transaction.onerror = () => reject(refusal ?? new Error('Could not read retained conversation edits.'));
  }); } finally { database.close(); }
}

function forkConversation(original: Record<string, unknown>, id: string): Record<string, unknown> {
  if (typeof original.name !== 'string' || typeof original.modelId !== 'string' || !Array.isArray(original.messages)) throw new Error('Retained conversation cannot be restored.');
  const messages = (original.messages as unknown[]).map(value => {
    const message = object(value);
    if (typeof message.id !== 'string' || !['user', 'assistant'].includes(String(message.role)) || typeof message.content !== 'string'
      || typeof message.timestamp !== 'number' || !Number.isFinite(message.timestamp)) throw new Error('Retained message cannot be restored.');
    // Copy visible public history only. In particular, provider tool turns,
    // continuation attempts, grants and future affinity fields cannot migrate.
    const following: Record<string, unknown> = { id: message.id, role: message.role, content: message.content, timestamp: message.timestamp };
    if (typeof message.thinkingContent === 'string') following.thinkingContent = message.thinkingContent;
    if (Array.isArray(message.attachments)) following.attachments = (message.attachments as unknown[]).map(value => {
      const attachment = object(value);
      if (typeof attachment.id !== 'string' || typeof attachment.name !== 'string' || typeof attachment.type !== 'string') throw new Error('Retained attachment cannot be restored.');
      const result: Record<string, unknown> = { id: attachment.id, name: attachment.name, type: attachment.type };
      for (const key of ['size', 'preview', 'textContent', 'documentId']) if (attachment[key] !== undefined) result[key] = attachment[key];
      return result;
    });
    return following;
  });
  return { id, name: original.name + ' (recovered)', modelId: original.modelId, messages, createdAt: Date.now(), updatedAt: Date.now() };
}

/** Atomically adopt saved history or fork local intent with a new, unbound identity. */
export async function resolveRetainedChatEdit(id: string, choice: 'fork' | 'adopt'): Promise<string | null> {
  const database = await openChatDatabase();
  try { return await new Promise((resolve, reject) => {
    const transaction = database.transaction(['snapshot', 'documents', 'recoveries'], 'readwrite', { durability: 'strict' });
    const recoveries = transaction.objectStore('recoveries'), snapshot = transaction.objectStore('snapshot');
    const retained = recoveries.get(id);
    let conversationId: string | null = null;
    let refusal: unknown;
    let failed = false;
    const refuse = (error: unknown) => { if (failed) return; failed = true; refusal = error; transaction.abort(); };
    retained.onsuccess = () => {
      try {
        if (retained.result === undefined) throw new Error('These retained edits were already resolved in another tab.');
        const record = edit(retained.result as unknown);
        const current = snapshot.get('current');
        current.onsuccess = () => {
          try {
            const saved = object(current.result as unknown), state = object(saved.state);
            if (!saved.state || typeof saved.revision !== 'number' || !Number.isSafeInteger(saved.revision) || saved.revision >= Number.MAX_SAFE_INTEGER) throw new Error('Saved conversation storage is invalid.');
            if (choice === 'fork') {
              if (!record.conversation) throw new Error('A retained deletion has no local history to fork.');
              conversationId = randomHex32();
              const fork = forkConversation(record.conversation, conversationId);
              putChatSnapshot(transaction, state, { ...state,
                conversations: { ...object(state.conversations), [conversationId]: fork },
                modelToConversationId: { ...object(state.modelToConversationId), [String(fork.modelId)]: conversationId },
              }, saved.revision + 1, refuse);
            }
            if (!failed) recoveries.delete(id);
          } catch (error) { refuse(error); }
        };
      } catch (error) { refuse(error); }
    };
    transaction.oncomplete = () => { announceRetainedEdits(); resolve(conversationId); };
    transaction.onabort = transaction.onerror = () => reject(refusal ?? new Error('Could not resolve retained edits. Both versions were preserved.'));
  }); } finally { database.close(); }
}

/** Notify all renderers without publishing conversation contents. */
export function announceRetainedEdits(): void {
  if (typeof BroadcastChannel === 'undefined') return;
  const channel = new BroadcastChannel('skulk-chat-writers-v2');
  channel.postMessage('committed');
  channel.close();
}
