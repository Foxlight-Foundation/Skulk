/** One transactional boundary for canonical history, staged bytes and recovery. */
export const CHAT_DATABASE = 'skulk-chat-history-v2';
/** Blob records participate in the same transaction as their history references. */
export const CHAT_DOCUMENT_STORE = 'documents';
const LEGACY_DOCUMENT_DATABASE = 'skulk-chat-documents-v1';
const MIGRATION_KEY = 'document-migration-v1';

interface MigrationProgress { lastKey: string | null; done: boolean }
interface LegacyDocument { key: string; value: unknown }

function object(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function progress(value: unknown): MigrationProgress {
  const record = object(value);
  return { lastKey: typeof record.lastKey === 'string' ? record.lastKey : null, done: record.done === true };
}

function openDatabase(name: string, version: number, stores: string[]): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    let failed = false;
    const request = indexedDB.open(name, version);
    request.onupgradeneeded = () => {
      for (const store of stores) if (!request.result.objectStoreNames.contains(store)) request.result.createObjectStore(store);
    };
    request.onerror = request.onblocked = () => { failed = true; reject(new Error('Conversation database is unavailable. Your draft was retained.')); };
    request.onsuccess = () => {
      if (failed) { request.result.close(); return; }
      request.result.onversionchange = () => request.result.close();
      resolve(request.result);
    };
  });
}

async function readProgress(database: IDBDatabase): Promise<MigrationProgress> {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction('snapshot');
    const request = transaction.objectStore('snapshot').get(MIGRATION_KEY);
    let result: MigrationProgress;
    request.onsuccess = () => { result = progress(request.result as unknown); };
    transaction.oncomplete = () => resolve(result);
    transaction.onabort = transaction.onerror = () => reject(new Error('Could not read document migration progress.'));
  });
}

async function nextLegacyDocument(database: IDBDatabase, lastKey: string | null): Promise<LegacyDocument | undefined> {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(CHAT_DOCUMENT_STORE);
    const request = transaction.objectStore(CHAT_DOCUMENT_STORE).openCursor(lastKey === null ? undefined : IDBKeyRange.lowerBound(lastKey, true));
    let result: LegacyDocument | undefined;
    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor) return;
      if (typeof cursor.key !== 'string') { transaction.abort(); return; }
      result = { key: cursor.key, value: cursor.value as unknown };
    };
    transaction.oncomplete = () => resolve(result);
    transaction.onabort = transaction.onerror = () => reject(new Error('Could not read existing conversation documents.'));
  });
}

async function advanceMigration(database: IDBDatabase, expected: MigrationProgress, document?: LegacyDocument): Promise<MigrationProgress> {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(['snapshot', CHAT_DOCUMENT_STORE], 'readwrite', { durability: 'strict' });
    const snapshot = transaction.objectStore('snapshot'), documents = transaction.objectStore(CHAT_DOCUMENT_STORE);
    const request = snapshot.get(MIGRATION_KEY);
    let result: MigrationProgress;
    request.onsuccess = () => {
      const actual = progress(request.result as unknown);
      result = actual;
      // Another renderer may have copied this page. Never move its cursor back
      // or resurrect a record after migration has completed and GC has run.
      if (actual.done || actual.lastKey !== expected.lastKey) return;
      result = { lastKey: document?.key ?? actual.lastKey, done: document === undefined };
      if (document) {
        const existing = documents.get(document.key);
        existing.onsuccess = () => {
          try { if (existing.result === undefined) {
            const old = object(document.value);
            const createdAt = typeof old.createdAt === 'number' ? old.createdAt : Date.now();
            documents.add({ ...old, createdAt, staged: { owner: 'legacy', expiresAt: createdAt + 86400000 } }, document.key);
          } } catch { transaction.abort(); }
        };
      } else {
        const saved = snapshot.get('current');
        saved.onsuccess = () => {
          if (saved.result !== undefined) return;
          try {
            const raw = localStorage.getItem('skulk-chat-v2') ?? localStorage.getItem('skulk-chat');
            const state = raw ? object(JSON.parse(raw) as unknown).state : { conversations: {}, modelToConversationId: {} };
            if (!state || typeof state !== 'object' || Array.isArray(state)) throw new Error('Invalid saved history');
            snapshot.put({ state, version: 2, revision: 0 }, 'current');
          } catch { transaction.abort(); }
        };
      }
      snapshot.put(result, MIGRATION_KEY);
    };
    transaction.oncomplete = () => resolve(result);
    transaction.onabort = transaction.onerror = () => reject(new Error('Could not migrate conversation documents. Existing storage was preserved.'));
  });
}

async function migrateDocuments(database: IDBDatabase): Promise<void> {
  let current = await readProgress(database);
  if (!current.done) {
    const legacy = await openDatabase(LEGACY_DOCUMENT_DATABASE, 1, [CHAT_DOCUMENT_STORE]);
    try {
      // One blob per page bounds migration memory. The durable cursor permits
      // crash recovery and competing renderers without a cross-database lease.
      while (!current.done) current = await advanceMigration(database, current, await nextLegacyDocument(legacy, current.lastKey));
    } finally { legacy.close(); }
  }
  // Version-two history admission has closed old version-one writers. Only
  // after every legacy blob is durably copied may its old database be retired.
  // An old tab can delay deletion, but cannot delay new history admission.
  const removal = indexedDB.deleteDatabase(LEGACY_DOCUMENT_DATABASE);
  removal.onerror = removal.onblocked = () => { /* A later open retries retirement. */ };
}

/** Open upgraded storage after resumable legacy migration; caller closes the handle. */
export async function openChatDatabase(): Promise<IDBDatabase> {
  const database = await openDatabase(CHAT_DATABASE, 2, ['snapshot', CHAT_DOCUMENT_STORE, 'recoveries']);
  try { await migrateDocuments(database); return database; }
  catch (error) { database.close(); throw error; }
}

/** Read only document handles from a snapshot, tolerating unrelated legacy fields. */
export function chatStorageDocumentIds(state: unknown, strict = false): Set<string> {
  const ids = new Set<string>();
  const conversations = object(state).conversations;
  if (strict && (!conversations || typeof conversations !== 'object' || Array.isArray(conversations))) throw new Error('Saved conversation references are invalid.');
  for (const conversation of Object.values(object(conversations))) {
    const messages = object(conversation).messages;
    if (!Array.isArray(messages)) { if (strict) throw new Error('Saved conversation references are invalid.'); continue; }
    for (const message of messages as unknown[]) {
      if (strict && (!message || typeof message !== 'object' || Array.isArray(message))) throw new Error('Saved message references are invalid.');
      const attachments = object(message).attachments;
      if (!Array.isArray(attachments)) { if (strict && attachments !== undefined) throw new Error('Saved attachment references are invalid.'); continue; }
      for (const attachment of attachments as unknown[]) {
        const id = object(attachment).documentId;
        if (strict && (!attachment || typeof attachment !== 'object' || Array.isArray(attachment) || (id !== undefined && typeof id !== 'string'))) throw new Error('Saved attachment references are invalid.');
        if (typeof id === 'string') ids.add(id);
      }
    }
  }
  return ids;
}

/** Validate newly introduced references and publish history in the same transaction. */
export function putChatSnapshot(transaction: IDBTransaction, before: unknown, after: unknown, revision: number, refuse: (error: Error) => void): void {
  const previous = chatStorageDocumentIds(before);
  const added = [...chatStorageDocumentIds(after)].filter(id => !previous.has(id));
  const put = () => {
    try { transaction.objectStore('snapshot').put({ state: after, version: 2, revision }, 'current'); }
    catch { refuse(new Error('Could not save conversation history. Your draft was retained.')); }
  };
  if (!added.length) { put(); return; }
  let remaining = added.length;
  for (const id of added) {
    const request = transaction.objectStore(CHAT_DOCUMENT_STORE).get(id);
    request.onsuccess = () => {
      const record = object(request.result as unknown);
      if (!(record.blob instanceof Blob)) {
        refuse(new Error('A conversation document is missing from this browser. Reattach it before sending.'));
        return;
      }
      try {
        // Claim staging bytes in the same transaction as the first canonical
        // reference, so another tab can later clean an explicitly deleted chat.
        transaction.objectStore(CHAT_DOCUMENT_STORE).put({ ...record, staged: { ...object(record.staged), claimed: true } }, id);
        if (--remaining === 0) put();
      } catch { refuse(new Error('Could not save conversation documents. Your draft was retained.')); }
    };
  }
}
