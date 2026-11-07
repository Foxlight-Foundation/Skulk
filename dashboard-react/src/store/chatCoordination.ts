import { canonicalContinuationJSON } from './chatContinuation';
import { CHAT_DOCUMENT_STORE, openChatDatabase, putChatSnapshot } from './chatDatabase';
import { retainChatEdits } from './chatRetainedEdits';

/** Durable JSON fields; browser-only drafts and signing keys never enter this store. */
export type ChatStorageObject = { [key: string]: unknown };
/** One compare-and-swap change, applied with all other changes or refused entirely. */
export interface ChatStorageChange { path: string[]; before: unknown; after: unknown }
/** Isolate current writes from older dashboard tabs that still replace legacy snapshots. */
export const CHAT_STORAGE_KEY = 'skulk-chat-v2';
const storageLock = 'skulk-chat-storage-v2';
const conversationPrefix = 'skulk-chat-conversation-v2:';
const owned = new Set<string>();
let queue: Promise<unknown> = Promise.resolve();
let unsaved: { before: ChatStorageObject; after: ChatStorageObject } | undefined;
let unopenedChanges: ChatStorageChange[] = [];
const pendingIntents = new Map<string, ChatStorageObject | undefined>();

/** A stale conversation must be quarantined before independent work continues. */
export class ChatStorageConflict extends Error {
  readonly conversationId: string;
  constructor(message: string, conversationId: string) { super(message); this.conversationId = conversationId; }
}

function changeConversation(change: ChatStorageChange): string | undefined {
  if (change.path[0] === 'conversations') return change.path[1];
  if (change.path[0] === 'modelToConversationId') return typeof change.after === 'string' ? change.after : typeof change.before === 'string' ? change.before : undefined;
}

function retainUnopenedChanges(changes: ChatStorageChange[]): void {
  for (const change of changes) {
    const index = unopenedChanges.findLastIndex(previous => equal(previous.path, change.path));
    const overlaps = (left: string[], right: string[]) => left.every((part, i) => right[i] === part) || right.every((part, i) => left[i] === part);
    if (index >= 0 && !unopenedChanges.slice(index + 1).some(previous => overlaps(previous.path, change.path))) {
      unopenedChanges[index] = { ...change, before: unopenedChanges[index].before };
    } else unopenedChanges.push(change);
  }
}
/** Ordinary save failure retains local intent without claiming durable custody. */
export class ChatStorageUnavailable extends Error {}
/** Whether local edits still require a successful durable write. */
export function hasUnsavedChatChanges(): boolean { return unsaved !== undefined || unopenedChanges.length > 0; }

function object(value: unknown): ChatStorageObject {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as ChatStorageObject : {};
}
function equal(left: unknown, right: unknown): boolean {
  if (left === undefined || right === undefined) return left === right;
  return canonicalContinuationJSON(left) === canonicalContinuationJSON(right);
}

const defaults = { conversations: {}, modelToConversationId: {}, selectedVoice: null, autoSpeakAssistant: false, narrateCodeBlocks: true, realtimeVoiceEnabled: true, autoSubmitVoice: false };
function storedState(value: unknown): ChatStorageObject {
  const state = object(value).state;
  if (!state || typeof state !== 'object' || Array.isArray(state)) throw new Error('Saved conversation storage is invalid.');
  return { ...defaults, ...object(state) };
}

/** Read a startup cache; inference admission always refreshes authoritative IndexedDB state. */
export function readChatStorage(): ChatStorageObject {
  const raw = localStorage.getItem(CHAT_STORAGE_KEY) ?? localStorage.getItem('skulk-chat');
  return raw ? storedState(JSON.parse(raw) as unknown) : { ...defaults };
}

async function historyDatabase(): Promise<IDBDatabase> {
  return openChatDatabase();
}

function storedRevision(value: unknown): number {
  const revision = object(value).revision ?? 0;
  if (typeof revision !== 'number' || !Number.isSafeInteger(revision) || revision < 0 || revision >= Number.MAX_SAFE_INTEGER) throw new Error('Saved conversation revision is invalid.');
  return revision;
}

/** One committed browser snapshot and its monotonically increasing storage revision. */
export interface ChatStorageSnapshot { state: ChatStorageObject; revision: number }

/** Read committed data independently of another renderer's storage cache. */
export async function readCommittedChatSnapshot(): Promise<ChatStorageSnapshot> {
  const database = await historyDatabase();
  try {
    return await new Promise((resolve, reject) => {
      const transaction = database.transaction('snapshot');
      const request = transaction.objectStore('snapshot').get('current');
      let snapshot: ChatStorageSnapshot;
      request.onsuccess = () => {
        try {
          const encoded = request.result as unknown;
          snapshot = { state: encoded === undefined ? readChatStorage() : storedState(encoded), revision: storedRevision(encoded) };
        }
        catch (error) { reject(error); }
      };
      transaction.oncomplete = () => resolve(snapshot);
      transaction.onerror = transaction.onabort = () => reject(new ChatStorageUnavailable('Could not read saved conversation history.'));
    });
  } finally { database.close(); }
}

/** Read committed fields for callers that do not need an adoption revision fence. */
export async function readCommittedChatStorage(): Promise<ChatStorageObject> {
  return (await readCommittedChatSnapshot()).state;
}

async function commitChatStorage(changes: ChatStorageChange[], pendingChanges: ChatStorageChange[], retainOrdinaryEdits: boolean): Promise<ChatStorageObject> {
  let database: IDBDatabase;
  try { database = await historyDatabase(); }
  catch (error) {
    // No transaction exists yet, but these ordinary edits are still local
    // intent. Keep their CAS preconditions so a later read cannot discard them.
    if (retainOrdinaryEdits) retainUnopenedChanges(changes);
    throw error;
  }
  let transactionStarted = false;
  try {
    return await new Promise((resolve, reject) => {
      const transaction = database.transaction(['snapshot', CHAT_DOCUMENT_STORE], 'readwrite', { durability: 'strict' });
      transactionStarted = true;
      const store = transaction.objectStore('snapshot'), request = store.get('current');
      let current: ChatStorageObject | undefined, next: ChatStorageObject | undefined;
      let refusal: unknown;
      let revision = 0;
      request.onsuccess = () => {
        try {
          const encoded = request.result as unknown;
          current = encoded === undefined ? readChatStorage() : storedState(encoded);
          // Failed local actions may depend on an earlier unsaved creation.
          // Validate their ordered edits in memory; the database commits the
          // entire result only after every precondition has passed.
          const retained = pendingChanges.reduce((state, change) => applyChatStorageChanges(state, [change]), current);
          next = applyChatStorageChanges(retained, changes);
          revision = storedRevision(encoded) + 1;
        } catch (error) { refusal = error; transaction.abort(); return; }
        try { putChatSnapshot(transaction, current, next, revision, error => { refusal = error; transaction.abort(); }); }
        catch { transaction.abort(); }
      };
      transaction.onabort = () => {
        if (refusal !== undefined) { if (retainOrdinaryEdits) retainUnopenedChanges(changes); reject(refusal); return; }
        if (retainOrdinaryEdits && current && next) { unsaved = { before: current, after: next }; unopenedChanges = []; }
        reject(new ChatStorageUnavailable('Could not save conversation history in this browser. Free browser storage and try again.'));
      };
      transaction.oncomplete = () => {
        unsaved = undefined;
        unopenedChanges = [];
        pendingIntents.clear();
        // This mirror is optional. A quota failure cannot undo or misreport an
        // authoritative transaction, and legacy writers cannot replace it.
        try { localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify({ state: next, version: 2, revision })); } catch { /* Hydrate from IndexedDB on next load. */ }
        announceWriter('committed');
        resolve(next!);
      };
    });
  } catch (error) {
    if (retainOrdinaryEdits && !transactionStarted) retainUnopenedChanges(changes);
    throw error;
  } finally { database.close(); }
}

/** Describe only local changes, so unrelated conversations in stale tabs stay untouched. */
export function chatStorageChanges(before: ChatStorageObject, after: ChatStorageObject): ChatStorageChange[] {
  const changes: ChatStorageChange[] = [];
  const difference = (path: string[], left: unknown, right: unknown) => {
    if (!equal(left, right)) changes.push({ path, before: left, after: right });
  };
  for (const key of new Set([...Object.keys(before), ...Object.keys(after)])) {
    if (key === 'conversations' || key === 'modelToConversationId') {
      const left = object(before[key]), right = object(after[key]);
      for (const id of new Set([...Object.keys(left), ...Object.keys(right)])) {
        // A destination change must compare the whole context, including a
        // concurrent enrollment that did not change modelId itself.
        if (key === 'conversations' && left[id] !== undefined && right[id] !== undefined && equal(object(left[id]).modelId, object(right[id]).modelId)) {
          const previous = object(left[id]), next = object(right[id]);
          const fields = [...new Set([...Object.keys(previous), ...Object.keys(next)])].filter(field => !equal(previous[field], next[field]));
          // Field-level CAS must not graft an old model's public history or
          // authority onto a context reassigned by an older renderer.
          if (fields.some(field => field !== 'updatedAt')) changes.push({ path: [key, id, 'modelId'], before: previous.modelId, after: previous.modelId });
          for (const field of fields) difference([key, id, field], previous[field], next[field]);
        } else difference([key, id], left[id], right[id]);
      }
    } else difference([key], before[key], after[key]);
  }
  return changes;
}

function getPath(state: ChatStorageObject, path: string[]): unknown {
  return path.reduce<unknown>((value, key) => Object.hasOwn(object(value), key) ? object(value)[key] : undefined, state);
}

/** Refuse stale fields before changing anything; already-applied acknowledgments are idempotent. */
export function applyChatStorageChanges(current: ChatStorageObject, changes: ChatStorageChange[]): ChatStorageObject {
  for (const change of changes) {
    if (change.path[0] === 'conversations' && change.path.length > 2 && getPath(current, change.path.slice(0, 2)) === undefined) throw new ChatStorageConflict('The conversation was deleted in another tab. Its history was not recreated.', change.path[1]);
    const actual = getPath(current, change.path);
    if (!equal(actual, change.before) && !equal(actual, change.after) && change.path.at(-1) !== 'updatedAt') {
      const message = 'This conversation changed in another tab. Its saved history was preserved; review the refreshed conversation before sending.';
      const id = changeConversation(change);
      if (id !== undefined) throw new ChatStorageConflict(message, id);
      throw new Error(message);
    }
  }
  const next = structuredClone(current);
  for (const { path, after } of changes) {
    let target = next;
    for (const key of path.slice(0, -1)) {
      if (!Object.hasOwn(target, key)) Object.defineProperty(target, key, { value: {}, enumerable: true, configurable: true, writable: true });
      target = object(target[key]);
    }
    const key = path.at(-1)!;
    if (after === undefined) delete target[key];
    else Object.defineProperty(target, key, { value: key === 'updatedAt' ? Math.max(Number(target[key]) || 0, Number(after) || 0) : structuredClone(after), enumerable: true, configurable: true, writable: true });
  }
  return next;
}

function locks(): LockManager {
  if (!navigator.locks) throw new Error('Coordinated chat requires browser locking on HTTPS or localhost.');
  return navigator.locks;
}

async function withConversationLocks<T>(ids: string[], work: () => Promise<T>): Promise<T> {
  const [id, ...rest] = ids;
  if (id === undefined || !navigator.locks) return work();
  if (owned.has(id)) return withConversationLocks(rest, work);
  return locks().request(conversationPrefix + id, { ifAvailable: true }, lock => {
    if (!lock) throw new Error('This conversation is active in another tab. Its history was preserved.');
    return withConversationLocks(rest, work);
  });
}

/** Serialize local writes and atomically merge fields, using shared locks where available. */
export async function writeChatStorage(changes: ChatStorageChange[], retainOrdinaryEdits = false, localIntent?: ChatStorageObject): Promise<ChatStorageObject> {
  const pending = queue.catch(() => {}).then(async () => {
    const commit = async () => {
      const intent = new Map(pendingIntents);
      if (retainOrdinaryEdits && localIntent) {
        for (const change of changes) {
          const id = changeConversation(change);
          if (id !== undefined) {
            const conversation = object(localIntent.conversations)[id];
            intent.set(id, conversation === undefined ? undefined : object(conversation));
          }
        }
      }
      let remaining = changes;
      for (;;) {
        const pendingChanges = [...(unsaved ? chatStorageChanges(unsaved.before, unsaved.after) : []), ...unopenedChanges];
        const ids = [...new Set([...pendingChanges, ...remaining].filter(change => change.path[0] === 'conversations' && !(change.path.length === 3 && change.path[2] === 'updatedAt')).map(change => change.path[1]))].sort();
        let started = false;
        try {
          const result = await withConversationLocks(ids, () => { started = true; return commitChatStorage(remaining, pendingChanges, retainOrdinaryEdits); });
          return result;
        } catch (error) {
          if (error instanceof ChatStorageConflict && intent.has(error.conversationId)) {
            // Publish custody before dropping any failed CAS. If this write
            // cannot commit, the pending intent stays in this tab unchanged.
            try { await retainChatEdits(new Map([[error.conversationId, intent.get(error.conversationId)]])); }
            catch (storageError) {
              if (retainOrdinaryEdits) {
                for (const [id, conversation] of intent) pendingIntents.set(id, conversation);
                retainUnopenedChanges(remaining);
              }
              throw storageError;
            }
            const independent = (change: ChatStorageChange) => changeConversation(change) !== error.conversationId;
            unopenedChanges = pendingChanges.filter(independent);
            unsaved = undefined;
            pendingIntents.delete(error.conversationId);
            intent.delete(error.conversationId);
            remaining = remaining.filter(independent);
            continue;
          }
          if (retainOrdinaryEdits) {
            for (const [id, conversation] of intent) pendingIntents.set(id, conversation);
            if (!started) retainUnopenedChanges(remaining);
          }
          throw error;
        }
      }
    };
    // IndexedDB transactions still serialize ordinary HTTP clients. Shared
    // long-lived writer ownership is available only in secure browser contexts.
    return await (navigator.locks ? locks().request(storageLock, commit) : commit());
  });
  queue = pending;
  return pending;
}

/** Wait until every write already queued by this tab has settled, including failures. */
export async function flushChatStorage(): Promise<void> { await queue; }

/** Hold a conversation across preparation, generation, client tools and terminal persistence. */
export async function acquireChatConversation(id: string): Promise<() => Promise<void>> {
  // The tab's own ordinary saves may briefly hold this same lock. Drain those
  // before attempting ownership, rather than mistaking them for another tab.
  await flushChatStorage().catch(() => {});
  if (owned.has(id)) throw new Error('This conversation is already active in this tab.');
  if (!navigator.locks) {
    owned.add(id);
    return async () => { owned.delete(id); };
  }
  let finishRelease!: () => void;
  const released = new Promise<void>(resolve => { finishRelease = resolve; });
  return new Promise((resolve, reject) => {
    void locks().request(conversationPrefix + id, { ifAvailable: true }, async lock => {
      if (!lock) { reject(new Error('This conversation is active in another tab. Your draft was retained.')); return; }
      // The separate presence lock distinguishes a long turn from a brief
      // transactional save; a sampled short write must not leave the UI busy.
      await locks().request(conversationPrefix + id + ':writer', async () => {
        owned.add(id);
        announceWriter();
        try { await new Promise<void>(release => resolve(async () => { release(); await released; })); }
        finally { owned.delete(id); }
      });
    }).then(() => { announceWriter(); finishRelease(); }, error => { finishRelease(); reject(error); });
  });
}

/** Observe actual browser lock ownership; notifications carry no chat contents. */
export function observeChatConversation(id: string, listener: (busy: boolean) => void): () => void {
  let active = true;
  let queryRevision = 0;
  let poll: number | undefined;
  const update = () => {
    const revision = ++queryRevision;
    if (!navigator.locks) return;
    void navigator.locks.query().then(snapshot => {
      if (!active || revision !== queryRevision) return;
      const busy = !owned.has(id) && snapshot.held?.some(lock => lock.name === conversationPrefix + id + ':writer') === true;
      listener(busy);
      // A closed renderer cannot broadcast its release. Poll only while a
      // foreign writer exists so a surviving tab can discover owner death.
      if (busy && poll === undefined) poll = window.setInterval(update, 1000);
      if (!busy && poll !== undefined) { window.clearInterval(poll); poll = undefined; }
    }).catch(() => { /* Ownership admission still checks the actual lock. */ });
  };
  const channel = typeof BroadcastChannel === 'undefined' ? undefined : new BroadcastChannel('skulk-chat-writers-v2');
  if (channel) channel.onmessage = update;
  window.addEventListener('focus', update);
  window.addEventListener('storage', update);
  update();
  return () => { active = false; if (poll !== undefined) window.clearInterval(poll); channel?.close(); window.removeEventListener('focus', update); window.removeEventListener('storage', update); };
}

function announceWriter(message = 'changed'): void {
  if (typeof BroadcastChannel === 'undefined') return;
  const channel = new BroadcastChannel('skulk-chat-writers-v2');
  channel.postMessage(message);
  channel.close();
}
