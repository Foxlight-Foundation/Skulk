/** Exact provider-neutral continuation destination advertised by the unified API. */
export interface ContinuationTarget {
  target_node: string;
  callable_id: string;
  descriptor_revision: string;
  model: string;
}

/** Owner-approved public client identity; contains no signing or account secret. */
export interface ClientEnrollment {
  client: { public_key: string; caller_id: string; models: string[]; expires_at: number };
  target: ContinuationTarget;
}

/** Content-free provider commit, bound to one conversation and original attempt. */
export interface ContinuationReceipt {
  contract: 'chat.continuation@0.1.0';
  conversation_id: string;
  handle: string;
  history: { count: number; sha256: string };
  request_nonce: string;
  expires_at: number;
}

/** Explicit per-conversation opt-in; pending attempts never automatically replay. */
export interface ChatContinuation {
  enrollment: ClientEnrollment;
  target: ContinuationTarget;
  conversation_id: string;
  phase: 'ready' | 'pending' | 'uncertain';
  receipt?: ContinuationReceipt;
  request_nonce?: string;
}

const hex = (bytes: ArrayBuffer | Uint8Array) => Array.from(new Uint8Array(bytes instanceof Uint8Array ? bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) : bytes), byte => byte.toString(16).padStart(2, '0')).join('');
const encoder = new TextEncoder();
const databaseName = 'skulk-chat-client-v1';

async function identityDatabase(): Promise<IDBDatabase> {
  if (!globalThis.crypto?.subtle || !globalThis.indexedDB) throw new Error('Continuation requires Web Crypto and private browser storage on HTTPS or localhost.');
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(databaseName, 1);
    request.onupgradeneeded = () => { request.result.createObjectStore('identity'); };
    request.onerror = request.onblocked = () => reject(new Error('Browser client-key storage is unavailable.'));
    request.onsuccess = () => resolve(request.result);
  });
}

function keyPair(value: unknown): CryptoKeyPair | undefined {
  if (!value || typeof value !== 'object' || !('privateKey' in value) || !('publicKey' in value)) return;
  if (!(value.privateKey instanceof CryptoKey) || !(value.publicKey instanceof CryptoKey) || value.privateKey.extractable || value.privateKey.algorithm.name !== 'Ed25519' || value.publicKey.algorithm.name !== 'Ed25519') throw new Error('The saved browser identity is invalid.');
  return { privateKey: value.privateKey, publicKey: value.publicKey };
}

/** Read the origin-owned key, creating it only after an explicit client action. */
export async function browserContinuationIdentity(create = false): Promise<CryptoKeyPair | undefined> {
  const database = await identityDatabase();
  try {
    const existing = await new Promise<unknown>((resolve, reject) => {
      const request = database.transaction('identity').objectStore('identity').get('client');
      request.onsuccess = () => resolve(request.result as unknown);
      request.onerror = () => reject(new Error('Could not read this browser’s client key.'));
    });
    if (existing !== undefined) return keyPair(existing);
    if (!create) return;
    const candidate = await crypto.subtle.generateKey('Ed25519', false, ['sign', 'verify']);
    return await new Promise<CryptoKeyPair>((resolve, reject) => {
      // Serialize publication across tabs: a concurrent winner remains the
      // identity. No read sees a partially published or replaced private key.
      const transaction = database.transaction('identity', 'readwrite');
      const store = transaction.objectStore('identity');
      const request = store.get('client');
      let chosen = candidate;
      request.onsuccess = () => {
        try {
          const saved = keyPair(request.result as unknown);
          if (saved) chosen = saved;
          else store.add(candidate, 'client');
        } catch {
          transaction.abort();
        }
      };
      transaction.oncomplete = () => resolve(chosen);
      transaction.onerror = transaction.onabort = () => reject(new Error('Could not save this browser’s client key.'));
    });
  } finally { database.close(); }
}

/** Export only the public half of a browser key for independent owner approval. */
export async function browserContinuationPublicKey(create = false): Promise<string | undefined> {
  const keys = await browserContinuationIdentity(create);
  return keys ? hex(await crypto.subtle.exportKey('raw', keys.publicKey)) : undefined;
}

/** Cryptographically random conversation/attempt identity with no insecure fallback. */
export function continuationNonce(): string { return hex(crypto.getRandomValues(new Uint8Array(32))); }

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid public continuation data.');
  return value as Record<string, unknown>;
}

function fields(value: Record<string, unknown>, allowed: string[]): void {
  if (Object.keys(value).length !== allowed.length || Object.keys(value).some(key => !allowed.includes(key))) throw new Error('Invalid public continuation fields.');
}

/** Compare destination identity explicitly, including its opaque descriptor revision. */
export function sameContinuationTarget(left: ContinuationTarget, right: ContinuationTarget): boolean {
  return ['target_node', 'callable_id', 'descriptor_revision', 'model'].every(key => left[key as keyof ContinuationTarget] === right[key as keyof ContinuationTarget]);
}

/** Validate public enrollment against the browser key and currently advertised model. */
export function parseClientEnrollment(raw: string, target: ContinuationTarget, publicKey: string): ClientEnrollment {
  if (raw.length > 65536) throw new Error('Public enrollment is too large.');
  const value = object(JSON.parse(raw));
  fields(value, ['client', 'target']);
  const client = object(value.client), destination = object(value.target);
  fields(client, ['public_key', 'caller_id', 'models', 'expires_at']);
  fields(destination, ['target_node', 'callable_id', 'descriptor_revision', 'model']);
  if (client.public_key !== publicKey || typeof client.caller_id !== 'string' || !/^[a-f0-9]{32}$/.test(client.caller_id) || !Array.isArray(client.models) || client.models.length < 1 || client.models.length > 128 || client.models.some((model: unknown) => typeof model !== 'string' || !/^[\w.:@-]{1,128}$/u.test(model)) || new Set(client.models).size !== client.models.length || !client.models.includes(target.model) || typeof client.expires_at !== 'number' || !Number.isFinite(client.expires_at) || client.expires_at * 1000 <= Date.now()) throw new Error('Enrollment must approve this browser key and model with unexpired access.');
  if (destination.target_node !== target.target_node || destination.callable_id !== target.callable_id || destination.descriptor_revision !== target.descriptor_revision || typeof destination.model !== 'string') throw new Error('Enrollment belongs to a different continuation destination.');
  return { client: { public_key: publicKey, caller_id: client.caller_id, models: client.models as string[], expires_at: client.expires_at }, target: { ...target, model: destination.model } };
}

/** Canonical JSON shared with Python proofs and ordered public-history fingerprints. */
export function canonicalContinuationJSON(value: unknown, ascii = false): string {
  const sorted = (item: unknown): string => {
    if (Array.isArray(item)) return '[' + item.map(entry => sorted(entry ?? null)).join(',') + ']';
    if (item !== null && typeof item === 'object') return '{' + Object.entries(item).filter(([, entry]) => entry !== undefined).sort(([a], [b]) => {
      const left = Array.from(a, character => character.codePointAt(0)!), right = Array.from(b, character => character.codePointAt(0)!);
      for (let index = 0; index < Math.min(left.length, right.length); index++) if (left[index] !== right[index]) return left[index] - right[index];
      return left.length - right.length;
    }).map(([key, entry]) => JSON.stringify(key) + ':' + sorted(entry)).join(',') + '}';
    if (typeof item === 'number' && !Number.isFinite(item)) throw new Error('Nonfinite continuation number.');
    const encoded = JSON.stringify(item);
    if (encoded === undefined) throw new Error('Invalid continuation JSON.');
    return encoded;
  };
  const encoded = sorted(value);
  return ascii ? encoded.replace(/[\u007f-\uffff]/g, character => '\\u' + character.charCodeAt(0).toString(16).padStart(4, '0')) : encoded;
}

/** Match the exact ordered public prefix without storing provider continuation items. */
export async function publicHistoryFingerprint(messages: Record<string, unknown>[]): Promise<{ count: number; sha256: string }> {
  let digest = new Uint8Array(await crypto.subtle.digest('SHA-256', encoder.encode('skulk.transcript.v1\0')));
  for (const message of messages) {
    const raw = encoder.encode(canonicalContinuationJSON(message));
    const size = new Uint8Array(8);
    new DataView(size.buffer).setBigUint64(0, BigInt(raw.byteLength));
    const framed = new Uint8Array(32 + 8 + raw.byteLength);
    framed.set(digest); framed.set(size, 32); framed.set(raw, 40);
    digest = new Uint8Array(await crypto.subtle.digest('SHA-256', framed));
  }
  return { count: messages.length, sha256: hex(digest) };
}

/** Check a committed public prefix before any new inference attempt is submitted. */
export async function verifyContinuationHistory(state: ChatContinuation, messages: Record<string, unknown>[]): Promise<void> {
  if (!state.receipt) return;
  const prefix = await publicHistoryFingerprint(messages.slice(0, state.receipt.history.count));
  if (prefix.count !== state.receipt.history.count || prefix.sha256 !== state.receipt.history.sha256) throw new Error('The committed public history is missing or changed. Explicitly start a new context using visible history.');
}

async function signRequest(state: ChatContinuation, request: Record<string, unknown>): Promise<Record<string, unknown>> {
  const keys = await browserContinuationIdentity();
  if (!keys || hex(await crypto.subtle.exportKey('raw', keys.publicKey)) !== state.enrollment.client.public_key || state.enrollment.client.expires_at * 1000 <= Date.now() || !state.enrollment.client.models.includes(state.target.model)) throw new Error('This browser has no current client approval for the selected model.');
  const signed = { caller_id: state.enrollment.client.caller_id, request };
  const signature = hex(await crypto.subtle.sign('Ed25519', keys.privateKey, encoder.encode(canonicalContinuationJSON(signed, true))));
  return { ...signed, signature };
}

/** Sign the exact already-serialized request; never normalize or retry its body. */
export async function prepareContinuationRequest(state: ChatContinuation, target: ContinuationTarget, raw: string): Promise<{ header: string; pending: ChatContinuation }> {
  if (state.phase !== 'ready') throw new Error('The previous continuation attempt is uncertain. Recover its receipt or explicitly start a new context.');
  if (!sameContinuationTarget(state.target, target)) throw new Error('The continuation destination changed. Explicitly start a new context.');
  const bytes = encoder.encode(raw), nonce = continuationNonce();
  const proof = await signRequest(state, { contract: 'chat.continuation@0.1.0', mode: state.receipt ? 'continue' : 'start', conversation_id: state.conversation_id, handle: state.receipt?.handle ?? null, target, body: { size_bytes: bytes.byteLength, sha256: hex(await crypto.subtle.digest('SHA-256', bytes)), media_type: 'application/vnd.skulk.chat-request+json' }, nonce,
    // A nonintegral exact binary fraction keeps canonical Python float JSON
    // identical to JavaScript JSON even at an integer wall-clock second.
    expires_at: Math.floor(Date.now() / 1000) + 30.125 });
  const header = btoa(String.fromCharCode(...encoder.encode(JSON.stringify(proof)))).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '');
  if (header.length > 4096) throw new Error('Continuation proof exceeds the HTTP header bound.');
  return { header, pending: { ...state, phase: 'pending', request_nonce: nonce } };
}

/** Validate a receipt before binding it to this attempt; no content recovery is implied. */
export function readContinuationReceipt(value: unknown, state: ChatContinuation): ContinuationReceipt {
  const receipt = object(value), history = object(receipt.history);
  fields(receipt, ['contract', 'conversation_id', 'handle', 'history', 'request_nonce', 'expires_at']);
  fields(history, ['count', 'sha256']);
  if (receipt.contract !== 'chat.continuation@0.1.0' || receipt.conversation_id !== state.conversation_id || receipt.request_nonce !== state.request_nonce || typeof receipt.handle !== 'string' || !/^ct1_[a-f0-9]{64}$/.test(receipt.handle) || typeof receipt.expires_at !== 'number' || !Number.isFinite(receipt.expires_at) || receipt.expires_at * 1000 <= Date.now() || typeof history.count !== 'number' || !Number.isInteger(history.count) || history.count < 0 || history.count > 4096 || typeof history.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(history.sha256)) throw new Error('Invalid or mismatched continuation receipt.');
  return { contract: 'chat.continuation@0.1.0', conversation_id: state.conversation_id, request_nonce: state.request_nonce!, handle: receipt.handle, expires_at: receipt.expires_at, history: { count: history.count, sha256: history.sha256 } };
}

/** Sign a separately domain-bound receipt lookup without submitting inference. */
export async function prepareReceiptLookup(state: ChatContinuation): Promise<Record<string, unknown>> {
  if (!state.request_nonce) throw new Error('There is no pending attempt to recover.');
  return signRequest(state, { contract: 'chat.continuation.receipt@0.1.0', conversation_id: state.conversation_id, target: state.target, request_nonce: state.request_nonce, nonce: continuationNonce(), expires_at: Math.floor(Date.now() / 1000) + 30.125 });
}
