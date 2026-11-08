import { describe, expect, it } from 'vitest';
import { browserContinuationIdentity, browserContinuationPublicKey, canonicalContinuationJSON, parseClientEnrollment, prepareContinuationRequest, prepareReceiptLookup, publicHistoryFingerprint, readContinuationReceipt, verifyContinuationHistory, type ChatContinuation } from './chatContinuation';

const target = { target_node: 'test.owner', callable_id: 'independent.continuation', descriptor_revision: 'a'.repeat(16), model: 'code' };

describe('client-owned continuation', () => {
  it('matches Python canonical ordering, Unicode escaping and history framing', async () => {
    expect(canonicalContinuationJSON({ '2': 2, '10': 10, '🦊': '\u007f hi 🦊' }, true)).toBe('{"10":10,"2":2,"\\ud83e\\udd8a":"\\u007f hi \\ud83e\\udd8a"}');
    // Generated independently by the provider-neutral Python append_message.
    expect(await publicHistoryFingerprint([{ role: 'user', content: ' hi 🦊' }, { role: 'assistant', content: 'hello\n', reasoning_content: 'short' }])).toEqual({ count: 2, sha256: '421ef8426f71a1940ee4253a040ca381b55417a7d49386bdaf21e70a41521a34' });
  });

  it('keeps one nonextractable browser key across reloads and concurrent creators', async () => {
    const keys = await Promise.all([browserContinuationIdentity(true), browserContinuationIdentity(true)]);
    expect(keys[0]?.privateKey.extractable).toBe(false);
    expect(await crypto.subtle.exportKey('raw', keys[0]!.publicKey)).toEqual(await crypto.subtle.exportKey('raw', keys[1]!.publicKey));
    await expect(crypto.subtle.exportKey('pkcs8', keys[0]!.privateKey)).rejects.toThrow();
    expect(await browserContinuationPublicKey()).toMatch(/^[a-f0-9]{64}$/);
  });

  it('binds approval, exact body, target, separate recovery and a one-use pending phase', async () => {
    const publicKey = await browserContinuationPublicKey(true);
    const enrollment = { client: { public_key: publicKey!, caller_id: 'b'.repeat(32), models: ['code'], expires_at: Date.now() / 1000 + 3600 }, target: { ...target, model: 'unselected' } };
    const state: ChatContinuation = { enrollment: parseClientEnrollment(JSON.stringify(enrollment), target, publicKey!), target, conversation_id: 'c'.repeat(64), phase: 'ready' };
    const raw = '{\n "model":"capability/independent.chat/code", "messages":[{"role":"user","content":" hi 🦊"}]}';
    const prepared = await prepareContinuationRequest(state, target, raw);
    const proof = JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(prepared.header.replaceAll('-', '+').replaceAll('_', '/')), character => character.charCodeAt(0)))) as { caller_id: string; request: { body: { size_bytes: number; sha256: string }; expires_at: number }; signature: string };
    expect(proof.request.body.size_bytes).toBe(new TextEncoder().encode(raw).length);
    expect(proof.request.expires_at % 1).toBe(0.125);
    const signature = Uint8Array.from(proof.signature.match(/../g)!, pair => parseInt(pair, 16));
    const keys = await browserContinuationIdentity();
    expect(await crypto.subtle.verify('Ed25519', keys!.publicKey, signature, new TextEncoder().encode(canonicalContinuationJSON({ caller_id: proof.caller_id, request: proof.request }, true)))).toBe(true);
    await expect(prepareContinuationRequest(prepared.pending, target, raw)).rejects.toThrow('uncertain');
    await expect(prepareContinuationRequest(state, { ...target, model: 'other' }, raw)).rejects.toThrow('destination changed');
    const lookup = await prepareReceiptLookup({ ...prepared.pending, phase: 'uncertain' });
    expect((lookup.request as Record<string, unknown>).contract).toBe('chat.continuation.receipt@0.1.0');
    expect((lookup.request as Record<string, unknown>).request_nonce).toBe(prepared.pending.request_nonce);
    expect((lookup.request as Record<string, unknown>).body).toBeUndefined();
    const receipt = readContinuationReceipt({ contract: 'chat.continuation@0.1.0', conversation_id: state.conversation_id, handle: 'ct1_' + 'd'.repeat(64), request_nonce: prepared.pending.request_nonce, history: { count: 2, sha256: '421ef8426f71a1940ee4253a040ca381b55417a7d49386bdaf21e70a41521a34' }, expires_at: Date.now() / 1000 + 300 }, prepared.pending);
    await expect(verifyContinuationHistory({ ...state, receipt }, [{ role: 'user', content: ' hi 🦊' }, { role: 'assistant', content: 'hello\n', reasoning_content: 'short' }, { role: 'user', content: 'next' }])).resolves.toBeUndefined();
    await expect(verifyContinuationHistory({ ...state, receipt }, [{ role: 'user', content: ' hi 🦊' }])).rejects.toThrow('missing or changed');
    expect(() => parseClientEnrollment(JSON.stringify({ ...enrollment, client: { ...enrollment.client, public_key: '0'.repeat(64) } }), target, publicKey!)).toThrow('browser key');
    expect(() => readContinuationReceipt({ ...receipt, request_nonce: 'f'.repeat(64) }, prepared.pending)).toThrow('mismatched');
  });
});
