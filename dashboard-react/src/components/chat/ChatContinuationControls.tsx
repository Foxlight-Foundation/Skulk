import { useState } from 'react';
import styled from 'styled-components';
import { Button } from '../common/Button';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import { useStore } from 'react-redux';
import type { RootState } from '../../store';
import { chatActions, persistChatContinuation } from '../../store/slices/chatSlice';
import { browserContinuationPublicKey, continuationNonce, parseClientEnrollment, prepareReceiptLookup, readContinuationReceipt, verifyContinuationHistory, type ChatContinuation, type ContinuationTarget } from '../../store/chatContinuation';
import { acquireChatConversation } from '../../store/chatCoordination';
import { synchronizeChatPersistence } from '../../store/slices/chatSlice';
import { hydrateChatDocuments } from '../../store/chatDocuments';
import { buildApiMessages } from '../pages/chatApiPayload';
import { operatorSession } from '../../auth/operatorSession';

const Panel = styled.details`
  min-width: 0;
  max-width: 100%;
  overflow-wrap: anywhere;
  max-height: 40vh;
  overflow: auto;
  padding: 8px 0;
  color: ${({ theme }) => theme.colors.textSecondary};
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  summary { cursor: pointer; }
  input, textarea { width: 100%; padding: 8px; background: ${({ theme }) => theme.colors.surface}; color: ${({ theme }) => theme.colors.text}; border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: ${({ theme }) => theme.radii.md}; }
  textarea { min-height: 80px; }
  label { display: block; margin: 8px 0; }
  button { margin: 4px 8px 4px 0; }
`;

/** Explicit browser-key enrollment and recovery for one model-owned conversation. */
export function ChatContinuationControls({ conversationId, target, disabled }: {
  /** Chat that owns this authority even while another model is selected. */
  conversationId: string;
  /** Current exact generic destination advertised by the selected model. */
  target?: ContinuationTarget;
  /** Prevent authority changes while a generation owns its current scope. */
  disabled: boolean;
}) {
  const { t } = useSkulkTranslation();
  const dispatch = useAppDispatch(), store = useStore<RootState>();
  const state = useAppSelector(root => root.chat.conversations[conversationId]?.continuation);
  const [publicKey, setPublicKey] = useState(''), [enrollment, setEnrollment] = useState('');
  const [message, setMessage] = useState(''), [busy, setBusy] = useState(false);
  const save = async (continuation?: ChatContinuation) => {
    const expected = store.getState().chat.conversations[conversationId]?.continuation;
    const release = await acquireChatConversation(conversationId);
    try {
      await synchronizeChatPersistence();
      if (JSON.stringify(store.getState().chat.conversations[conversationId]?.continuation) !== JSON.stringify(expected)) throw new Error('The conversation changed. Review its refreshed recovery state first.');
      await persistChatContinuation(store.getState().chat, conversationId, continuation);
      dispatch(chatActions.setConversationContinuation({ conversationId, continuation }));
    } finally { await release(); }
  };
  const action = async (work: () => Promise<void>) => {
    if (busy || disabled) return;
    setBusy(true);
    try { await work(); } catch (error) { setMessage(error instanceof Error ? error.message : 'Client setup failed.'); }
    finally { setBusy(false); }
  };
  const phaseLabel = state?.phase === 'ready' ? t('chat.continuation.ready', 'Ready')
    : state?.phase === 'pending' && disabled ? t('chat.continuation.inProgress', 'In progress')
    : state ? t('chat.continuation.needsRecovery', 'Needs recovery') : t('chat.continuation.off', 'Off');
  return <Panel>
    <summary>{t('chat.continuation.title', 'Conversation continuity')} · {phaseLabel}</summary>
    {!state && target && <>
    <p>{t('chat.continuation.help', 'Opt in to protected continuation. Generate this browser’s public key, approve it under Continuation clients in the integration, then paste its public enrollment here. The private key stays in this browser.')}</p>
    <Button size="sm" disabled={disabled || busy} onClick={() => { void action(async () => { setPublicKey(await browserContinuationPublicKey(true) ?? ''); }); }}>{t('chat.continuation.key', 'Get browser public key')}</Button>
    {publicKey && <label>{t('chat.continuation.publicKey', 'Browser public key')}<input aria-label={t('chat.continuation.publicKey', 'Browser public key')} value={publicKey} readOnly /></label>}
    <label>{t('chat.continuation.enrollment', 'Public enrollment JSON')}<textarea aria-label={t('chat.continuation.enrollment', 'Public enrollment JSON')} value={enrollment} maxLength={65536} onChange={event => setEnrollment(event.target.value)} disabled={disabled || busy} /></label>
    <Button size="sm" disabled={disabled || busy || !enrollment || !!state} onClick={() => { void action(async () => {
      const key = await browserContinuationPublicKey();
      if (!key) throw new Error('Generate this browser’s key and obtain owner approval first.');
      if (!target) throw new Error('The destination no longer advertises continuation.');
      await save({ enrollment: parseClientEnrollment(enrollment, target, key), target, conversation_id: continuationNonce(), phase: 'ready' });
      setEnrollment('');
      setMessage(t('chat.continuation.enabled', 'Continuation enabled for this conversation.'));
    }); }}>{t('chat.continuation.enable', 'Enable for this conversation')}</Button>
    </>}
    {state && <>
      <p>{t('chat.continuation.expiry', 'Client access expires {date}.', { date: new Date(state.enrollment.client.expires_at * 1000).toLocaleString() })}</p>
      {state.phase !== 'ready' && !disabled && <p role="status">{t('chat.continuation.uncertain', 'Delivery is uncertain. Receipt recovery never regenerates a response or recovers missing assistant text.')}</p>}
      <Button size="sm" disabled={disabled || busy || !state.request_nonce} onClick={() => { void action(async () => {
        const proof = await prepareReceiptLookup(state);
        const response = await operatorSession.fetch('/v1/chat/continuation/receipt', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Skulk-Dashboard': 'pairing-v1' }, body: JSON.stringify(proof) });
        if (!response.ok) throw new Error('Receipt lookup could not complete. No inference was retried.');
        const outcome = await response.json() as { receipt?: unknown; error?: string };
        if (!outcome.receipt) throw new Error(`Receipt unavailable: ${outcome.error ?? 'unknown'}. No inference was retried.`);
        const recovered = { ...state, receipt: readContinuationReceipt(outcome.receipt, state) };
        const conversation = store.getState().chat.conversations[conversationId];
        if (!conversation) return;
        const messages = buildApiMessages(await hydrateChatDocuments(conversation.messages), { text: true, pdf: true });
        await verifyContinuationHistory(recovered, messages);
        if (store.getState().chat.conversations[conversationId]?.continuation !== state) {
          throw new Error('The conversation changed during receipt lookup. Its current recovery state was preserved.');
        }
        await save({ ...recovered, phase: 'ready' });
        setMessage(t('chat.continuation.recovered', 'Receipt recovered and public history verified. No inference was retried.'));
      }); }}>{t('chat.continuation.recover', 'Recover receipt')}</Button>
      <Button size="sm" variant="outline" disabled={disabled || busy} onClick={() => { void action(async () => {
        await save();
        setMessage(t('chat.continuation.reset', 'Continuation disabled. The next request starts a fresh context with visible history. Client approval remains managed by the integration.'));
      }); }}>{t('chat.continuation.resetButton', 'Use full history instead')}</Button>
    </>}
    {message && <p role="status">{message}</p>}
  </Panel>;
}
