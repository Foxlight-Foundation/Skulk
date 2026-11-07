import { useState } from 'react';
import { useStore } from 'react-redux';
import styled from 'styled-components';
import { Button } from '../common/Button';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import type { RootState } from '../../store';
import { chatActions, synchronizeChatPersistence } from '../../store/slices/chatSlice';
import { resolveRetainedChatEdit, type RetainedChatEdit } from '../../store/chatRetainedEdits';

const noEdits: RetainedChatEdit[] = [];
const Panel = styled.section`
  margin: 8px 0;
  padding: 12px;
  min-width: 0;
  max-width: 100%;
  max-height: 40vh;
  overflow: auto;
  overflow-wrap: anywhere;
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  h3 { margin: 0 0 8px; font-size: inherit; color: ${({ theme }) => theme.colors.text}; }
  article + article { margin-top: 12px; }
`;
const Actions = styled.div`display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px;`;
const Versions = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 20rem), 1fr));
  gap: 12px;
  margin-top: 12px;
  > div { min-width: 0; max-height: 35vh; overflow: auto; }
  pre { white-space: pre-wrap; overflow-wrap: anywhere; font-family: inherit; }
`;

function Messages({ value }: { value: unknown }) {
  if (!Array.isArray(value)) return null;
  return <>{(value as unknown[]).map((message, index) => {
    if (!message || typeof message !== 'object' || !('content' in message) || typeof message.content !== 'string') return null;
    return <pre key={index}>{message.content}</pre>;
  })}</>;
}

function RetainedVersion({ record, disabled, resolve }: {
  record: RetainedChatEdit;
  disabled: boolean;
  resolve: (id: string, choice: 'fork' | 'adopt') => void;
}) {
  const { t } = useSkulkTranslation();
  const [expanded, setExpanded] = useState(false);
  const savedExists = useAppSelector(state => !!state.chat.conversations[record.conversationId]);
  const saved = useAppSelector(state => expanded ? state.chat.conversations[record.conversationId] : undefined);
  const name = typeof record.conversation?.name === 'string' ? record.conversation.name : t('chat.retained.unnamed', 'Conversation');
  return <article>
    <strong>{name}</strong>
    {!record.conversation ? <p>{t('chat.retained.deletion', 'This tab requested deletion, but the saved conversation changed before that deletion could be committed.')}</p> : null}
    <Actions>
      {record.conversation ? <Button size="sm" variant="outline" onClick={() => setExpanded(value => !value)}>{t('chat.retained.review', 'Review versions')}</Button> : null}
      {record.conversation ? <Button size="sm" disabled={disabled} onClick={() => resolve(record.id, 'fork')}>{t('chat.retained.fork', 'Save as separate conversation')}</Button> : null}
      <Button size="sm" variant="outline" disabled={disabled} onClick={() => resolve(record.id, 'adopt')}>{savedExists ? t('chat.retained.adopt', 'Use saved version') : t('chat.retained.acceptDeletion', 'Accept deletion')}</Button>
    </Actions>
    {expanded ? <Versions>
      <div><strong>{t('chat.retained.local', 'Your retained messages')}</strong><Messages value={record.conversation?.messages} /></div>
      <div><strong>{t('chat.retained.saved', 'Saved messages')}</strong><Messages value={saved?.messages} /></div>
    </Versions> : null}
  </article>;
}

/** Review quarantined local intent and explicitly adopt or create a fresh context. */
export function ChatRetainedEdits({ disabled }: {
  /** Avoid changing the current selection while its active generation is running. */
  disabled: boolean;
}) {
  const { t } = useSkulkTranslation();
  const dispatch = useAppDispatch(), store = useStore<RootState>();
  const edits = useAppSelector(state => state.chat.retainedEdits ?? noEdits);
  const [busy, setBusy] = useState(false), [error, setError] = useState('');
  const resolve = async (id: string, choice: 'fork' | 'adopt') => {
    if (busy || disabled) return;
    const selectedBefore = store.getState().chat.activeConversationId;
    setBusy(true); setError('');
    try {
      const conversationId = await resolveRetainedChatEdit(id, choice);
      await synchronizeChatPersistence();
      // An asynchronous save must not steal focus from a newer user selection.
      if (conversationId && store.getState().chat.activeConversationId === selectedBefore) dispatch(chatActions.selectConversation(conversationId));
      dispatch(chatActions.setPersistenceError(undefined));
    } catch (error) {
      setError(error instanceof Error ? error.message : t('chat.retained.failed', 'Could not resolve these edits. Both versions were preserved.'));
      await synchronizeChatPersistence().catch(() => {});
    } finally { setBusy(false); }
  };
  if (!edits.length) return null;
  return <Panel aria-label={t('chat.retained.title', 'Retained local edits')}>
    <h3>{t('chat.retained.title', 'Retained local edits')} ({edits.length})</h3>
    <p>{t('chat.retained.help', 'These edits were saved separately after another tab changed the conversation. Review both versions. Saving separately creates a new conversation from your visible messages, with conversation continuity off.')}</p>
    {edits.map(record => <RetainedVersion key={record.id} record={record} disabled={disabled || busy} resolve={(id, choice) => { void resolve(id, choice); }} />)}
    {error ? <p role="alert">{error}</p> : null}
  </Panel>;
}
