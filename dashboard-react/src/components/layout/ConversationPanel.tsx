import { useState } from 'react';
import styled from 'styled-components';
import { DEFAULT_CONVERSATION_NAME, type Conversation } from '../../types/chat';
import { useSkulkTranslation, type SkulkTranslate } from '../../i18n/tolgee';

/* ── Types ────────────────────────────────────────────── */

/** Saved conversation selection and management shared by model and Steward chats. */
export interface ConversationPanelProps {
  conversations: Conversation[];
  activeConversationId: string | null;
  onSelect: (conversationId: string) => void;
  onDelete: (conversationId: string) => void;
  onNewChat: () => void;
  onRename?: (conversationId: string, name: string) => void;
  className?: string;
}

/* ── Helpers ──────────────────────────────────────────── */

function formatDate(ts: number, t: SkulkTranslate): string {
  const d = new Date(ts);
  const now = new Date();
  const diffMs = now.getTime() - ts;
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffDays === 0) {
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  if (diffDays === 1) return t('chat.sidebar.yesterday', 'Yesterday');
  if (diffDays < 7) return d.toLocaleDateString([], { weekday: 'long' });
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function modelLabel(modelId: string): string {
  if (modelId === 'skulk/steward') return '✦ Skulk';
  const parts = modelId.split('/');
  return parts[parts.length - 1];
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  return text.slice(0, max) + '...';
}

function conversationDisplayName(name: string, t: SkulkTranslate): string {
  return name === DEFAULT_CONVERSATION_NAME
    ? t('chat.conversation.newConversation', 'New conversation')
    : name;
}

/* ── Styles ───────────────────────────────────────────── */

const Panel = styled.aside`
  width: 340px;
  max-width: 100%;
  flex-shrink: 0;
  border-right: 1px solid ${({ theme }) => theme.colors.border};
  background: transparent;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const PanelHeader = styled.div`
  padding: 12px 16px;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.textSecondary};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const NewChatBtn = styled.button`
  all: unset;
  cursor: pointer;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.accentText};
  padding: 2px 8px;
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
  border-radius: ${({ theme }) => theme.radii.sm};
  transition: all 0.15s;

  &:hover {
    background: ${({ theme }) => theme.colors.goldBg};
  }
`;

const CardList = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
`;

const Card = styled.div<{ $active: boolean }>`
  all: unset;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 10px 12px;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid ${({ $active, theme }) => $active ? theme.colors.goldDim : 'transparent'};
  background: ${({ $active, theme }) => $active ? theme.colors.goldBg : 'transparent'};
  transition: all 0.15s;

  &:hover {
    background: ${({ theme }) => theme.colors.surfaceHover};
    border-color: ${({ theme }) => theme.colors.border};
  }
`;

const CardTitle = styled.button<{ $active: boolean }>`
  text-align: left; min-width: 0; width: 100%;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  font-weight: 500;
  color: ${({ $active, theme }) => $active ? theme.colors.gold : theme.colors.text};
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const CardMeta = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.subtleText};
`;

const Dot = styled.span`
  color: ${({ theme }) => theme.colors.subtleText};
`;

const CardSummary = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.subtleText};
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const DeleteBtn = styled.button`
  cursor: pointer;
  color: ${({ theme }) => theme.colors.subtleText};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  margin-left: auto;
  opacity: 0.7;
  transition: all 0.15s;

  ${Card}:hover & {
    opacity: 1;
  }

  &:hover {
    color: ${({ theme }) => theme.colors.error};
  }
`;

const EmptyText = styled.div`
  padding: 24px 16px;
  text-align: center;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.subtleText};
`;

const RenameForm = styled.form`
  display: flex; flex-wrap: wrap; gap: 8px;
  input { width: 100%; min-width: 0; padding: 6px 8px; border: 1px solid ${({ theme }) => theme.colors.borderControl}; border-radius: ${({ theme }) => theme.radii.sm}; background: ${({ theme }) => theme.colors.surface}; color: ${({ theme }) => theme.colors.text}; font: inherit; }
  button { padding: 4px 8px; border: 1px solid ${({ theme }) => theme.colors.borderControl}; border-radius: ${({ theme }) => theme.radii.sm}; }
`;

/* ── Component ────────────────────────────────────────── */

/** Render history with keyboard-accessible selection, rename and deletion. */
export function ConversationPanel({
  conversations,
  activeConversationId,
  onSelect,
  onDelete,
  onNewChat,
  onRename,
  className,
}: ConversationPanelProps) {
  const { t } = useSkulkTranslation();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [name, setName] = useState('');

  return (
    <Panel className={className}>
      <PanelHeader>
        {t('conversationPanel.history', 'History')}
        <NewChatBtn onClick={onNewChat}>{t('conversationPanel.new', '+ New')}</NewChatBtn>
      </PanelHeader>
      <CardList>
        {conversations.length === 0 ? (
          <EmptyText>{t('conversationPanel.empty', 'No conversations yet')}</EmptyText>
        ) : (
          conversations.map((convo) => {
            const active = convo.id === activeConversationId;
            const description = convo.summary
              ?? convo.messages.find((m) => m.role === 'user')?.content
              ?? '';
            return (
              <Card
                key={convo.id}
                $active={active}
                onClick={() => onSelect(convo.id)}
              >
                <CardTitle type="button" $active={active} onClick={event => { event.stopPropagation(); onSelect(convo.id); }}>
                  {truncate(conversationDisplayName(convo.name, t), 40)}
                </CardTitle>
                <CardMeta>
                  <span>{formatDate(convo.updatedAt, t)}</span>
                  <Dot>&middot;</Dot>
                  <span>{modelLabel(convo.modelId)}</span>
                  {onRename && <DeleteBtn type="button" aria-label={t('conversationPanel.rename', 'Rename conversation')} onClick={event => { event.stopPropagation(); setEditingId(convo.id); setName(convo.name); }}>✎</DeleteBtn>}
                  <DeleteBtn type="button" aria-label={t('conversationPanel.delete', 'Delete conversation')} onClick={(e) => { e.stopPropagation(); onDelete(convo.id); }}>
                    &times;
                  </DeleteBtn>
                </CardMeta>
                {editingId === convo.id && <RenameForm onClick={event => event.stopPropagation()} onSubmit={event => { event.preventDefault(); if (name.trim()) { onRename?.(convo.id, name.trim()); setEditingId(null); } }}>
                  <input aria-label={t('conversationPanel.name', 'Conversation name')} value={name} onChange={event => setName(event.target.value)} autoFocus onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); setEditingId(null); } }} />
                  <button type="submit" disabled={!name.trim()}>{t('common.save', 'Save')}</button>
                  <button type="button" onClick={() => setEditingId(null)}>{t('common.cancel', 'Cancel')}</button>
                </RenameForm>}
                {description && (
                  <CardSummary>{truncate(description, 60)}</CardSummary>
                )}
              </Card>
            );
          })
        )}
      </CardList>
    </Panel>
  );
}
