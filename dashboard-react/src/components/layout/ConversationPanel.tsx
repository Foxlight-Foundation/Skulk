import styled from 'styled-components';
import type { Conversation } from '../../types/chat';

/* ── Types ────────────────────────────────────────────── */

export interface ConversationPanelProps {
  conversations: Conversation[];
  activeConversationId: string | null;
  onSelect: (conversationId: string) => void;
  onDelete: (conversationId: string) => void;
  onNewChat: () => void;
  className?: string;
}

/* ── Helpers ──────────────────────────────────────────── */

function formatDate(ts: number): string {
  const d = new Date(ts);
  const now = new Date();
  const diffMs = now.getTime() - ts;
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffDays === 0) {
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return d.toLocaleDateString([], { weekday: 'long' });
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function modelLabel(modelId: string): string {
  const parts = modelId.split('/');
  return parts[parts.length - 1];
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  return text.slice(0, max) + '...';
}

/* ── Styles ───────────────────────────────────────────── */

const Panel = styled.aside`
  width: 340px;
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
  color: ${({ theme }) => theme.colors.gold};
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

const Card = styled.button<{ $active: boolean }>`
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

const CardTitle = styled.div<{ $active: boolean }>`
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
  color: ${({ theme }) => theme.colors.textMuted};
`;

const Dot = styled.span`
  color: ${({ theme }) => theme.colors.textMuted};
`;

const CardSummary = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const DeleteBtn = styled.span`
  cursor: pointer;
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  margin-left: auto;
  opacity: 0;
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
  color: ${({ theme }) => theme.colors.textMuted};
`;

/* ── Component ────────────────────────────────────────── */

export function ConversationPanel({
  conversations,
  activeConversationId,
  onSelect,
  onDelete,
  onNewChat,
  className,
}: ConversationPanelProps) {
  return (
    <Panel className={className}>
      <PanelHeader>
        History
        <NewChatBtn onClick={onNewChat}>+ New</NewChatBtn>
      </PanelHeader>
      <CardList>
        {conversations.length === 0 ? (
          <EmptyText>No conversations yet</EmptyText>
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
                <CardTitle $active={active}>
                  {truncate(convo.name, 40)}
                </CardTitle>
                <CardMeta>
                  <span>{formatDate(convo.updatedAt)}</span>
                  <Dot>&middot;</Dot>
                  <span>{modelLabel(convo.modelId)}</span>
                  <DeleteBtn onClick={(e) => { e.stopPropagation(); onDelete(convo.id); }}>
                    &times;
                  </DeleteBtn>
                </CardMeta>
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
