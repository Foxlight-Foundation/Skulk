import { useMemo, useState } from 'react';
import styled, { css } from 'styled-components';
import type { Conversation } from '../../types/chat';
import { Button } from '../common/Button';

export interface ChatSidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  onNewChat: () => void;
  onSelectConversation: (id: string) => void;
  onRenameConversation: (id: string, name: string) => void;
  onDeleteConversation: (id: string) => void;
  onDeleteAllConversations: () => void;
  className?: string;
}

/* ---- helpers ---- */

function formatDate(ts: number): string {
  const now = new Date();
  const d = new Date(ts);
  const diffMs = now.getTime() - d.getTime();
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffDays === 0) {
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return d.toLocaleDateString([], { weekday: 'short' });
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

/* ---- styles ---- */

const Sidebar = styled.aside`
  display: flex;
  flex-direction: column;
  height: 100%;
  background: ${({ theme }) => theme.colors.surface};
  border-right: 1px solid rgba(255, 215, 0, 0.1);
  width: 280px;
  overflow: hidden;
`;

const Header = styled.div`
  padding: 16px;
`;


const SearchWrap = styled.div`
  padding: 0 16px 12px;
`;

const SearchInput = styled.input`
  all: unset;
  width: 100%;
  padding: 8px 10px;
  background: ${({ theme }) => theme.colors.bg};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  font-size: 12px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.text};
  box-sizing: border-box;

  &::placeholder { color: ${({ theme }) => theme.colors.textMuted}; }
  &:focus { border-color: rgba(255, 215, 0, 0.4); }
`;

const SectionLabel = styled.div`
  font-size: 10px;
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  letter-spacing: 2px;
  color: ${({ theme }) => theme.colors.textMuted};
  padding: 8px 16px 4px;
`;

const List = styled.div`
  flex: 1;
  overflow-y: auto;
`;

const ConvItem = styled.div<{ $active: boolean }>`
  padding: 10px 16px;
  cursor: pointer;
  transition: background 0.15s;
  position: relative;

  &:hover { background: ${({ theme }) => theme.colors.surfaceHover}; }

  ${({ $active }) =>
    $active &&
    css`
      background: rgba(255, 215, 0, 0.05);
      border-left: 2px solid #FFD700;
    `}
`;

const ConvName = styled.div<{ $active: boolean }>`
  font-size: 13px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ $active }) => ($active ? '#FFD700' : '#e5e5e5')};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const ConvDate = styled.div`
  font-size: 10px;
  color: ${({ theme }) => theme.colors.textMuted};
  margin-top: 2px;
`;

const HoverActions = styled.div`
  position: absolute;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  gap: 4px;
  opacity: 0;
  transition: opacity 0.15s;
  ${ConvItem}:hover & { opacity: 1; }
`;


const EditInput = styled.input`
  all: unset;
  width: 100%;
  padding: 4px 8px;
  background: ${({ theme }) => theme.colors.bg};
  border: 1px solid rgba(255, 215, 0, 0.4);
  border-radius: ${({ theme }) => theme.radii.sm};
  font-size: 12px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.text};
  box-sizing: border-box;
`;

const ConfirmBox = styled.div<{ $danger?: boolean }>`
  padding: 8px;
  margin-top: 4px;
  border: 1px solid ${({ $danger }) => ($danger ? 'rgba(239,68,68,0.3)' : 'rgba(255,215,0,0.2)')};
  border-radius: ${({ theme }) => theme.radii.sm};
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ $danger, theme }) => ($danger ? '#fca5a5' : theme.colors.textSecondary)};
`;

const BtnRow = styled.div`
  display: flex;
  gap: 6px;
  margin-top: 6px;
`;


const Footer = styled.div`
  border-top: 1px solid ${({ theme }) => theme.colors.border};
  padding: 12px 16px;
`;


const ConvCount = styled.div`
  font-size: 10px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.textMuted};
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-top: 8px;
`;

const EmptyState = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 32px 16px;
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: 12px;
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  letter-spacing: 1px;
`;

/* ---- component ---- */

export function ChatSidebar({
  conversations,
  activeId,
  onNewChat,
  onSelectConversation,
  onRenameConversation,
  onDeleteConversation,
  onDeleteAllConversations,
  className,
}: ChatSidebarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [showDeleteAll, setShowDeleteAll] = useState(false);

  const filtered = useMemo(() => {
    if (!searchQuery.trim()) return conversations;
    const q = searchQuery.toLowerCase();
    return conversations.filter((c) => c.name.toLowerCase().includes(q));
  }, [conversations, searchQuery]);

  const startRename = (id: string, name: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(id);
    setEditingName(name);
  };

  const saveRename = () => {
    if (editingId && editingName.trim()) {
      onRenameConversation(editingId, editingName.trim());
    }
    setEditingId(null);
  };

  const confirmDelete = (id: string) => {
    onDeleteConversation(id);
    setDeleteConfirmId(null);
  };

  return (
    <Sidebar className={className}>
      <Header>
        <Button variant="primary" size="md" block onClick={onNewChat}>+ New Chat</Button>
      </Header>

      <SearchWrap>
        <SearchInput
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search conversations…"
        />
      </SearchWrap>

      <SectionLabel>{searchQuery ? 'Search Results' : 'Conversations'}</SectionLabel>

      <List>
        {filtered.length === 0 ? (
          <EmptyState>
            {searchQuery ? 'No results' : 'No conversations'}
          </EmptyState>
        ) : (
          filtered.map((conv) => (
            <ConvItem
              key={conv.id}
              $active={conv.id === activeId}
              onClick={() => onSelectConversation(conv.id)}
            >
              {editingId === conv.id ? (
                <div>
                  <EditInput
                    value={editingName}
                    onChange={(e) => setEditingName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') saveRename();
                      if (e.key === 'Escape') setEditingId(null);
                    }}
                    autoFocus
                    onClick={(e) => e.stopPropagation()}
                  />
                  <BtnRow>
                    <Button variant="primary" size="sm" onClick={(e) => { e.stopPropagation(); saveRename(); }}>Save</Button>
                    <Button variant="outline" size="sm" onClick={(e) => { e.stopPropagation(); setEditingId(null); }}>Cancel</Button>
                  </BtnRow>
                </div>
              ) : deleteConfirmId === conv.id ? (
                <ConfirmBox $danger onClick={(e) => e.stopPropagation()}>
                  Delete "{conv.name}"?
                  <BtnRow>
                    <Button variant="danger" size="sm" onClick={(e) => { e.stopPropagation(); confirmDelete(conv.id); }}>Delete</Button>
                    <Button variant="outline" size="sm" onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(null); }}>Cancel</Button>
                  </BtnRow>
                </ConfirmBox>
              ) : (
                <>
                  <ConvName $active={conv.id === activeId}>{conv.name}</ConvName>
                  <ConvDate>{formatDate(conv.updatedAt)}</ConvDate>
                  <HoverActions>
                    <Button variant="ghost" size="sm" icon onClick={(e) => startRename(conv.id, conv.name, e)} title="Rename">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                      </svg>
                    </Button>
                    <Button variant="danger" size="sm" icon onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(conv.id); }} title="Delete">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                      </svg>
                    </Button>
                  </HoverActions>
                </>
              )}
            </ConvItem>
          ))
        )}
      </List>

      <Footer>
        {showDeleteAll ? (
          <ConfirmBox $danger>
            Delete all {conversations.length} conversations?
            <BtnRow>
              <Button variant="danger" size="sm" onClick={() => { onDeleteAllConversations(); setShowDeleteAll(false); }}>Delete All</Button>
              <Button variant="outline" size="sm" onClick={() => setShowDeleteAll(false)}>Cancel</Button>
            </BtnRow>
          </ConfirmBox>
        ) : conversations.length > 0 ? (
          <Button variant="danger" size="sm" onClick={() => setShowDeleteAll(true)}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
            </svg>
            Delete all chats
          </Button>
        ) : null}
        <ConvCount>{conversations.length} conversation{conversations.length !== 1 ? 's' : ''}</ConvCount>
      </Footer>
    </Sidebar>
  );
}
