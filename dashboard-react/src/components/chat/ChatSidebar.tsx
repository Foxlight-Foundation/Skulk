/**
 * ChatSidebar
 *
 * Conversation list with search, rename, delete, date formatting,
 * per-conversation stats, and mobile drawer mode.
 * Ported from ChatSidebar.svelte.
 */
import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { useTranslate } from '@tolgee/react';
import { useChatStore } from '../../stores/chatStore';
import { useTopologyStore } from '../../stores/topologyStore';
import { useUIStore } from '../../stores/uiStore';
import type { Conversation, Instance } from '../../api/types';

// ─── Styled components ────────────────────────────────────────────────────────

const SidebarEl = styled.aside`
  display: flex;
  flex-direction: column;
  height: 100%;
  background: ${({ theme }) => theme.colors.darkGray};
  border-right: 1px solid oklch(0.85 0.18 85 / 0.1);
`;

const SidebarHeader = styled.div`
  padding: 14px;
`;

const NewChatBtn = styled.button`
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 8px 14px;
  background: transparent;
  border: 1px solid oklch(0.85 0.18 85 / 0.3);
  color: ${({ theme }) => theme.colors.yellow};
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  letter-spacing: 0.1em;
  text-transform: uppercase;
  cursor: pointer;
  transition: border-color ${({ theme }) => theme.transitions.fast};
  &:hover { border-color: oklch(0.85 0.18 85 / 0.5); }
`;

const SearchWrapper = styled.div`
  padding: 10px 14px;
  position: relative;
`;

const SearchIcon = styled.svg`
  position: absolute;
  left: 24px;
  top: 50%;
  transform: translateY(-50%);
  width: 14px;
  height: 14px;
  color: oklch(1 0 0 / 0.5);
  pointer-events: none;
`;

const SearchInput = styled.input`
  width: 100%;
  background: oklch(0.08 0 0 / 0.4);
  border: 1px solid oklch(0.3 0 0 / 0.3);
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 6px 10px 6px 32px;
  font-size: 11px;
  color: oklch(1 0 0 / 0.9);
  &::placeholder { color: oklch(1 0 0 / 0.4); }
  &:focus { outline: none; border-color: oklch(0.85 0.18 85 / 0.3); }
`;

const ListContainer = styled.div`
  flex: 1;
  overflow-y: auto;
`;

const ListSection = styled.div`
  padding: 6px 0;
`;

const ListSectionLabel = styled.div`
  padding: 6px 14px;
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.lightGray};
`;

const ConvItemWrapper = styled.div`
  padding: 0 8px;
`;

const ConvItem = styled.div<{ $active: boolean }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px;
  border-radius: ${({ theme }) => theme.radius.md};
  margin-bottom: 2px;
  cursor: pointer;
  border: 1px solid ${({ theme, $active }) =>
    $active ? 'oklch(0.85 0.18 85 / 0.3)' : 'transparent'};
  background: ${({ $active }) =>
    $active ? 'oklch(0.85 0.18 85 / 0.04)' : 'transparent'};
  transition: all ${({ theme }) => theme.transitions.fast};
  &:hover {
    background: ${({ $active }) =>
      $active ? 'oklch(0.85 0.18 85 / 0.04)' : 'oklch(1 0 0 / 0.02)'};
    border-color: ${({ $active }) =>
      $active ? 'oklch(0.85 0.18 85 / 0.3)' : 'oklch(1 0 0 / 0.08)'};
  }
`;

const ConvInfo = styled.div`
  flex: 1;
  min-width: 0;
  padding-right: 6px;
`;

const ConvTitle = styled.div<{ $active: boolean }>`
  font-size: 13px;
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: ${({ $active }) => ($active ? 'oklch(0.85 0.18 85)' : 'oklch(1 0 0)')};
`;

const ConvMeta = styled.div`
  font-size: 11px;
  color: oklch(1 0 0 / 0.6);
  margin-top: 1px;
`;

const ConvModel = styled.div`
  font-size: 11px;
  color: ${({ theme }) => theme.colors.lightGray};
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const ConvStats = styled.div`
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: oklch(1 0 0 / 0.7);
  margin-top: 4px;
`;

const ConvActions = styled.div`
  display: flex;
  align-items: center;
  gap: 2px;
  opacity: 0;
  transition: opacity ${({ theme }) => theme.transitions.fast};
  ${ConvItem}:hover & { opacity: 1; }
`;

const ActionBtn = styled.button<{ $danger?: boolean }>`
  padding: 4px;
  background: none;
  border: none;
  cursor: pointer;
  color: ${({ theme }) => theme.colors.lightGray};
  border-radius: ${({ theme }) => theme.radius.sm};
  transition: color ${({ theme }) => theme.transitions.fast};
  &:hover {
    color: ${({ theme, $danger }) => ($danger ? '#f87171' : theme.colors.yellow)};
  }
`;

const InlineEditWrapper = styled.div`
  padding: 6px;
  background: transparent;
  border: 1px solid oklch(0.85 0.18 85 / 0.2);
  border-radius: ${({ theme }) => theme.radius.md};
  margin-bottom: 2px;
`;

const InlineInput = styled.input`
  width: 100%;
  background: oklch(0.08 0 0 / 0.6);
  border: 1px solid oklch(0.85 0.18 85 / 0.3);
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 5px 8px;
  font-size: 11px;
  color: ${({ theme }) => theme.colors.lightGray};
  margin-bottom: 6px;
  &:focus { outline: none; border-color: oklch(0.85 0.18 85 / 0.5); }
`;

const InlineBtnRow = styled.div`
  display: flex;
  gap: 6px;
`;

const InlineBtn = styled.button<{ $primary?: boolean }>`
  flex: 1;
  padding: 5px;
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  letter-spacing: 0.08em;
  text-transform: uppercase;
  cursor: pointer;
  border-radius: ${({ theme }) => theme.radius.sm};
  background: transparent;
  border: 1px solid ${({ theme, $primary }) =>
    $primary ? 'oklch(0.85 0.18 85 / 0.3)' : theme.colors.border};
  color: ${({ theme, $primary }) =>
    $primary ? theme.colors.yellow : theme.colors.lightGray};
  transition: all ${({ theme }) => theme.transitions.fast};
  &:hover {
    border-color: ${({ theme, $primary }) =>
      $primary ? 'oklch(0.85 0.18 85 / 0.5)' : theme.colors.lightGray};
  }
`;

const ConfirmWrapper = styled.div<{ $danger?: boolean }>`
  padding: 6px;
  background: ${({ $danger }) =>
    $danger ? 'oklch(0.4 0.15 15 / 0.1)' : 'transparent'};
  border: 1px solid ${({ $danger }) =>
    $danger ? 'oklch(0.4 0.15 15 / 0.3)' : 'oklch(0.25 0 0 / 0.3)'};
  border-radius: ${({ theme }) => theme.radius.md};
  margin-bottom: 2px;
`;

const ConfirmText = styled.p`
  font-size: 11px;
  color: oklch(0.7 0.15 15);
  margin: 0 0 6px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const EmptyState = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 16px;
  text-align: center;
  gap: 10px;
`;

const EmptyIcon = styled.div`
  width: 48px;
  height: 48px;
  border: 1px solid oklch(0.85 0.18 85 / 0.2);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const EmptyTitle = styled.p`
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: oklch(1 0 0 / 0.7);
  margin: 0;
`;

const EmptyHint = styled.p`
  font-size: 13px;
  color: oklch(1 0 0 / 0.5);
  margin: 0;
`;

const Footer = styled.div`
  padding: 10px;
  border-top: 1px solid oklch(0.85 0.18 85 / 0.1);
`;

const DeleteAllSection = styled.div`
  margin-bottom: 6px;
`;

const DeleteAllConfirm = styled.div`
  background: oklch(0.4 0.15 15 / 0.1);
  border: 1px solid oklch(0.4 0.15 15 / 0.3);
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 8px;
`;

const DeleteAllConfirmText = styled.p`
  font-size: 11px;
  color: oklch(0.7 0.15 15);
  text-align: center;
  margin: 0 0 6px;
`;

const DeleteAllBtn = styled.button`
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 5px;
  font-size: 13px;
  font-family: ${({ theme }) => theme.fonts.mono};
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: oklch(1 0 0 / 0.7);
  border: 1px solid transparent;
  background: none;
  cursor: pointer;
  border-radius: ${({ theme }) => theme.radius.sm};
  transition: all ${({ theme }) => theme.transitions.fast};
  &:hover {
    color: #f87171;
    background: oklch(0.4 0.15 15 / 0.1);
    border-color: oklch(0.4 0.15 15 / 0.2);
  }
`;

const FooterTools = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
`;

const FooterCount = styled.div`
  font-size: 11px;
  color: oklch(1 0 0 / 0.6);
  font-family: ${({ theme }) => theme.fonts.mono};
  letter-spacing: 0.1em;
  text-align: center;
`;

const ToolBtn = styled.button<{ $active?: boolean }>`
  padding: 5px;
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radius.sm};
  background: none;
  cursor: pointer;
  transition: border-color ${({ theme }) => theme.transitions.fast};
  color: ${({ theme, $active }) =>
    $active ? theme.colors.yellow : theme.colors.border};
  &:hover { border-color: oklch(0.85 0.18 85 / 0.5); }
`;

// Mobile overlay/drawer
const Overlay = styled.button`
  position: fixed;
  inset: 0;
  background: oklch(0 0 0 / 0.6);
  backdrop-filter: blur(4px);
  z-index: 40;
  border: none;
  cursor: pointer;
`;

const Drawer = styled.aside`
  position: fixed;
  left: 0;
  top: 0;
  bottom: 0;
  width: 288px;
  background: ${({ theme }) => theme.colors.darkGray};
  border-right: 1px solid oklch(0.85 0.18 85 / 0.1);
  z-index: 50;
  display: flex;
  flex-direction: column;
`;

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatDate(timestamp: number): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));
  if (diffDays === 0) return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return date.toLocaleDateString('en-US', { weekday: 'short' });
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function getLastAssistantStats(conversation: Conversation): { ttft?: number; tps?: number } | null {
  for (let i = conversation.messages.length - 1; i >= 0; i--) {
    const msg = conversation.messages[i];
    if (msg != null && msg.role === 'assistant' && (msg.ttft || msg.ttftMs || msg.tps)) {
      return { ttft: msg.ttft ?? msg.ttftMs, tps: msg.tps };
    }
  }
  return null;
}

function formatModelName(modelId: string | null | undefined): string {
  if (!modelId) return 'Unknown Model';
  const parts = modelId.split('/');
  return parts[parts.length - 1] ?? modelId;
}

function getTaggedValue(obj: unknown): [string | null, unknown] {
  if (!obj || typeof obj !== 'object') return [null, null];
  const keys = Object.keys(obj as Record<string, unknown>);
  if (keys.length === 1) {
    const key = keys[0];
    if (key == null) return [null, null];
    return [key, (obj as Record<string, unknown>)[key]];
  }
  return [null, null];
}

function extractInstanceModelId(instanceWrapped: unknown): string | null {
  const [, instance] = getTaggedValue(instanceWrapped);
  if (!instance || typeof instance !== 'object') return null;
  const inst = instance as { shardAssignments?: { modelId?: string } };
  return inst.shardAssignments?.modelId ?? null;
}

function describeInstance(instanceWrapped: unknown): { sharding: string | null; instanceType: string | null } {
  const [instanceTag, instance] = getTaggedValue(instanceWrapped);
  if (!instance || typeof instance !== 'object') return { sharding: null, instanceType: null };
  let instanceType: string | null = null;
  if (instanceTag === 'MlxRingInstance') instanceType = 'MLX Ring';
  else if (instanceTag === 'MlxJacclInstance') instanceType = 'MLX RDMA';
  const inst = instance as { shardAssignments?: { runnerToShard?: Record<string, unknown> } };
  const runnerToShard = inst.shardAssignments?.runnerToShard || {};
  const firstShardWrapped = Object.values(runnerToShard)[0];
  let sharding: string | null = null;
  if (firstShardWrapped) {
    const [shardTag] = getTaggedValue(firstShardWrapped);
    if (shardTag === 'PipelineShardMetadata') sharding = 'Pipeline';
    else if (shardTag === 'TensorShardMetadata') sharding = 'Tensor';
    else if (shardTag === 'PrefillDecodeShardMetadata') sharding = 'Prefill/Decode';
  }
  return { sharding, instanceType };
}

// ─── Component ─────────────────────────────────────────────────────────────────

export interface ChatSidebarProps {
  onNewChat?: () => void;
  onSelectConversation?: () => void;
  isMobileDrawer?: boolean;
  isOpen?: boolean;
  onClose?: () => void;
  className?: string;
}

export const ChatSidebar: React.FC<ChatSidebarProps> = ({
  onNewChat,
  onSelectConversation,
  isMobileDrawer = false,
  isOpen = false,
  onClose,
  className,
}) => {
  const { t } = useTranslate();
  const conversations = useChatStore((s) => s.conversations);
  const activeConversationId = useChatStore((s) => s.activeConversationId);
  const setActiveConversation = useChatStore((s) => s.setActiveConversation);
  const deleteConversation = useChatStore((s) => s.deleteConversation);
  const clearAllConversations = useChatStore((s) => s.clearAllConversations);
  const renameConversation = useChatStore((s) => s.renameConversation);
  const instances = useTopologyStore((s) => s.instances);
  const debugMode = useUIStore((s) => s.debugMode);
  const toggleDebugMode = useUIStore((s) => s.toggleDebugMode);
  const topologyOnlyMode = useUIStore((s) => s.topologyOnlyMode);
  const toggleTopologyOnlyMode = useUIStore((s) => s.toggleTopologyOnlyMode);

  const [searchQuery, setSearchQuery] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [showDeleteAllConfirm, setShowDeleteAllConfirm] = useState(false);

  const filtered = searchQuery.trim()
    ? conversations.filter((c) =>
        c.title.toLowerCase().includes(searchQuery.toLowerCase()),
      )
    : conversations;

  function handleSelect(id: string) {
    onSelectConversation?.();
    setActiveConversation(id);
    if (isMobileDrawer && isOpen) onClose?.();
  }

  function startEdit(id: string, title: string, e: React.MouseEvent) {
    e.stopPropagation();
    setEditingId(id);
    setEditingName(title);
    setDeleteConfirmId(null);
  }

  function saveEdit() {
    if (editingId && editingName.trim()) renameConversation(editingId, editingName.trim());
    setEditingId(null);
    setEditingName('');
  }

  function cancelEdit() {
    setEditingId(null);
    setEditingName('');
  }

  function resolveConvInfo(conv: Conversation) {
    let matchedInstance: unknown = null;
    let modelId = conv.modelId ?? null;
    if (modelId) {
      for (const [, wrap] of Object.entries(instances as Record<string, unknown>)) {
        if (extractInstanceModelId(wrap) === modelId) { matchedInstance = wrap; break; }
      }
    }
    if (!matchedInstance) {
      const first = Object.values(instances as Record<string, unknown>)[0];
      if (first) { matchedInstance = first; modelId = modelId ?? extractInstanceModelId(first); }
    }
    const details = matchedInstance ? describeInstance(matchedInstance) : { sharding: null, instanceType: null };
    const sharding = conv.sharding ?? details.sharding ?? 'Unknown';
    const instanceType = conv.instanceType ?? details.instanceType;
    const strategyLabel = instanceType ? `${sharding} (${instanceType})` : sharding;
    return { modelLabel: formatModelName(modelId), strategyLabel };
  }

  const sidebarContent = (
    <>
      {/* Header */}
      <SidebarHeader>
        <NewChatBtn type="button" onClick={() => onNewChat?.()}>
          <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
          </svg>
          New Chat
        </NewChatBtn>
      </SidebarHeader>

      {/* Search */}
      <SearchWrapper>
        <SearchIcon fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </SearchIcon>
        <SearchInput
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search conversations..."
        />
      </SearchWrapper>

      {/* List */}
      <ListContainer>
        {filtered.length > 0 ? (
          <ListSection>
            <ListSectionLabel>
              {searchQuery ? 'Search Results' : 'Conversations'}
            </ListSectionLabel>
            {filtered.map((conv) => {
              const { modelLabel } = resolveConvInfo(conv);
              const stats = getLastAssistantStats(conv);
              const isActive = conv.id === activeConversationId;
              if (editingId === conv.id) {
                return (
                  <ConvItemWrapper key={conv.id}>
                    <InlineEditWrapper>
                      <InlineInput
                        type="text"
                        value={editingName}
                        onChange={(e) => setEditingName(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') saveEdit();
                          else if (e.key === 'Escape') cancelEdit();
                        }}
                        autoFocus
                      />
                      <InlineBtnRow>
                        <InlineBtn type="button" $primary onClick={saveEdit}>Save</InlineBtn>
                        <InlineBtn type="button" onClick={cancelEdit}>Cancel</InlineBtn>
                      </InlineBtnRow>
                    </InlineEditWrapper>
                  </ConvItemWrapper>
                );
              }
              if (deleteConfirmId === conv.id) {
                return (
                  <ConvItemWrapper key={conv.id}>
                    <ConfirmWrapper $danger>
                      <ConfirmText>Delete &ldquo;{conv.title}&rdquo;?</ConfirmText>
                      <InlineBtnRow>
                        <InlineBtn
                          type="button"
                          style={{ color: '#f87171', borderColor: 'oklch(0.4 0.15 15 / 0.3)' }}
                          onClick={() => { deleteConversation(conv.id); setDeleteConfirmId(null); }}
                        >
                          Delete
                        </InlineBtn>
                        <InlineBtn type="button" onClick={() => setDeleteConfirmId(null)}>Cancel</InlineBtn>
                      </InlineBtnRow>
                    </ConfirmWrapper>
                  </ConvItemWrapper>
                );
              }
              return (
                <ConvItemWrapper key={conv.id}>
                  <ConvItem
                    $active={isActive}
                    role="button"
                    tabIndex={0}
                    onClick={() => handleSelect(conv.id)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSelect(conv.id)}
                  >
                    <ConvInfo>
                      <ConvTitle $active={isActive}>{conv.title}</ConvTitle>
                      <ConvMeta>{formatDate(conv.updatedAt)}</ConvMeta>
                      <ConvModel>{modelLabel}</ConvModel>
                      {stats && (
                        <ConvStats>
                          {stats.ttft != null && (
                            <><span style={{ color: 'oklch(1 0 0 / 0.5)' }}>TTFT</span>{' '}
                            <span style={{ color: 'oklch(0.85 0.18 85 / 0.8)' }}>{stats.ttft.toFixed(0)}ms</span></>
                          )}
                          {stats.ttft != null && stats.tps != null && <span style={{ opacity: 0.3, margin: '0 5px' }}>·</span>}
                          {stats.tps != null && (
                            <><span style={{ color: 'oklch(0.85 0.18 85 / 0.8)' }}>{stats.tps.toFixed(1)}</span>{' '}
                            <span style={{ color: 'oklch(1 0 0 / 0.5)' }}>tok/s</span></>
                          )}
                        </ConvStats>
                      )}
                    </ConvInfo>
                    <ConvActions>
                      <ActionBtn
                        type="button"
                        onClick={(e) => startEdit(conv.id, conv.title, e)}
                        title="Rename"
                      >
                        <svg width="12" height="12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                        </svg>
                      </ActionBtn>
                      <ActionBtn
                        type="button"
                        $danger
                        onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(conv.id); }}
                        title="Delete"
                      >
                        <svg width="12" height="12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </ActionBtn>
                    </ConvActions>
                  </ConvItem>
                </ConvItemWrapper>
              );
            })}
          </ListSection>
        ) : (
          <EmptyState>
            <EmptyIcon>
              <svg width="24" height="24" fill="none" viewBox="0 0 24 24" stroke="currentColor" style={{ color: 'oklch(0.85 0.18 85 / 0.4)' }}>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </EmptyIcon>
            <EmptyTitle>{searchQuery ? 'No Results' : 'No Conversations'}</EmptyTitle>
            <EmptyHint>{searchQuery ? 'Try a different search' : 'Start a new chat to begin'}</EmptyHint>
          </EmptyState>
        )}
      </ListContainer>

      {/* Footer */}
      <Footer>
        {conversations.length > 0 && (
          <DeleteAllSection>
            {showDeleteAllConfirm ? (
              <DeleteAllConfirm>
                <DeleteAllConfirmText>
                  Delete all {conversations.length} conversations?
                </DeleteAllConfirmText>
                <InlineBtnRow>
                  <InlineBtn
                    type="button"
                    style={{ color: '#f87171', borderColor: 'oklch(0.4 0.15 15 / 0.3)' }}
                    onClick={() => { clearAllConversations(); setShowDeleteAllConfirm(false); }}
                  >
                    Delete All
                  </InlineBtn>
                  <InlineBtn type="button" onClick={() => setShowDeleteAllConfirm(false)}>Cancel</InlineBtn>
                </InlineBtnRow>
              </DeleteAllConfirm>
            ) : (
              <DeleteAllBtn type="button" onClick={() => setShowDeleteAllConfirm(true)}>
                <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                Delete All Chats
              </DeleteAllBtn>
            )}
          </DeleteAllSection>
        )}
        <FooterTools>
          <ToolBtn
            type="button"
            $active={debugMode}
            onClick={toggleDebugMode}
            title="Toggle debug mode"
          >
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
              <path d="M19 8h-1.81A6.002 6.002 0 0 0 12 2a6.002 6.002 0 0 0-5.19 3H5a1 1 0 0 0 0 2h1v2H5a1 1 0 0 0 0 2h1v2H5a1 1 0 0 0 0 2h1.81A6.002 6.002 0 0 0 12 22a6.002 6.002 0 0 0 5.19-3H19a1 1 0 0 0 0-2h-1v-2h1a1 1 0 0 0 0-2h-1v-2h1a1 1 1 0 1 0 0-2Zm-5 10.32V19a1 1 0 1 1-2 0v-.68a3.999 3.999 0 0 1-3-3.83V9.32a3.999 3.999 0 0 1 3-3.83V5a1 1 0 0 1 2 0v.49a3.999 3.999 0 0 1 3 3.83v5.17a3.999 3.999 0 0 1-3 3.83Z" />
            </svg>
          </ToolBtn>
          <FooterCount>
            {conversations.length} Conversation{conversations.length !== 1 ? 's' : ''}
          </FooterCount>
          <ToolBtn
            type="button"
            $active={topologyOnlyMode}
            onClick={toggleTopologyOnlyMode}
            title="Toggle topology only mode"
          >
            <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <circle cx="12" cy="5" r="2" fill="currentColor" />
              <circle cx="5" cy="19" r="2" fill="currentColor" />
              <circle cx="19" cy="19" r="2" fill="currentColor" />
              <path strokeLinecap="round" d="M12 7v5m0 0l-5 5m5-5l5 5" />
            </svg>
          </ToolBtn>
        </FooterTools>
      </Footer>
    </>
  );

  if (isMobileDrawer) {
    if (!isOpen) return null;
    return (
      <>
        <Overlay onClick={() => onClose?.()} aria-label="Close sidebar" />
        <Drawer>{sidebarContent}</Drawer>
      </>
    );
  }

  return <SidebarEl className={className}>{sidebarContent}</SidebarEl>;
};
