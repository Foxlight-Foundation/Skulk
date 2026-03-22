/**
 * MessageBubble
 *
 * Renders a single chat message with:
 * - Role indicator (user / assistant)
 * - Markdown rendering for assistant messages
 * - Thinking panel (collapsible)
 * - Token heatmap toggle
 * - Prefill progress bar
 * - Streaming cursor
 * - Generated image display
 * - Edit / copy / regenerate / delete actions
 * - Inline message editing
 *
 * Extracted from ChatMessages.svelte.
 */
import React, { useState, useRef, useEffect } from 'react';
import styled, { keyframes, css } from 'styled-components';
import { useTranslate } from '@tolgee/react';
import { MarkdownContent } from './MarkdownContent';
import { TokenHeatmap } from './TokenHeatmap';
import { PrefillProgressBar } from './PrefillProgressBar';
import { ImageLightbox } from './ImageLightbox';
import type { ChatMessage } from '../../api/types';
import { useChatStore } from '../../stores/chatStore';

// ─── Styled components ────────────────────────────────────────────────────────

const blink = keyframes`
  0%, 100% { opacity: 1; }
  50%       { opacity: 0; }
`;

const Wrapper = styled.div<{ $isUser: boolean }>`
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: ${({ $isUser }) => ($isUser ? 'flex-end' : 'flex-start')};
  padding: 4px 0;
`;

const Meta = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 10px;
  letter-spacing: 0.07em;
  color: ${({ theme }) => theme.colors.lightGray};
  opacity: 0.7;
`;

const Role = styled.span<{ $isUser: boolean }>`
  text-transform: uppercase;
  color: ${({ theme, $isUser }) => ($isUser ? theme.colors.lightGray : theme.colors.yellow)};
  font-weight: 700;
`;

const Timestamp = styled.span`
  font-size: 9px;
`;

const Perf = styled.span`
  font-size: 9px;
  color: ${({ theme }) => theme.colors.lightGray};
  opacity: 0.7;
`;

const Bubble = styled.div<{ $isUser: boolean }>`
  max-width: min(85%, 680px);
  padding: 10px 14px;
  border-radius: ${({ $isUser }) =>
    $isUser ? '12px 12px 2px 12px' : '12px 12px 12px 2px'};
  background: ${({ theme, $isUser }) =>
    $isUser ? theme.colors.mediumGray : theme.colors.darkGray};
  border: 1px solid ${({ theme }) => theme.colors.border};
  font-size: 13px;
  line-height: 1.6;
  position: relative;
`;

const StreamingCursor = styled.span`
  display: inline-block;
  width: 8px;
  height: 13px;
  background: ${({ theme }) => theme.colors.yellow};
  vertical-align: text-bottom;
  margin-left: 2px;
  animation: ${blink} 1s step-end infinite;
`;

const ThinkingToggle = styled.button`
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.lightGray};
  background: none;
  border: none;
  cursor: pointer;
  padding: 4px 0;
  transition: color ${({ theme }) => theme.transitions.fast};
  &:hover { color: ${({ theme }) => theme.colors.yellow}; }
`;

const ThinkingPanel = styled.div`
  margin: 6px 0;
  padding: 8px 12px;
  border-left: 2px solid ${({ theme }) => theme.colors.yellowDarker};
  background: oklch(0.15 0 0 / 0.6);
  border-radius: 0 ${({ theme }) => theme.radius.sm} ${({ theme }) => theme.radius.sm} 0;
  font-size: 12px;
  color: ${({ theme }) => theme.colors.lightGray};
  font-style: italic;
`;

const ImageWrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const GeneratedImg = styled.img`
  max-width: 100%;
  max-height: 400px;
  object-fit: contain;
  border-radius: ${({ theme }) => theme.radius.md};
  border: 1px solid ${({ theme }) => theme.colors.border};
  cursor: zoom-in;
`;

const Actions = styled.div`
  display: flex;
  gap: 4px;
  opacity: 0;
  transition: opacity ${({ theme }) => theme.transitions.fast};
  ${Wrapper}:hover & { opacity: 1; }
`;

const ActionBtn = styled.button`
  background: none;
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radius.sm};
  color: ${({ theme }) => theme.colors.lightGray};
  cursor: pointer;
  font-size: 10px;
  letter-spacing: 0.06em;
  padding: 3px 7px;
  text-transform: uppercase;
  transition: color ${({ theme }) => theme.transitions.fast},
    border-color ${({ theme }) => theme.transitions.fast};
  &:hover {
    color: ${({ theme }) => theme.colors.yellow};
    border-color: ${({ theme }) => theme.colors.yellowDarker};
  }
`;

const DestructiveBtn = styled(ActionBtn)`
  &:hover {
    color: ${({ theme }) => theme.colors.destructive};
    border-color: ${({ theme }) => theme.colors.destructive};
  }
`;

const EditArea = styled.textarea`
  width: 100%;
  background: ${({ theme }) => theme.colors.mediumGray};
  border: 1px solid ${({ theme }) => theme.colors.yellowDarker};
  border-radius: ${({ theme }) => theme.radius.md};
  color: ${({ theme }) => theme.colors.foreground};
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 13px;
  line-height: 1.6;
  padding: 8px 12px;
  resize: none;
  min-height: 60px;
  max-height: 200px;
  &:focus { outline: none; }
`;

const ConfirmRow = styled.div`
  display: flex;
  gap: 6px;
  margin-top: 4px;
`;

const PrimaryBtn = styled(ActionBtn)`
  background: ${({ theme }) => theme.colors.yellow};
  color: ${({ theme }) => theme.colors.black};
  border-color: ${({ theme }) => theme.colors.yellow};
  &:hover {
    background: ${({ theme }) => theme.colors.yellowDarker};
    color: ${({ theme }) => theme.colors.black};
    border-color: ${({ theme }) => theme.colors.yellowDarker};
  }
`;

// ─── Helpers ───────────────────────────────────────────────────────────────────

function formatTimestamp(ts: number): string {
  return new Date(ts).toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function extractThinkContent(text: string): { thinking: string; response: string } | null {
  const m = text.match(/<think>([\s\S]*?)<\/think>([\s\S]*)/);
  if (!m) return null;
  return { thinking: (m[1] ?? '').trim(), response: (m[2] ?? '').trim() };
}

function getTextContent(content: ChatMessage['content']): string {
  if (typeof content === 'string') return content;
  const first = content[0];
  if (!first) return '';
  return first.type === 'text' ? first.text : '';
}

// ─── Component ─────────────────────────────────────────────────────────────────

export interface MessageBubbleProps {
  message: ChatMessage;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const { t } = useTranslate();
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState('');
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [expandedThinking, setExpandedThinking] = useState(false);
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);
  const editRef = useRef<HTMLTextAreaElement>(null);

  const editMessage = useChatStore((s) => s.editMessage);
  const deleteMessageAndAfter = useChatStore((s) => s.deleteMessageAndAfter);

  const isUser = message.role === 'user';
  const textContent = getTextContent(message.content);
  const thinkSplit = !isUser ? extractThinkContent(textContent) : null;
  const displayText = thinkSplit ? thinkSplit.response : textContent;

  const handleStartEdit = () => {
    setEditContent(textContent);
    setIsEditing(true);
    setTimeout(() => {
      editRef.current?.focus();
    }, 10);
  };

  const handleSaveEdit = () => {
    editMessage(message.id, editContent);
    setIsEditing(false);
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(textContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { /* ignore */ }
  };

  return (
    <Wrapper $isUser={isUser}>
      <Meta>
        <Role $isUser={isUser}>
          {isUser ? t('chat.message_you') : t('chat.message_assistant')}
        </Role>
        <Timestamp>{formatTimestamp(message.createdAt)}</Timestamp>
        {!isUser && message.ttft != null && (
          <Perf>{t('chat.ttft', { ms: Math.round(message.ttft) })}</Perf>
        )}
        {!isUser && message.tps != null && (
          <Perf>{t('chat.tokens_per_second', { tps: message.tps.toFixed(1) })}</Perf>
        )}
      </Meta>

      {isEditing ? (
        <div style={{ maxWidth: 'min(85%, 680px)', width: '100%' }}>
          <EditArea
            ref={editRef}
            value={editContent}
            onChange={(e) => setEditContent(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleSaveEdit();
              if (e.key === 'Escape') setIsEditing(false);
            }}
          />
          <ConfirmRow>
            <PrimaryBtn onClick={handleSaveEdit}>{t('common.save')}</PrimaryBtn>
            <ActionBtn onClick={() => setIsEditing(false)}>{t('common.cancel')}</ActionBtn>
          </ConfirmRow>
        </div>
      ) : (
        <Bubble $isUser={isUser}>
          {/* Thinking panel */}
          {thinkSplit && thinkSplit.thinking && (
            <>
              <ThinkingToggle onClick={() => setExpandedThinking((p) => !p)}>
                {expandedThinking ? '▼' : '▶'}{' '}
                {t('chat.thinking')}
              </ThinkingToggle>
              {expandedThinking && (
                <ThinkingPanel>
                  <MarkdownContent content={thinkSplit.thinking} />
                </ThinkingPanel>
              )}
            </>
          )}

          {/* Prefill progress */}
          {message.isPrefilling && <PrefillProgressBar />}

          {/* Main content */}
          {message.imageUrl ? (
            <ImageWrapper>
              <GeneratedImg
                src={message.imageUrl}
                alt="Generated"
                onClick={() => setLightboxSrc(message.imageUrl!)}
              />
            </ImageWrapper>
          ) : isUser ? (
            <div style={{ whiteSpace: 'pre-wrap' }}>{displayText}</div>
          ) : (
            <MarkdownContent content={displayText} />
          )}

          {/* Streaming cursor */}
          {message.isStreaming && !message.isPrefilling && <StreamingCursor />}

          {/* Token heatmap */}
          {!isUser && message.tokenHeatmap && message.tokenHeatmap.length > 0 && (
            <>
              <ActionBtn
                style={{ marginTop: 6, display: 'block' }}
                onClick={() => setShowHeatmap((p) => !p)}
              >
                {showHeatmap ? 'Hide heatmap' : 'Show heatmap'}
              </ActionBtn>
              {showHeatmap && <TokenHeatmap tokens={message.tokenHeatmap} />}
            </>
          )}
        </Bubble>
      )}

      {/* Action bar */}
      {!isEditing && !message.isStreaming && (
        <Actions>
          <ActionBtn onClick={handleCopy} title={t('common.copy')}>
            {copied ? t('common.copied') : t('common.copy')}
          </ActionBtn>
          {isUser && (
            <ActionBtn onClick={handleStartEdit} title={t('chat.edit_message')}>
              {t('common.edit')}
            </ActionBtn>
          )}
          {!showDeleteConfirm ? (
            <DestructiveBtn onClick={() => setShowDeleteConfirm(true)}>
              {t('common.delete')}
            </DestructiveBtn>
          ) : (
            <>
              <DestructiveBtn onClick={() => deleteMessageAndAfter(message.id)}>
                {t('common.confirm')}
              </DestructiveBtn>
              <ActionBtn onClick={() => setShowDeleteConfirm(false)}>
                {t('common.cancel')}
              </ActionBtn>
            </>
          )}
        </Actions>
      )}

      {lightboxSrc && (
        <ImageLightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />
      )}
    </Wrapper>
  );
};
