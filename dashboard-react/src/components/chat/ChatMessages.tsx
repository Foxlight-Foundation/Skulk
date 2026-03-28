import { useCallback, useEffect, useRef, useState } from 'react';
import styled, { css, keyframes } from 'styled-components';
import type { ChatMessage } from '../../types/chat';
import { getFileIcon } from '../../types/chat';
import { MarkdownContent } from '../display/MarkdownContent';
import { TokenHeatmap } from '../display/TokenHeatmap';
import { PrefillProgressBar, type PrefillProgress } from '../display/PrefillProgressBar';
import { ImageLightbox } from '../display/ImageLightbox';
import { Button } from '../common/Button';

export interface ChatMessagesProps {
  messages: ChatMessage[];
  /** Current streaming response text, or null if not streaming. */
  streamingContent?: string | null;
  isLoading?: boolean;
  prefillProgress?: PrefillProgress | null;
  onDelete?: (messageId: string) => void;
  onEdit?: (messageId: string, content: string) => void;
  onRegenerate?: () => void;
  onRegenerateFromToken?: (tokenIndex: number) => void;
  className?: string;
}

/* ---- helpers ---- */

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

/* ---- animations ---- */

const blink = keyframes`
  0%, 100% { opacity: 1; }
  50%      { opacity: 0; }
`;

/* ---- styles ---- */

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
  position: relative;
`;

const EmptyState = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 0;
  gap: 16px;
  color: ${({ theme }) => theme.colors.textMuted};
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
`;

const Circle = styled.div<{ $size: number; $opacity: number }>`
  position: absolute;
  width: ${({ $size }) => $size}px;
  height: ${({ $size }) => $size}px;
  border-radius: 50%;
  border: 1px solid rgba(255, 215, 0, ${({ $opacity }) => $opacity});
`;

const MessageCard = styled.div<{ $role: 'user' | 'assistant' }>`
  padding: 12px 16px;
  border-radius: ${({ theme }) => theme.radii.lg};
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  position: relative;

  ${({ $role }) =>
    $role === 'user'
      ? css`
          align-self: flex-end;
          max-width: 70%;
          border-color: ${({ theme }) => theme.colors.border};
        `
      : css`
          border-left: 2px solid rgba(255, 215, 0, 0.3);
        `}
`;

const MsgHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
`;

const RoleLabel = styled.span<{ $role: 'user' | 'assistant' }>`
  color: ${({ $role }) => ($role === 'assistant' ? '#FFD700' : '#999')};
  font-weight: 600;
`;

const Timestamp = styled.span`
  color: ${({ theme }) => theme.colors.textMuted};
`;

const StatLabel = styled.span`
  color: ${({ theme }) => theme.colors.textMuted};
  font-variant-numeric: tabular-nums;
  & > span { color: rgba(255, 215, 0, 0.7); }
`;

const Dot = styled.span<{ $color: string }>`
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: ${({ $color }) => $color};
`;

const Spacer = styled.span`flex: 1;`;

const UserContent = styled.div`
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.md};
  white-space: pre-wrap;
  line-height: 1.5;
  color: ${({ theme }) => theme.colors.text};
`;

const Cursor = styled.span`
  display: inline-block;
  width: 8px;
  height: 16px;
  background: #FFD700;
  margin-left: 2px;
  vertical-align: text-bottom;
  animation: ${blink} 0.8s step-end infinite;
`;

const Actions = styled.div`
  display: flex;
  gap: 4px;
  margin-top: 8px;
  opacity: 0;
  transition: opacity 0.15s;
  ${MessageCard}:hover & { opacity: 1; }
`;

const ActiveGhostBtn = styled(Button)<{ $active?: boolean }>`
  ${({ $active }) =>
    $active &&
    css`
      color: #FFD700;
      background: rgba(255, 215, 0, 0.1);
    `}
`;

const EditArea = styled.textarea`
  all: unset;
  width: 100%;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.md};
  color: ${({ theme }) => theme.colors.text};
  background: ${({ theme }) => theme.colors.bg};
  border: 1px solid rgba(255, 215, 0, 0.4);
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 8px;
  resize: none;
  min-height: 40px;
  max-height: 200px;
  box-sizing: border-box;
`;

const BtnRow = styled.div`
  display: flex;
  gap: 6px;
  margin-top: 6px;
`;


const ConfirmBox = styled.div`
  padding: 8px;
  margin-top: 8px;
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: ${({ theme }) => theme.radii.sm};
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-family: ${({ theme }) => theme.fonts.body};
  color: #fca5a5;
`;


const ThinkingBlock = styled.div<{ $open: boolean }>`
  margin: 8px 0;
  border: 1px solid rgba(255, 215, 0, 0.15);
  border-radius: ${({ theme }) => theme.radii.md};
  overflow: hidden;
`;

const ThinkingHeader = styled.button`
  all: unset;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 6px 10px;
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-family: ${({ theme }) => theme.fonts.body};
  color: rgba(255, 215, 0, 0.6);
  transition: background 0.15s;
  box-sizing: border-box;
  &:hover { background: rgba(255, 215, 0, 0.05); }
`;

const ThinkingChevron = styled.span<{ $open: boolean }>`
  transition: transform 0.15s;
  ${({ $open }) => $open && css`transform: rotate(90deg);`}
`;

const ThinkingContent = styled.div`
  padding: 8px 10px;
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  color: ${({ theme }) => theme.colors.textSecondary};
  border-top: 1px solid rgba(255, 215, 0, 0.1);
`;

const ImageGrid = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 8px 0;
`;

const GenImage = styled.img`
  max-width: 256px;
  max-height: 256px;
  border-radius: ${({ theme }) => theme.radii.md};
  cursor: pointer;
  border: 1px solid ${({ theme }) => theme.colors.border};
  transition: border-color 0.15s;
  &:hover { border-color: rgba(255, 215, 0, 0.4); }
`;

const AttachmentRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 8px;
`;

const AttachThumb = styled.img`
  width: 48px;
  height: 48px;
  object-fit: cover;
  border-radius: ${({ theme }) => theme.radii.sm};
  cursor: pointer;
  border: 1px solid rgba(255, 215, 0, 0.2);
`;

const AttachFile = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const ScrollBtn = styled.button`
  all: unset;
  cursor: pointer;
  position: sticky;
  bottom: 16px;
  align-self: center;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid rgba(255, 215, 0, 0.3);
  color: #FFD700;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
  transition: all 0.15s;

  &:hover {
    background: rgba(255, 215, 0, 0.1);
    border-color: rgba(255, 215, 0, 0.5);
  }
`;

/* ---- component ---- */

export function ChatMessages({
  messages,
  streamingContent,
  isLoading = false,
  prefillProgress,
  onDelete,
  onEdit,
  onRegenerate,
  onRegenerateFromToken,
  className,
}: ChatMessagesProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState('');
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [expandedThinking, setExpandedThinking] = useState<Set<string>>(new Set());
  const [heatmapVisible, setHeatmapVisible] = useState<Set<string>>(new Set());
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);
  const lastCountRef = useRef(messages.length);

  // Auto-scroll on new messages
  useEffect(() => {
    if (messages.length > lastCountRef.current) {
      containerRef.current?.parentElement?.scrollTo({ top: containerRef.current.parentElement.scrollHeight, behavior: 'smooth' });
    }
    lastCountRef.current = messages.length;
  }, [messages.length]);

  // Scroll button visibility
  useEffect(() => {
    const parent = containerRef.current?.parentElement;
    if (!parent) return;
    const handler = () => {
      const dist = parent.scrollHeight - parent.scrollTop - parent.clientHeight;
      setShowScrollBtn(dist > 100);
    };
    parent.addEventListener('scroll', handler, { passive: true });
    return () => parent.removeEventListener('scroll', handler);
  }, []);

  const scrollToBottom = useCallback(() => {
    containerRef.current?.parentElement?.scrollTo({ top: containerRef.current.parentElement.scrollHeight, behavior: 'smooth' });
  }, []);

  const copyMessage = useCallback((id: string, content: string) => {
    navigator.clipboard.writeText(content);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  }, []);

  const saveEdit = useCallback(() => {
    if (editingId && editContent.trim() && onEdit) {
      onEdit(editingId, editContent.trim());
    }
    setEditingId(null);
  }, [editingId, editContent, onEdit]);

  const toggleThinking = useCallback((id: string) => {
    setExpandedThinking((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  const toggleHeatmap = useCallback((id: string) => {
    setHeatmapVisible((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  const isLastAssistant = (i: number) =>
    messages[i].role === 'assistant' && !messages.slice(i + 1).some((m) => m.role === 'assistant');

  if (messages.length === 0 && !streamingContent) {
    return (
      <Container ref={containerRef} className={className}>
        <EmptyState>
          <div style={{ position: 'relative', width: 80, height: 80 }}>
            <Circle $size={80} $opacity={0.15} />
            <Circle $size={56} $opacity={0.1} style={{ top: 12, left: 12 }} />
            <Circle $size={32} $opacity={0.08} style={{ top: 24, left: 24 }} />
          </div>
          Awaiting Input
        </EmptyState>
      </Container>
    );
  }

  return (
    <Container ref={containerRef} className={className}>
      {messages.map((msg, i) => (
        <MessageCard key={msg.id} $role={msg.role}>
          {/* Header */}
          <MsgHeader>
            {msg.role === 'assistant' ? (
              <>
                <Dot $color="#FFD700" />
                <RoleLabel $role="assistant">Skulk</RoleLabel>
                <Timestamp>{formatTime(msg.timestamp)}</Timestamp>
                {msg.ttftMs != null && <StatLabel>TTFT <span>{Math.round(msg.ttftMs)}ms</span></StatLabel>}
                {msg.tps != null && <StatLabel>TPS <span>{msg.tps.toFixed(1)}</span></StatLabel>}
              </>
            ) : (
              <>
                <Timestamp>{formatTime(msg.timestamp)}</Timestamp>
                <Spacer />
                <RoleLabel $role="user">Query</RoleLabel>
                <Dot $color="#999" />
              </>
            )}
          </MsgHeader>

          {/* Edit mode */}
          {editingId === msg.id ? (
            <div>
              <EditArea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); saveEdit(); } if (e.key === 'Escape') setEditingId(null); }}
                autoFocus
              />
              <BtnRow>
                <Button variant="primary" size="sm" onClick={saveEdit}>Save</Button>
                <Button variant="outline" size="sm" onClick={() => setEditingId(null)}>Cancel</Button>
              </BtnRow>
            </div>
          ) : deleteConfirmId === msg.id ? (
            <ConfirmBox>
              Delete this message?
              <BtnRow>
                <Button variant="danger" size="sm" onClick={() => { onDelete?.(msg.id); setDeleteConfirmId(null); }}>Delete</Button>
                <Button variant="outline" size="sm" onClick={() => setDeleteConfirmId(null)}>Cancel</Button>
              </BtnRow>
            </ConfirmBox>
          ) : (
            <>
              {/* Attachments (user messages) */}
              {msg.attachments && msg.attachments.length > 0 && (
                <AttachmentRow>
                  {msg.attachments.map((att) =>
                    att.preview ? (
                      <AttachThumb key={att.id} src={att.preview} alt={att.name} onClick={() => setLightboxSrc(att.preview!)} />
                    ) : (
                      <AttachFile key={att.id}>{getFileIcon(att.type, att.name)} {att.name}</AttachFile>
                    ),
                  )}
                </AttachmentRow>
              )}

              {/* Thinking block */}
              {msg.thinkingContent && (
                <ThinkingBlock $open={expandedThinking.has(msg.id)}>
                  <ThinkingHeader onClick={() => toggleThinking(msg.id)}>
                    <ThinkingChevron $open={expandedThinking.has(msg.id)}>▶</ThinkingChevron>
                    Thinking
                  </ThinkingHeader>
                  {expandedThinking.has(msg.id) && (
                    <ThinkingContent>{msg.thinkingContent}</ThinkingContent>
                  )}
                </ThinkingBlock>
              )}

              {/* Prefill progress */}
              {isLoading && isLastAssistant(i) && !msg.content && prefillProgress && (
                <PrefillProgressBar progress={prefillProgress} />
              )}

              {/* Generated images */}
              {msg.generatedImages && msg.generatedImages.length > 0 && (
                <ImageGrid>
                  {msg.generatedImages.map((src, j) => (
                    <GenImage key={j} src={src} alt={`Generated ${j + 1}`} onClick={() => setLightboxSrc(src)} />
                  ))}
                </ImageGrid>
              )}

              {/* Content */}
              {msg.role === 'assistant' ? (
                <>
                  {msg.content && <MarkdownContent content={msg.content} />}
                  {isLoading && isLastAssistant(i) && <Cursor />}
                </>
              ) : (
                <UserContent>{msg.content}</UserContent>
              )}

              {/* Token heatmap */}
              {heatmapVisible.has(msg.id) && msg.tokens && (
                <TokenHeatmap
                  tokens={msg.tokens}
                  isGenerating={isLoading}
                  onRegenerateFrom={onRegenerateFromToken}
                />
              )}

              {/* Action buttons */}
              <Actions>
                <Button variant="ghost" size="sm" onClick={() => copyMessage(msg.id, msg.content)}>
                  {copiedId === msg.id ? '✓' : 'Copy'}
                </Button>
                {msg.role === 'assistant' && msg.tokens && (
                  <ActiveGhostBtn
                    variant="ghost"
                    size="sm"
                    $active={heatmapVisible.has(msg.id)}
                    onClick={() => toggleHeatmap(msg.id)}
                  >
                    Heatmap
                  </ActiveGhostBtn>
                )}
                {msg.role === 'user' && onEdit && (
                  <Button variant="ghost" size="sm" onClick={() => { setEditingId(msg.id); setEditContent(msg.content); }}>
                    Edit
                  </Button>
                )}
                {msg.role === 'assistant' && isLastAssistant(i) && !isLoading && onRegenerate && (
                  <Button variant="ghost" size="sm" onClick={onRegenerate}>Regenerate</Button>
                )}
                {onDelete && (
                  <Button variant="danger" size="sm" onClick={() => setDeleteConfirmId(msg.id)}>Delete</Button>
                )}
              </Actions>
            </>
          )}
        </MessageCard>
      ))}

      {/* Streaming response (not yet a full message) */}
      {streamingContent && (
        <MessageCard $role="assistant">
          <MsgHeader>
            <Dot $color="#FFD700" />
            <RoleLabel $role="assistant">Skulk</RoleLabel>
          </MsgHeader>
          <MarkdownContent content={streamingContent} />
          <Cursor />
        </MessageCard>
      )}

      {showScrollBtn && (
        <ScrollBtn onClick={scrollToBottom} aria-label="Scroll to bottom">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </ScrollBtn>
      )}

      <ImageLightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />
    </Container>
  );
}
