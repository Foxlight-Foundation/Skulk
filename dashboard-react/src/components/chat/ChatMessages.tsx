import { useCallback, useEffect, useRef, useState } from 'react';
import styled, { css, keyframes, useTheme } from 'styled-components';
import type { Theme } from '../../theme';
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
  /** Current streaming thinking text, or null if not streaming thinking. */
  streamingThinking?: string | null;
  isLoading?: boolean;
  prefillProgress?: PrefillProgress | null;
  onDelete?: (messageId: string) => void;
  onEdit?: (messageId: string, content: string) => void;
  onRegenerate?: () => void;
  onRegenerateFromToken?: (tokenIndex: number) => void;
  /** Externally controlled expanded thinking message IDs */
  expandedThinkingIds?: Set<string>;
  onToggleThinking?: (messageId: string) => void;
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
  color: ${({ theme }) => theme.colors.textSecondary};
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.md};
  letter-spacing: 1px;
`;

const Circle = styled.div<{ $size: number; $opacity: number }>`
  position: absolute;
  width: ${({ $size }) => $size}px;
  height: ${({ $size }) => $size}px;
  border-radius: 50%;
  border: 1.5px solid ${({ theme }) => theme.colors.gold};
  box-shadow: 0 0 8px ${({ theme }) => theme.colors.gold};
  opacity: ${({ $opacity }) => $opacity};
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
          border-left: 2px solid ${({ theme }) => theme.colors.goldDim};
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
  color: ${({ $role, theme }) => ($role === 'assistant' ? theme.colors.gold : theme.colors.textSecondary)};
  font-weight: 600;
`;

const Timestamp = styled.span`
  color: ${({ theme }) => theme.colors.textMuted};
`;

const StatLabel = styled.span`
  color: ${({ theme }) => theme.colors.textMuted};
  font-variant-numeric: tabular-nums;
  & > span { color: ${({ theme }) => theme.colors.goldDim}; }
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
  background: ${({ theme }) => theme.colors.gold};
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
      color: ${({ theme }) => theme.colors.gold};
      background: ${({ theme }) => theme.colors.goldBg};
    `}
`;

const EditArea = styled.textarea`
  all: unset;
  width: 100%;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.md};
  color: ${({ theme }) => theme.colors.text};
  background: ${({ theme }) => theme.colors.bg};
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
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
  border: 1px solid ${({ theme }) => theme.colors.errorBg};
  border-radius: ${({ theme }) => theme.radii.sm};
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.errorText};
`;


const ThinkingBlock = styled.div<{ $open: boolean }>`
  margin: 8px 0;
  border: 1px solid ${({ theme }) => theme.colors.goldBg};
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
  color: ${({ theme }) => theme.colors.goldDim};
  transition: background 0.15s;
  box-sizing: border-box;
  &:hover { background: ${({ theme }) => theme.colors.goldBg}; }
`;

const ThinkingChevron = styled.span<{ $open: boolean }>`
  transition: transform 0.15s;
  ${({ $open }) => $open && css`transform: rotate(90deg);`}
`;

const ThinkingContent = styled.div`
  padding: 8px 10px;
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  color: ${({ theme }) => theme.colors.textMuted};
  border-top: 1px solid ${({ theme }) => theme.colors.goldBg};

  & * {
    color: inherit;
  }
`;

const ShowHideBtn = styled.span`
  cursor: pointer;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
  text-transform: uppercase;
  letter-spacing: 0.5px;
  transition: color 0.15s;
  &:hover { color: ${({ theme }) => theme.colors.text}; }
`;

const StreamThinkingBody = styled.div`
  max-height: 400px;
  overflow-y: auto;
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
  &:hover { border-color: ${({ theme }) => theme.colors.goldDim}; }
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
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
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
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
  color: ${({ theme }) => theme.colors.gold};
  box-shadow: 0 4px 12px ${({ theme }) => theme.colors.shadow};
  transition: all 0.15s;

  &:hover {
    background: ${({ theme }) => theme.colors.goldBg};
    border-color: ${({ theme }) => theme.colors.goldDim};
  }
`;

/* ---- component ---- */

export function ChatMessages({
  messages,
  streamingContent,
  streamingThinking,
  isLoading = false,
  prefillProgress,
  onDelete,
  onEdit,
  onRegenerate,
  onRegenerateFromToken,
  expandedThinkingIds: externalExpanded,
  onToggleThinking: externalToggle,
  className,
}: ChatMessagesProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const pinnedToBottomRef = useRef(true);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState('');
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [internalExpanded, setInternalExpanded] = useState<Set<string>>(new Set());
  const expandedThinking = externalExpanded ?? internalExpanded;
  const [heatmapVisible, setHeatmapVisible] = useState<Set<string>>(new Set());
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);
  const [streamThinkingOpen, setStreamThinkingOpen] = useState(false);
  const lastCountRef = useRef(messages.length);
  const theme = useTheme() as Theme;

  const getScrollParent = useCallback(
    () => containerRef.current?.parentElement ?? null,
    [],
  );

  const updateScrollState = useCallback(() => {
    const parent = getScrollParent();
    if (!parent) return;
    const dist = parent.scrollHeight - parent.scrollTop - parent.clientHeight;
    const pinnedToBottom = dist <= 100;
    pinnedToBottomRef.current = pinnedToBottom;
    setShowScrollBtn(!pinnedToBottom);
  }, [getScrollParent]);

  const scrollToBottom = useCallback(
    (behavior: ScrollBehavior = 'smooth') => {
      const parent = getScrollParent();
      if (!parent) return;
      parent.scrollTo({ top: parent.scrollHeight, behavior });
      pinnedToBottomRef.current = true;
      setShowScrollBtn(false);
    },
    [getScrollParent],
  );

  // Reset streaming thinking toggle when new stream starts
  const prevStreamingThinking = useRef(streamingThinking);
  useEffect(() => {
    if (streamingThinking && !prevStreamingThinking.current) {
      setStreamThinkingOpen(false);
    }
    prevStreamingThinking.current = streamingThinking;
  }, [streamingThinking]);

  // Auto-scroll on new messages
  useEffect(() => {
    if (messages.length > lastCountRef.current) {
      scrollToBottom();
    }
    lastCountRef.current = messages.length;
  }, [messages.length, scrollToBottom]);

  // Keep the viewport pinned while a streamed assistant response grows.
  useEffect(() => {
    if (
      !pinnedToBottomRef.current
      || (streamingContent == null && !streamingThinking)
    ) {
      return;
    }

    const animationFrameId = requestAnimationFrame(() => {
      scrollToBottom('auto');
    });

    return () => cancelAnimationFrame(animationFrameId);
  }, [streamingContent, streamingThinking, scrollToBottom]);

  // Scroll button visibility
  useEffect(() => {
    const parent = getScrollParent();
    if (!parent) return;
    updateScrollState();
    parent.addEventListener('scroll', updateScrollState, { passive: true });
    return () => parent.removeEventListener('scroll', updateScrollState);
  }, [getScrollParent, updateScrollState]);

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
    if (externalToggle) {
      externalToggle(id);
    } else {
      setInternalExpanded((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id); else next.add(id);
        return next;
      });
    }
  }, [externalToggle]);

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
            <Circle $size={80} $opacity={0.35} />
            <Circle $size={56} $opacity={0.25} style={{ top: 12, left: 12 }} />
            <Circle $size={32} $opacity={0.18} style={{ top: 24, left: 24 }} />
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
                <Dot $color={theme.colors.gold} />
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
                <Dot $color={theme.colors.textSecondary} />
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
                    <Spacer />
                    <ShowHideBtn onClick={(e) => { e.stopPropagation(); toggleThinking(msg.id); }}>
                      {expandedThinking.has(msg.id) ? 'Hide' : 'Show'}
                    </ShowHideBtn>
                  </ThinkingHeader>
                  {expandedThinking.has(msg.id) && (
                    <ThinkingContent><MarkdownContent content={msg.thinkingContent} /></ThinkingContent>
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
      {(streamingContent != null || streamingThinking) && (
        <MessageCard $role="assistant">
          <MsgHeader>
            <Dot $color={theme.colors.gold} />
            <RoleLabel $role="assistant">Skulk</RoleLabel>
          </MsgHeader>
          {streamingThinking && (
            <ThinkingBlock $open={streamThinkingOpen}>
              <ThinkingHeader onClick={() => setStreamThinkingOpen((v) => !v)}>
                <ThinkingChevron $open={streamThinkingOpen}>▶</ThinkingChevron>
                Thinking...
                <Spacer />
                <ShowHideBtn onClick={(e) => { e.stopPropagation(); setStreamThinkingOpen((v) => !v); }}>
                  {streamThinkingOpen ? 'Hide' : 'Show'}
                </ShowHideBtn>
              </ThinkingHeader>
              {streamThinkingOpen && (
                <StreamThinkingBody>
                  <ThinkingContent><MarkdownContent content={streamingThinking} /></ThinkingContent>
                </StreamThinkingBody>
              )}
            </ThinkingBlock>
          )}
          {streamingContent ? (
            <>
              <MarkdownContent content={streamingContent} />
              <Cursor />
            </>
          ) : (
            <Cursor />
          )}
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
