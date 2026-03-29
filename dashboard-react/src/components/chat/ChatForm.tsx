import { useCallback, useEffect, useRef, useState } from 'react';
import styled, { css, keyframes } from 'styled-components';
import type { ChatUploadedFile } from '../../types/chat';
import { ChatAttachments } from './ChatAttachments';
import { Button } from '../common/Button';

export interface ChatFormProps {
  onSend: (message: string, files: ChatUploadedFile[]) => void;
  onCancel?: () => void;
  isLoading?: boolean;
  placeholder?: string;
  autoFocus?: boolean;
  /** Current model label shown in header. */
  modelLabel?: string;
  onOpenModelPicker?: () => void;
  /** Optional inline model selector element (replaces modelLabel click). */
  modelSelector?: React.ReactNode;
  /** TTFT in ms. */
  ttftMs?: number | null;
  /** Tokens per second. */
  tps?: number | null;
  /** Context window size in tokens (0 = unknown). */
  contextLength?: number;
  /** Enable thinking toggle. */
  showThinkingToggle?: boolean;
  thinkingEnabled?: boolean;
  onToggleThinking?: () => void;
  className?: string;
}

/* ---- styles ---- */

const Form = styled.form<{ $dragOver: boolean }>`
  position: relative;
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg};
  overflow: hidden;
  transition: border-color 0.15s;

  &:focus-within {
    border-color: rgba(255, 215, 0, 0.3);
  }

  ${({ $dragOver }) =>
    $dragOver &&
    css`
      border-color: #FFD700;
      box-shadow: 0 0 12px rgba(255, 215, 0, 0.2);
    `}
`;

const AccentLine = styled.div`
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.3), transparent);
`;

const HeaderRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const ModelBtn = styled.button`
  all: unset;
  cursor: pointer;
  color: #FFD700;
  font: inherit;
  transition: opacity 0.15s;
  &:hover { opacity: 0.8; }
`;

const ThinkingBtn = styled.button<{ $active: boolean }>`
  all: unset;
  cursor: pointer;
  font: inherit;
  padding: 2px 8px;
  border-radius: ${({ theme }) => theme.radii.sm};
  border: 1px solid;
  transition: all 0.15s;

  ${({ $active }) =>
    $active
      ? css`border-color: #FFD700; color: #FFD700; background: rgba(255,215,0,0.1);`
      : css`border-color: rgba(80,80,80,0.4); color: rgba(179,179,179,0.6); &:hover { border-color: rgba(255,215,0,0.3); color: #FFD700; }`}
`;

const Stat = styled.span`
  color: ${({ theme }) => theme.colors.textMuted};
  font-variant-numeric: tabular-nums;
`;

const StatValue = styled.span`
  color: rgba(255, 215, 0, 0.7);
`;

const Spacer = styled.span`
  flex: 1;
`;

const InputRow = styled.div`
  display: flex;
  align-items: flex-end;
  gap: 8px;
  padding: 8px 12px;
`;

const AttachBtn = styled(Button)`
  flex-shrink: 0;
`;

const Prompt = styled.span`
  color: #FFD700;
  font-size: ${({ theme }) => theme.fontSizes.lg};
  font-family: ${({ theme }) => theme.fonts.body};
  flex-shrink: 0;
  line-height: 28px;
`;

const TextArea = styled.textarea`
  all: unset;
  flex: 1;
  font-size: ${({ theme }) => theme.fontSizes.md};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.text};
  min-height: 28px;
  max-height: 150px;
  resize: none;
  line-height: 1.5;

  &::placeholder { color: ${({ theme }) => theme.colors.textMuted}; }
`;

const SendBtn = styled(Button)`
  flex-shrink: 0;
`;

const DragOverlay = styled.div`
  position: absolute;
  inset: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.8);
  border: 2px dashed #FFD700;
  border-radius: ${({ theme }) => theme.radii.lg};
  font-size: ${({ theme }) => theme.fontSizes.md};
  font-family: ${({ theme }) => theme.fonts.body};
  color: #FFD700;
`;

const HelperText = styled.div`
  padding: 4px 12px 8px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
  text-align: center;
`;

/* ---- component ---- */

export function ChatForm({
  onSend,
  onCancel,
  isLoading = false,
  placeholder = 'Type a message…',
  autoFocus = true,
  modelLabel,
  onOpenModelPicker,
  modelSelector,
  ttftMs,
  tps,
  contextLength = 0,
  showThinkingToggle = false,
  thinkingEnabled = false,
  onToggleThinking,
  className,
}: ChatFormProps) {
  const [message, setMessage] = useState('');
  const [files, setFiles] = useState<ChatUploadedFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const canSend = message.trim().length > 0 || files.length > 0;

  // Auto-resize textarea
  const resize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 150)}px`;
  }, []);

  // Refocus after loading completes
  const prevLoading = useRef(isLoading);
  useEffect(() => {
    if (prevLoading.current && !isLoading) {
      setTimeout(() => textareaRef.current?.focus(), 10);
    }
    prevLoading.current = isLoading;
  }, [isLoading]);

  const handleSubmit = useCallback(
    (e?: React.FormEvent) => {
      e?.preventDefault();
      if (isLoading || !canSend) return;
      onSend(message.trim(), files);
      setMessage('');
      setFiles([]);
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    },
    [isLoading, canSend, message, files, onSend],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey && !(e.nativeEvent as any).isComposing) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  // Drag and drop
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => setIsDragOver(false), []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const dropped = Array.from(e.dataTransfer.files);
    addFiles(dropped);
  }, []);

  const addFiles = useCallback((fileList: File[]) => {
    const newFiles: ChatUploadedFile[] = fileList.map((f) => ({
      id: `file-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      name: f.name,
      type: f.type,
      size: f.size,
      preview: f.type.startsWith('image/') ? URL.createObjectURL(f) : undefined,
    }));
    setFiles((prev) => [...prev, ...newFiles]);
  }, []);

  const removeFile = useCallback((id: string) => {
    setFiles((prev) => {
      const file = prev.find((f) => f.id === id);
      if (file?.preview) URL.revokeObjectURL(file.preview);
      return prev.filter((f) => f.id !== id);
    });
  }, []);

  const showHeader = modelLabel || modelSelector || showThinkingToggle || ttftMs != null || tps != null;

  return (
    <Form
      className={className}
      $dragOver={isDragOver}
      onSubmit={handleSubmit}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <AccentLine />

      {isDragOver && <DragOverlay>Drop files here</DragOverlay>}

      {/* Header: model + thinking + stats */}
      {showHeader && (
        <HeaderRow>
          {(modelLabel || modelSelector) && (
            <>
              <span>Model:</span>
              {modelSelector ?? <ModelBtn onClick={onOpenModelPicker}>{modelLabel}</ModelBtn>}
            </>
          )}
          {contextLength > 0 && (
            <Stat>{(contextLength / 1024).toFixed(0)}K ctx</Stat>
          )}
          {showThinkingToggle && (
            <ThinkingBtn $active={thinkingEnabled} onClick={onToggleThinking}>
              Thinking
            </ThinkingBtn>
          )}
          <Spacer />
          {ttftMs != null && (
            <Stat>TTFT <StatValue>{ttftMs.toFixed(1)}ms</StatValue></Stat>
          )}
          {tps != null && (
            <Stat>TPS <StatValue>{tps.toFixed(1)} tok/s</StatValue> ({(1000 / tps).toFixed(1)} ms/tok)</Stat>
          )}
        </HeaderRow>
      )}

      {/* Attachments */}
      {files.length > 0 && (
        <div style={{ padding: '0 12px' }}>
          <ChatAttachments files={files} onRemove={removeFile} />
        </div>
      )}

      {/* Input row */}
      <InputRow>
        <AttachBtn
          variant="ghost"
          size="sm"
          icon
          type="button"
          onClick={() => fileInputRef.current?.click()}
          aria-label="Attach file"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
          </svg>
        </AttachBtn>
        <Prompt>▶</Prompt>
        <TextArea
          ref={textareaRef}
          value={message}
          onChange={(e) => {
            setMessage(e.target.value);
            resize();
          }}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={1}
          autoFocus={autoFocus}
        />
        {isLoading ? (
          <SendBtn variant="danger" size="sm" icon type="button" onClick={onCancel} aria-label="Cancel generation">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="6" width="12" height="12" rx="2" />
            </svg>
          </SendBtn>
        ) : (
          <SendBtn variant="primary" size="sm" icon type="submit" disabled={!canSend} aria-label="Send message">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </SendBtn>
        )}
      </InputRow>

      <AccentLine />
      <HelperText>Enter to send · Shift+Enter for new line · Drag & drop files</HelperText>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        style={{ display: 'none' }}
        onChange={(e) => {
          if (e.target.files) addFiles(Array.from(e.target.files));
          e.target.value = '';
        }}
      />
    </Form>
  );
}
