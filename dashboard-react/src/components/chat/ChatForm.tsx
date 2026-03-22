/**
 * ChatForm
 *
 * Message input with:
 * - Auto-resizing textarea, IME composition guard
 * - Drag-and-drop + paste file upload
 * - Model selector button (opens ModelPickerModal)
 * - Thinking toggle, performance stats
 * - Edit mode banner for image editing
 * - Send / Generate / Edit / Cancel button
 * - ChatAttachments preview strip
 *
 * Ported from ChatForm.svelte.
 */
import React, {
  useState,
  useRef,
  useEffect,
  useCallback,
  useLayoutEffect,
} from 'react';
import styled from 'styled-components';
import { useTranslate } from '@tolgee/react';
import { ChatAttachments } from './ChatAttachments';
import { ImageParamsPanel } from './ImageParamsPanel';
import type { ChatUploadedFile } from '../../types/files';
import { processUploadedFiles, getAcceptString } from '../../types/files';
import { useChatStore } from '../../stores/chatStore';

// ─── Styled components ────────────────────────────────────────────────────────

const FormWrapper = styled.form<{ $isDragOver: boolean }>`
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 0;
`;

const CommandPanel = styled.div<{ $isDragOver: boolean }>`
  position: relative;
  overflow: hidden;
  border-radius: ${({ theme }) => theme.radius.md};
  transition: box-shadow 0.2s;
  background: oklch(0.14 0 0);
  border: 1px solid ${({ theme, $isDragOver }) =>
    $isDragOver ? theme.colors.yellow : theme.colors.border};
  box-shadow: ${({ $isDragOver }) =>
    $isDragOver ? '0 0 0 2px oklch(0.85 0.18 85 / 0.3)' : 'none'};
`;

const TopAccent = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(to right, transparent, oklch(0.85 0.18 85 / 0.5), transparent);
`;

const BottomAccent = styled.div`
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(to right, transparent, oklch(0.85 0.18 85 / 0.3), transparent);
`;

const DragOverlay = styled.div`
  position: absolute;
  inset: 0;
  background: oklch(0.16 0 0 / 0.8);
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const DragText = styled.div`
  font-size: 12px;
  font-family: ${({ theme }) => theme.fonts.mono};
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.yellow};
`;

const EditBanner = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 12px;
  background: oklch(0.85 0.18 85 / 0.08);
  border-bottom: 1px solid oklch(0.85 0.18 85 / 0.25);
`;

const EditBannerLabel = styled.span`
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.yellow};
`;

const CancelEditBtn = styled.button`
  padding: 3px 8px;
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  letter-spacing: 0.08em;
  text-transform: uppercase;
  background: oklch(0.25 0 0 / 0.3);
  color: ${({ theme }) => theme.colors.lightGray};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radius.sm};
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.fast};
  &:hover {
    background: oklch(0.25 0 0 / 0.5);
    color: ${({ theme }) => theme.colors.yellow};
  }
`;

const ModelRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px 12px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
`;

const ModelLabel = styled.span`
  font-size: 11px;
  color: ${({ theme }) => theme.colors.lightGray};
  text-transform: uppercase;
  letter-spacing: 0.08em;
  flex-shrink: 0;
`;

const ModelBtn = styled.button`
  flex: 1;
  max-width: 240px;
  background: oklch(0.25 0 0 / 0.5);
  border: 1px solid oklch(0.85 0.18 85 / 0.3);
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 4px 28px 4px 10px;
  font-size: 12px;
  font-family: ${({ theme }) => theme.fonts.mono};
  text-align: left;
  cursor: pointer;
  position: relative;
  transition: border-color ${({ theme }) => theme.transitions.fast};
  &:hover { border-color: oklch(0.85 0.18 85 / 0.5); }
  &:focus { outline: none; border-color: ${({ theme }) => theme.colors.yellow}; }
`;

const ModelBtnValue = styled.span`
  color: ${({ theme }) => theme.colors.yellow};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  display: block;
`;

const ModelBtnPlaceholder = styled.span`
  color: oklch(0.65 0 0 / 0.5);
`;

const ModelBtnChevron = styled.span`
  position: absolute;
  right: 8px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 9px;
  color: oklch(0.85 0.18 85 / 0.6);
  pointer-events: none;
`;

const ThinkBtn = styled.button<{ $active: boolean }>`
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border-radius: ${({ theme }) => theme.radius.sm};
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  letter-spacing: 0.08em;
  text-transform: uppercase;
  cursor: pointer;
  flex-shrink: 0;
  transition: all ${({ theme }) => theme.transitions.fast};
  border: 1px solid ${({ theme, $active }) =>
    $active ? 'oklch(0.85 0.18 85 / 0.4)' : theme.colors.border};
  background: ${({ $active }) =>
    $active ? 'oklch(0.85 0.18 85 / 0.12)' : 'oklch(0.25 0 0 / 0.3)'};
  color: ${({ theme, $active }) =>
    $active ? theme.colors.yellow : 'oklch(0.65 0 0 / 0.6)'};
  &:hover { color: ${({ theme }) => theme.colors.yellow}; }
`;

const PerfStats = styled.div`
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  flex-shrink: 0;
`;

const StatLabel = styled.span`color: oklch(1 0 0 / 0.7);`;
const StatValue = styled.span`color: ${({ theme }) => theme.colors.yellow};`;
const StatUnit = styled.span`color: oklch(1 0 0 / 0.6);`;

const AttachmentsWrapper = styled.div`
  padding: 10px 12px 0;
`;

const InputRow = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 10px 12px;
`;

const AttachBtn = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: ${({ theme }) => theme.radius.sm};
  background: none;
  border: none;
  color: ${({ theme }) => theme.colors.lightGray};
  cursor: pointer;
  flex-shrink: 0;
  transition: color ${({ theme }) => theme.transitions.fast};
  &:hover { color: ${({ theme }) => theme.colors.yellow}; }
  &:disabled { opacity: 0.4; cursor: not-allowed; }
`;

const Prompt = styled.span`
  color: ${({ theme }) => theme.colors.yellow};
  font-size: 14px;
  font-weight: 700;
  flex-shrink: 0;
  line-height: 28px;
`;

const Textarea = styled.textarea`
  flex: 1;
  resize: none;
  background: transparent;
  border: none;
  outline: none;
  color: ${({ theme }) => theme.colors.foreground};
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 13px;
  line-height: 1.75;
  min-height: 28px;
  max-height: 150px;
  &::placeholder {
    color: oklch(0.65 0 0 / 0.5);
    letter-spacing: 0.12em;
  }
`;

const SendBtn = styled.button<{ $canSend: boolean }>`
  padding: 5px 12px;
  border-radius: ${({ theme }) => theme.radius.sm};
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-weight: 500;
  cursor: ${({ $canSend }) => ($canSend ? 'pointer' : 'not-allowed')};
  white-space: nowrap;
  border: none;
  transition: all ${({ theme }) => theme.transitions.fast};
  background: ${({ theme, $canSend }) =>
    $canSend ? theme.colors.yellow : 'oklch(0.25 0 0 / 0.5)'};
  color: ${({ theme, $canSend }) =>
    $canSend ? theme.colors.black : theme.colors.lightGray};
  &:hover {
    box-shadow: ${({ $canSend }) =>
      $canSend ? '0 0 20px oklch(0.85 0.18 85 / 0.3)' : 'none'};
  }
`;

const StopBtn = styled.button`
  padding: 5px 12px;
  border-radius: ${({ theme }) => theme.radius.sm};
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;
  border: none;
  background: oklch(0.25 0 0 / 0.7);
  color: ${({ theme }) => theme.colors.lightGray};
  transition: all ${({ theme }) => theme.transitions.fast};
  &:hover { background: oklch(0.3 0 0); color: ${({ theme }) => theme.colors.foreground}; }
`;

const HelperText = styled.p`
  margin-top: 8px;
  text-align: center;
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.lightGray};
`;

const Kbd = styled.kbd`
  padding: 1px 5px;
  border-radius: ${({ theme }) => theme.radius.sm};
  background: oklch(0.25 0 0 / 0.3);
  color: ${({ theme }) => theme.colors.lightGray};
  border: 1px solid ${({ theme }) => theme.colors.border};
  font-family: ${({ theme }) => theme.fonts.mono};
`;

// ─── Component ─────────────────────────────────────────────────────────────────

export interface ChatFormProps {
  showModelSelector?: boolean;
  modelTasks?: Record<string, string[]>;
  modelCapabilities?: Record<string, string[]>;
  showHelperText?: boolean;
  autofocus?: boolean;
  onAutoSend: (content: string, files?: ChatUploadedFile[]) => void;
  onOpenModelPicker?: () => void;
}

const ACCEPT = getAcceptString(['image', 'text', 'pdf']);

export const ChatForm: React.FC<ChatFormProps> = ({
  showModelSelector = false,
  modelTasks = {},
  modelCapabilities = {},
  showHelperText = false,
  autofocus = true,
  onAutoSend,
  onOpenModelPicker,
}) => {
  const { t } = useTranslate();
  const [message, setMessage] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState<ChatUploadedFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isComposingRef = useRef(false);

  const selectedModelId = useChatStore((s) => s.selectedModelId);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const stopStreaming = useChatStore((s) => s.stopStreaming);
  const editingImageBase64 = useChatStore((s) => s.editingImageBase64);
  const setEditingImage = useChatStore((s) => s.setEditingImage);
  const getActiveConversation = useChatStore((s) => s.getActiveConversation);
  const setConversationThinking = useChatStore((s) => s.setConversationThinking);
  const getLastTtft = useChatStore((s) => s.getLastTtft);
  const getLastTps = useChatStore((s) => s.getLastTps);

  const activeConv = getActiveConversation();
  const thinkingEnabled = activeConv?.thinkingEnabled ?? false;
  const ttft = getLastTtft();
  const currentTps = getLastTps();

  const currentModelLabel = selectedModelId
    ? (selectedModelId.split('/').pop() ?? selectedModelId)
    : '';

  // Model capability checks
  const supportsTextToImage = (id: string) => (modelTasks[id] ?? []).includes('TextToImage');
  const supportsImageToImage = (id: string) => (modelTasks[id] ?? []).includes('ImageToImage');
  const supportsOnlyImageEditing = (id: string) =>
    (modelTasks[id] ?? []).includes('ImageToImage') &&
    !(modelTasks[id] ?? []).includes('TextToImage');
  const isImageModel = selectedModelId
    ? supportsTextToImage(selectedModelId) || supportsImageToImage(selectedModelId)
    : false;
  const supportsThinking = selectedModelId
    ? (modelCapabilities[selectedModelId] ?? []).includes('thinking_toggle') &&
      (modelCapabilities[selectedModelId] ?? []).includes('text')
    : false;
  const isEditMode = editingImageBase64 !== null;
  const isEditOnlyWithoutImage =
    selectedModelId !== null &&
    supportsOnlyImageEditing(selectedModelId ?? '') &&
    !isEditMode &&
    uploadedFiles.length === 0;
  const shouldShowEditMode =
    isEditMode ||
    (selectedModelId !== null && supportsImageToImage(selectedModelId) && uploadedFiles.length > 0);

  // Autofocus
  useEffect(() => {
    if (autofocus) {
      setTimeout(() => textareaRef.current?.focus(), 10);
    }
  }, [autofocus]);

  // Refocus after streaming ends
  const prevStreamingRef = useRef(isStreaming);
  useEffect(() => {
    if (prevStreamingRef.current && !isStreaming) {
      setTimeout(() => textareaRef.current?.focus(), 50);
    }
    prevStreamingRef.current = isStreaming;
  }, [isStreaming]);

  // Textarea auto-resize
  const resizeTextarea = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 150)}px`;
  }, []);

  const handleFiles = useCallback(async (files: File[]) => {
    if (files.length === 0) return;
    const processed = await processUploadedFiles(files);
    setUploadedFiles((prev) => [...prev, ...processed]);
  }, []);

  const handleFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0) {
        void handleFiles(Array.from(e.target.files));
        e.target.value = '';
      }
    },
    [handleFiles],
  );

  const handlePaste = useCallback(
    (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
      if (!e.clipboardData) return;
      const files = Array.from(e.clipboardData.items)
        .filter((item) => item.kind === 'file')
        .map((item) => item.getAsFile())
        .filter((f): f is File => f !== null);
      if (files.length > 0) {
        e.preventDefault();
        void handleFiles(files);
        return;
      }
      const text = e.clipboardData.getData('text/plain');
      if (text.length > 2500) {
        e.preventDefault();
        void handleFiles([new File([text], 'pasted-text.txt', { type: 'text/plain' })]);
      }
    },
    [handleFiles],
  );

  const handleSubmit = useCallback(() => {
    if ((!message.trim() && uploadedFiles.length === 0) || isStreaming) return;
    if (isEditOnlyWithoutImage) return;
    const content = message.trim();
    const files = [...uploadedFiles];
    setMessage('');
    setUploadedFiles([]);
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    onAutoSend(content, files);
    setTimeout(() => textareaRef.current?.focus(), 10);
  }, [message, uploadedFiles, isStreaming, isEditOnlyWithoutImage, onAutoSend]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.nativeEvent.isComposing || e.nativeEvent.keyCode === 229) return;
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  const canSend = message.trim().length > 0 || uploadedFiles.length > 0;

  const placeholder = isEditOnlyWithoutImage
    ? 'Attach an image to edit...'
    : isEditMode
    ? 'Describe how to edit this image...'
    : isImageModel
    ? 'Describe the image you want to generate...'
    : t('chat.placeholder');

  let submitLabel: React.ReactNode = 'SEND';
  if (shouldShowEditMode) {
    submitLabel = (
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
        </svg>
        EDIT
      </span>
    );
  } else if (isImageModel) {
    submitLabel = (
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <circle cx="8.5" cy="8.5" r="1.5" />
          <polyline points="21 15 16 10 5 21" />
        </svg>
        GENERATE
      </span>
    );
  }

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPT}
        multiple
        style={{ display: 'none' }}
        onChange={handleFileInputChange}
      />
      <FormWrapper
        $isDragOver={isDragOver}
        onSubmit={(e) => { e.preventDefault(); handleSubmit(); }}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={(e) => { e.preventDefault(); setIsDragOver(false); }}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragOver(false);
          if (e.dataTransfer?.files) void handleFiles(Array.from(e.dataTransfer.files));
        }}
      >
        <CommandPanel $isDragOver={isDragOver}>
          <TopAccent />

          {/* Drag overlay */}
          {isDragOver && (
            <DragOverlay>
              <DragText>Drop Files Here</DragText>
            </DragOverlay>
          )}

          {/* Edit mode banner */}
          {isEditMode && editingImageBase64 && (
            <EditBanner>
              <img
                src={editingImageBase64}
                alt="Source for editing"
                style={{ width: 40, height: 40, objectFit: 'cover', borderRadius: 4, border: '1px solid oklch(0.85 0.18 85 / 0.3)' }}
              />
              <div style={{ flex: 1 }}>
                <EditBannerLabel>Editing Image</EditBannerLabel>
              </div>
              <CancelEditBtn type="button" onClick={() => setEditingImage(null)}>
                Cancel
              </CancelEditBtn>
            </EditBanner>
          )}

          {/* Model selector row */}
          {showModelSelector && (
            <ModelRow>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1 }}>
                <ModelLabel>Model:</ModelLabel>
                <div style={{ position: 'relative', flex: 1, maxWidth: 240 }}>
                  <ModelBtn type="button" onClick={() => onOpenModelPicker?.()}>
                    {currentModelLabel
                      ? <ModelBtnValue>{currentModelLabel}</ModelBtnValue>
                      : <ModelBtnPlaceholder>— Select Model —</ModelBtnPlaceholder>
                    }
                    <ModelBtnChevron>▼</ModelBtnChevron>
                  </ModelBtn>
                </div>
              </div>

              {/* Thinking toggle */}
              {supportsThinking && (
                <ThinkBtn
                  type="button"
                  $active={thinkingEnabled}
                  onClick={() => setConversationThinking(!thinkingEnabled)}
                  title={thinkingEnabled ? 'Thinking enabled — click to disable' : 'Thinking disabled — click to enable'}
                >
                  <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7zM9 20h6M10 22h4" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <span>{thinkingEnabled ? 'THINK' : 'NO THINK'}</span>
                </ThinkBtn>
              )}

              {/* Performance stats */}
              {(ttft !== null || currentTps !== null) && (
                <PerfStats>
                  {ttft !== null && (
                    <span>
                      <StatLabel>TTFT </StatLabel>
                      <StatValue>{ttft.toFixed(1)}</StatValue>
                      <StatUnit>ms</StatUnit>
                    </span>
                  )}
                  {currentTps !== null && (
                    <span>
                      <StatLabel>TPS </StatLabel>
                      <StatValue>{currentTps.toFixed(1)}</StatValue>
                      <StatUnit> tok/s</StatUnit>
                      <StatUnit style={{ opacity: 0.5 }}> ({(1000 / currentTps).toFixed(1)} ms/tok)</StatUnit>
                    </span>
                  )}
                </PerfStats>
              )}
            </ModelRow>
          )}

          {/* Image params panel (shown for image models) */}
          {showModelSelector && (isImageModel || isEditMode) && (
            <ImageParamsPanel isEditMode={isEditMode} />
          )}

          {/* Attachments preview */}
          {uploadedFiles.length > 0 && (
            <AttachmentsWrapper>
              <ChatAttachments
                files={uploadedFiles}
                onRemove={(id) => setUploadedFiles((prev) => prev.filter((f) => f.id !== id))}
              />
            </AttachmentsWrapper>
          )}

          {/* Input area */}
          <InputRow>
            <AttachBtn
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isStreaming}
              title="Attach file"
            >
              <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
              </svg>
            </AttachBtn>

            <Prompt>▶</Prompt>

            <Textarea
              ref={textareaRef}
              value={message}
              placeholder={placeholder}
              rows={1}
              onChange={(e) => { setMessage(e.target.value); resizeTextarea(); }}
              onKeyDown={handleKeyDown}
              onPaste={handlePaste}
              onCompositionStart={() => { isComposingRef.current = true; }}
              onCompositionEnd={() => { isComposingRef.current = false; }}
            />

            {isStreaming ? (
              <StopBtn type="button" onClick={stopStreaming} aria-label="Stop generation">
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                  <svg width="12" height="12" fill="currentColor" viewBox="0 0 24 24">
                    <rect x="6" y="6" width="12" height="12" rx="1" />
                  </svg>
                  <span>Cancel</span>
                </span>
              </StopBtn>
            ) : (
              <SendBtn
                type="submit"
                $canSend={canSend && !isEditOnlyWithoutImage}
                disabled={!canSend || isEditOnlyWithoutImage}
                aria-label={shouldShowEditMode ? 'Edit image' : isImageModel ? 'Generate image' : 'Send message'}
              >
                {submitLabel}
              </SendBtn>
            )}
          </InputRow>

          <BottomAccent />
        </CommandPanel>

        {showHelperText && (
          <HelperText>
            <Kbd>Enter</Kbd> to send
            {' '}<span style={{ opacity: 0.4, margin: '0 6px' }}>|</span>
            <Kbd>Shift+Enter</Kbd> new line
            {' '}<span style={{ opacity: 0.4, margin: '0 6px' }}>|</span>
            <span>Drag & drop or paste files</span>
          </HelperText>
        )}
      </FormWrapper>
    </>
  );
};
