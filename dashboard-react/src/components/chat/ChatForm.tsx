import { useCallback, useEffect, useRef, useState } from 'react';
import { MOBILE_BREAKPOINT_PX } from '../../hooks/useMediaQuery';
import styled, { css } from 'styled-components';
import { FiMic, FiPaperclip, FiSend, FiSquare, FiVolume2, FiVolumeX, FiX } from 'react-icons/fi';
import type { ChatSpeechModelOption, ChatUploadedFile } from '../../types/chat';
import { ChatAttachments } from './ChatAttachments';
import { Button } from '../common/Button';
import { useSkulkTranslation } from '../../i18n/tolgee';

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
  /** Whether the selected model accepts image attachments. */
  supportsImageAttachments?: boolean;
  /** Whether the text chat send action is currently available. */
  canSendMessages?: boolean;
  /** Mounted STT models available for browser microphone transcription. */
  transcriptionModels?: ChatSpeechModelOption[];
  /** Mounted TTS models available for browser playback. */
  speechModels?: ChatSpeechModelOption[];
  /** Selected STT model id, persisted by the chat slice. */
  selectedTranscriptionModelId?: string | null;
  /** Selected TTS model id, persisted by the chat slice. */
  selectedSpeechModelId?: string | null;
  /** Optional model-specific TTS voice or preset. */
  selectedVoice?: string | null;
  /** Whether final assistant messages should be spoken automatically. */
  autoSpeakAssistant?: boolean;
  /** Whether a dashboard-managed audio playback is active. */
  isSpeaking?: boolean;
  /** Speech-related API error surfaced from the page container. */
  voiceError?: string | null;
  /** Persist the selected STT model. */
  onSelectTranscriptionModel?: (modelId: string | null) => void;
  /** Persist the selected TTS model. */
  onSelectSpeechModel?: (modelId: string | null) => void;
  /** Persist the optional model-specific voice or preset. */
  onSelectedVoiceChange?: (voice: string | null) => void;
  /** Toggle automatic TTS playback for final assistant messages. */
  onAutoSpeakAssistantChange?: (enabled: boolean) => void;
  /** Transcribe a browser-recorded audio blob and return transcript text. */
  onTranscribeAudio?: (audio: Blob) => Promise<string>;
  /** Speak the current draft text with the selected TTS model. */
  onSpeakText?: (text: string) => void | Promise<void>;
  /** Stop dashboard-managed audio playback. */
  onStopSpeaking?: () => void;
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
    border-color: ${({ theme }) => theme.colors.goldDim};
  }

  ${({ $dragOver }) =>
    $dragOver &&
    css`
      border-color: ${({ theme }) => theme.colors.gold};
      box-shadow: 0 0 12px ${({ theme }) => theme.colors.goldDim};
    `}
`;

const AccentLine = styled.div`
  height: 1px;
  background: linear-gradient(90deg, transparent, ${({ theme }) => theme.colors.goldDim}, transparent);
`;

const HeaderRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};

  @media (max-width: ${MOBILE_BREAKPOINT_PX}px) {
    flex-wrap: wrap;
    row-gap: 4px;
  }
`;

const VoiceRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  padding: 0 12px 8px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const VoiceGroup = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
`;

const VoiceLabel = styled.span`
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: ${({ theme }) => theme.fontSizes.xs};
`;

const VoiceSelect = styled.select`
  appearance: none;
  min-width: 96px;
  max-width: 180px;
  height: 28px;
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.sm};
  background: transparent;
  color: ${({ theme }) => theme.colors.textSecondary};
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  outline: none;
  padding: 0 8px;

  &:focus {
    border-color: ${({ theme }) => theme.colors.goldDim};
  }

  &:disabled {
    color: ${({ theme }) => theme.colors.textMuted};
  }

  option {
    background: ${({ theme }) => theme.colors.surface};
    color: ${({ theme }) => theme.colors.text};
  }
`;

const VoiceInput = styled.input`
  all: unset;
  box-sizing: border-box;
  width: 92px;
  height: 28px;
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.sm};
  color: ${({ theme }) => theme.colors.textSecondary};
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  padding: 0 8px;

  &:focus {
    border-color: ${({ theme }) => theme.colors.goldDim};
  }

  &::placeholder {
    color: ${({ theme }) => theme.colors.textMuted};
  }
`;

const VoiceToggle = styled.button<{ $active: boolean }>`
  all: unset;
  cursor: pointer;
  font: inherit;
  height: 28px;
  padding: 0 8px;
  border-radius: ${({ theme }) => theme.radii.sm};
  border: 1px solid;
  transition: all 0.15s;

  ${({ $active }) =>
    $active
      ? css`border-color: ${({ theme }) => theme.colors.gold}; color: ${({ theme }) => theme.colors.gold}; background: ${({ theme }) => theme.colors.goldBg};`
      : css`border-color: ${({ theme }) => theme.colors.border}; color: ${({ theme }) => theme.colors.textMuted}; &:hover { border-color: ${({ theme }) => theme.colors.goldDim}; color: ${({ theme }) => theme.colors.gold}; }`}

  &:disabled {
    opacity: 0.88;
    cursor: not-allowed;
  }

  &:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px ${({ theme }) => theme.colors.goldDim};
  }
`;

const VoiceIconBtn = styled(Button)<{ $active?: boolean }>`
  ${({ $active }) =>
    $active &&
    css`
      color: ${({ theme }) => theme.colors.gold};
      border-color: ${({ theme }) => theme.colors.goldDim};
      background: ${({ theme }) => theme.colors.goldBg};
    `}
`;

const VoiceStatus = styled.span<{ $error?: boolean }>`
  color: ${({ $error, theme }) => ($error ? theme.colors.error : theme.colors.goldDim)};
  font-variant-numeric: tabular-nums;
`;

/*
 * Groups the model label + name so that, at phone width, the model gets its
 * own full line and the stats (ctx / thinking / TTFT / TPS) flow onto the
 * next line, instead of the long model id wrapping mid-name in a narrow
 * column. On desktop the wrapper is display: contents (layout unchanged).
 */
const ModelLine = styled.div`
  display: contents;

  @media (max-width: ${MOBILE_BREAKPOINT_PX}px) {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-basis: 100%;
    min-width: 0;
  }
`;

const ModelBtn = styled.button`
  all: unset;
  cursor: pointer;
  color: ${({ theme }) => theme.colors.gold};
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
      ? css`border-color: ${({ theme }) => theme.colors.gold}; color: ${({ theme }) => theme.colors.gold}; background: ${({ theme }) => theme.colors.goldBg};`
      : css`border-color: ${({ theme }) => theme.colors.border}; color: ${({ theme }) => theme.colors.textMuted}; &:hover { border-color: ${({ theme }) => theme.colors.goldDim}; color: ${({ theme }) => theme.colors.gold}; }`}
`;

const Stat = styled.span`
  color: ${({ theme }) => theme.colors.textMuted};
  font-variant-numeric: tabular-nums;
`;

const StatValue = styled.span`
  color: ${({ theme }) => theme.colors.goldDim};
`;

const Spacer = styled.span`
  flex: 1;
`;

const InputRow = styled.div`
  display: flex;
  align-items: flex-end;
  gap: 8px;
  padding: 8px 12px;

  @media (max-width: ${MOBILE_BREAKPOINT_PX}px) {
    gap: 6px;
  }
`;

const AttachBtn = styled(Button)`
  flex-shrink: 0;
`;

const Prompt = styled.span`
  color: ${({ theme }) => theme.colors.gold};
  font-size: ${({ theme }) => theme.fontSizes.lg};
  font-family: ${({ theme }) => theme.fonts.body};
  flex-shrink: 0;
  line-height: 28px;
`;

const TextArea = styled.textarea`
  all: unset;
  flex: 1;
  min-width: 0;
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
  background: ${({ theme }) => theme.colors.overlay};
  border: 2px dashed ${({ theme }) => theme.colors.gold};
  border-radius: ${({ theme }) => theme.radii.lg};
  font-size: ${({ theme }) => theme.fontSizes.md};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.gold};
`;

const HelperText = styled.div`
  padding: 4px 12px 8px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
  text-align: center;

  /* Keyboard hints (Enter / Shift+Enter / drag & drop) mean nothing on a
   * phone; they and their rule hide below the breakpoint. */
  @media (max-width: ${MOBILE_BREAKPOINT_PX}px) {
    display: none;
  }
`;

const BottomAccentLine = styled(AccentLine)`
  @media (max-width: ${MOBILE_BREAKPOINT_PX}px) {
    display: none;
  }
`;

function isLocalBrowserHostname(hostname: string): boolean {
  return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1';
}

/* ---- component ---- */

export function ChatForm({
  onSend,
  onCancel,
  isLoading = false,
  placeholder,
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
  supportsImageAttachments = false,
  canSendMessages = true,
  transcriptionModels = [],
  speechModels = [],
  selectedTranscriptionModelId = null,
  selectedSpeechModelId = null,
  selectedVoice = null,
  autoSpeakAssistant = false,
  isSpeaking = false,
  voiceError = null,
  onSelectTranscriptionModel,
  onSelectSpeechModel,
  onSelectedVoiceChange,
  onAutoSpeakAssistantChange,
  onTranscribeAudio,
  onSpeakText,
  onStopSpeaking,
  className,
}: ChatFormProps) {
  const { t } = useSkulkTranslation();
  const [message, setMessage] = useState('');
  const [files, setFiles] = useState<ChatUploadedFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [mediaError, setMediaError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const filesRef = useRef<ChatUploadedFile[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordingCancelledRef = useRef(false);
  const recordingStartedAtRef = useRef(0);
  const recordingTimerRef = useRef<number | null>(null);

  const canSend = canSendMessages && (message.trim().length > 0 || files.length > 0);
  const inputPlaceholder = placeholder ?? t('chat.form.placeholder', 'Type a message...');
  const selectedTranscriptionId = selectedTranscriptionModelId ?? transcriptionModels[0]?.modelId ?? null;
  const selectedSpeechId = selectedSpeechModelId ?? speechModels[0]?.modelId ?? null;
  const secureRecordingContext = typeof window === 'undefined'
    || window.isSecureContext
    || isLocalBrowserHostname(window.location.hostname);
  const browserRecordingAvailable = secureRecordingContext
    && typeof navigator !== 'undefined'
    && Boolean(navigator.mediaDevices?.getUserMedia)
    && typeof MediaRecorder !== 'undefined';
  const recordingUnavailableReason = !secureRecordingContext
    ? t('chat.form.voiceErrors.secureContextRequired', 'Microphone requires HTTPS or localhost.')
    : !browserRecordingAvailable
      ? t('chat.form.voiceErrors.unsupportedRecording', 'Browser audio recording is unavailable.')
      : null;
  const speechReady = Boolean(selectedSpeechId && onSpeakText);
  const transcriptionReady = Boolean(selectedTranscriptionId && onTranscribeAudio);
  const canRecordAudio = transcriptionReady && browserRecordingAvailable;
  const canSpeakDraft = speechReady && message.trim().length > 0 && !isRecording && !isTranscribing;
  const displayVoiceError = mediaError ?? voiceError;
  const showVoiceControls = transcriptionModels.length > 0 || speechModels.length > 0 || Boolean(displayVoiceError);

  // Auto-resize textarea
  const resize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 150)}px`;
  }, []);

  useEffect(() => {
    resize();
  }, [message, resize]);

  const stopRecordingTimer = useCallback(() => {
    if (recordingTimerRef.current !== null) {
      window.clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
  }, []);

  const cleanupRecordingResources = useCallback(() => {
    stopRecordingTimer();
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;
    mediaRecorderRef.current = null;
    setIsRecording(false);
    setRecordingSeconds(0);
  }, [stopRecordingTimer]);

  const appendTranscript = useCallback((transcript: string) => {
    const trimmed = transcript.trim();
    if (!trimmed) return;
    setMessage((prev) => {
      const current = prev.trimEnd();
      if (!current) return trimmed;
      return `${current}\n${trimmed}`;
    });
  }, []);

  const startRecording = useCallback(async () => {
    if (!transcriptionReady || isLoading || isTranscribing) return;
    if (!secureRecordingContext) {
      setMediaError(t('chat.form.voiceErrors.secureContextRequired', 'Microphone requires HTTPS or localhost.'));
      return;
    }
    if (
      typeof navigator === 'undefined'
      || !navigator.mediaDevices?.getUserMedia
      || typeof MediaRecorder === 'undefined'
    ) {
      setMediaError(t('chat.form.voiceErrors.unsupportedRecording', 'Browser audio recording is unavailable.'));
      return;
    }

    try {
      setMediaError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      mediaStreamRef.current = stream;
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];
      recordingCancelledRef.current = false;
      recordingStartedAtRef.current = Date.now();

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        recordingCancelledRef.current = true;
        setMediaError(t('chat.form.voiceErrors.recordingFailed', 'Recording failed.'));
        if (recorder.state === 'inactive') {
          cleanupRecordingResources();
          return;
        }

        try {
          recorder.stop();
        } catch {
          cleanupRecordingResources();
        }
      };

      recorder.onstop = () => {
        const chunks = [...audioChunksRef.current];
        const cancelled = recordingCancelledRef.current;
        const mimeType = recorder.mimeType || chunks[0]?.type || 'audio/webm';
        cleanupRecordingResources();
        audioChunksRef.current = [];

        if (cancelled || chunks.length === 0 || !onTranscribeAudio) return;

        const audio = new Blob(chunks, { type: mimeType });
        setIsTranscribing(true);
        void onTranscribeAudio(audio)
          .then(appendTranscript)
          .catch((error: unknown) => {
            const messageText = error instanceof Error
              ? error.message
              : t('chat.form.voiceErrors.transcriptionFailed', 'Transcription failed.');
            setMediaError(messageText);
          })
          .finally(() => setIsTranscribing(false));
      };

      recorder.start();
      setIsRecording(true);
      setRecordingSeconds(0);
      stopRecordingTimer();
      recordingTimerRef.current = window.setInterval(() => {
        setRecordingSeconds(Math.floor((Date.now() - recordingStartedAtRef.current) / 1000));
      }, 250);
    } catch (error) {
      cleanupRecordingResources();
      const messageText = error instanceof DOMException && error.name === 'NotAllowedError'
        ? t('chat.form.voiceErrors.microphoneDenied', 'Microphone permission denied.')
        : t('chat.form.voiceErrors.microphoneUnavailable', 'Microphone unavailable.');
      setMediaError(messageText);
    }
  }, [
    appendTranscript,
    cleanupRecordingResources,
    isLoading,
    isTranscribing,
    onTranscribeAudio,
    secureRecordingContext,
    stopRecordingTimer,
    t,
    transcriptionReady,
  ]);

  const stopRecording = useCallback((cancelled: boolean) => {
    const recorder = mediaRecorderRef.current;
    if (!recorder) return;
    recordingCancelledRef.current = cancelled;
    if (recorder.state === 'inactive') {
      cleanupRecordingResources();
      return;
    }
    recorder.stop();
  }, [cleanupRecordingResources]);

  useEffect(() => {
    return () => {
      stopRecordingTimer();
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, [stopRecordingTimer]);

  // Refocus after loading completes
  const prevLoading = useRef(isLoading);
  useEffect(() => {
    if (prevLoading.current && !isLoading) {
      setTimeout(() => textareaRef.current?.focus(), 10);
    }
    prevLoading.current = isLoading;
  }, [isLoading]);

  useEffect(() => {
    filesRef.current = files;
  }, [files]);

  const clearFiles = useCallback(() => {
    setFiles((prev) => {
      prev.forEach((file) => {
        if (file.preview) URL.revokeObjectURL(file.preview);
      });
      return [];
    });
  }, []);

  useEffect(() => {
    if (!supportsImageAttachments) {
      setIsDragOver(false);
    }
    if (!supportsImageAttachments && files.length > 0) {
      clearFiles();
    }
  }, [supportsImageAttachments, files.length, clearFiles]);

  useEffect(() => {
    return () => {
      filesRef.current.forEach((file) => {
        if (file.preview) URL.revokeObjectURL(file.preview);
      });
    };
  }, []);

  const handleSubmit = useCallback(
    (e?: React.FormEvent) => {
      e?.preventDefault();
      if (isLoading || !canSend) return;
      onSend(message.trim(), files);
      setMessage('');
      clearFiles();
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    },
    [isLoading, canSend, message, files, onSend, clearFiles],
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  const addFiles = useCallback((fileList: File[]) => {
    const imageFiles = fileList.filter((file) => file.type.startsWith('image/'));
    if (imageFiles.length === 0) {
      return;
    }
    const newFiles: ChatUploadedFile[] = imageFiles.map((f) => ({
      id: `file-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      name: f.name,
      type: f.type,
      size: f.size,
      file: f,
      preview: f.type.startsWith('image/') ? URL.createObjectURL(f) : undefined,
    }));
    setFiles((prev) => [...prev, ...newFiles]);
  }, []);

  // Drag and drop
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!supportsImageAttachments) return;
    setIsDragOver(true);
  }, [supportsImageAttachments]);

  const handleDragLeave = useCallback(() => setIsDragOver(false), []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (!supportsImageAttachments) return;
    const dropped = Array.from(e.dataTransfer.files);
    addFiles(dropped);
  }, [addFiles, supportsImageAttachments]);

  const removeFile = useCallback((id: string) => {
    setFiles((prev) => {
      const file = prev.find((f) => f.id === id);
      if (file?.preview) URL.revokeObjectURL(file.preview);
      return prev.filter((f) => f.id !== id);
    });
  }, []);

  const formatRecordingSeconds = (seconds: number) => {
    const minutes = Math.floor(seconds / 60).toString().padStart(2, '0');
    const remaining = (seconds % 60).toString().padStart(2, '0');
    return `${minutes}:${remaining}`;
  };

  const showHeader = modelLabel || modelSelector || showThinkingToggle || ttftMs != null || tps != null || contextLength > 0;

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

      {isDragOver && <DragOverlay>{t('chat.form.dropFilesHere', 'Drop files here')}</DragOverlay>}

      {/* Header: model + thinking + stats */}
      {showHeader && (
        <HeaderRow>
          {(modelLabel || modelSelector) && (
            <ModelLine>
              <span>{t('chat.form.modelLabel', 'Model:')}</span>
              {modelSelector ?? <ModelBtn onClick={onOpenModelPicker}>{modelLabel}</ModelBtn>}
            </ModelLine>
          )}
          {contextLength > 0 && (
            <Stat>
              {t('chat.form.contextShort', '{count}K ctx', { count: (contextLength / 1024).toFixed(0) })}
            </Stat>
          )}
          {showThinkingToggle && (
            <ThinkingBtn $active={thinkingEnabled} onClick={onToggleThinking}>
              {t('chat.form.thinking', 'Thinking')}
            </ThinkingBtn>
          )}
          <Spacer />
          {ttftMs != null && (
            <Stat>
              {t('chat.form.ttftLabel', 'TTFT')} <StatValue>{ttftMs.toFixed(1)}{t('chat.form.milliseconds', 'ms')}</StatValue>
            </Stat>
          )}
          {tps != null && (
            <Stat>
              {t('chat.form.tpsLabel', 'TPS')} <StatValue>{tps.toFixed(1)} {t('chat.form.tokensPerSecond', 'tok/s')}</StatValue>{' '}
              ({(1000 / tps).toFixed(1)} {t('chat.form.millisecondsPerToken', 'ms/tok')})
            </Stat>
          )}
        </HeaderRow>
      )}

      {showVoiceControls && (
        <VoiceRow>
          <VoiceGroup>
            <VoiceLabel>{t('chat.form.sttLabel', 'STT')}</VoiceLabel>
            <VoiceSelect
              value={selectedTranscriptionId ?? ''}
              disabled={transcriptionModels.length === 0 || isRecording || isTranscribing}
              onChange={(event) => onSelectTranscriptionModel?.(event.target.value || null)}
              aria-label={t('chat.form.selectTranscriptionModel', 'Select transcription model')}
            >
              {transcriptionModels.length === 0 ? (
                <option value="">{t('chat.form.noTranscriptionModels', 'No STT')}</option>
              ) : (
                transcriptionModels.map((model) => (
                  <option key={model.modelId} value={model.modelId}>
                    {model.label}
                  </option>
                ))
              )}
            </VoiceSelect>
            {isRecording ? (
              <>
                <VoiceIconBtn
                  variant="danger"
                  size="sm"
                  icon
                  type="button"
                  onClick={() => stopRecording(false)}
                  aria-label={t('chat.form.stopRecording', 'Stop recording')}
                  title={t('chat.form.stopRecording', 'Stop recording')}
                  $active
                >
                  <FiSquare size={14} />
                </VoiceIconBtn>
                <VoiceIconBtn
                  variant="ghost"
                  size="sm"
                  icon
                  type="button"
                  onClick={() => stopRecording(true)}
                  aria-label={t('chat.form.cancelRecording', 'Cancel recording')}
                  title={t('chat.form.cancelRecording', 'Cancel recording')}
                >
                  <FiX size={15} />
                </VoiceIconBtn>
                <VoiceStatus>{formatRecordingSeconds(recordingSeconds)}</VoiceStatus>
              </>
            ) : (
              <VoiceIconBtn
                variant="ghost"
                size="sm"
                icon
                type="button"
                loading={isTranscribing}
                disabled={!canRecordAudio || isLoading}
                onClick={startRecording}
                aria-label={recordingUnavailableReason ?? t('chat.form.startRecording', 'Start recording')}
                title={recordingUnavailableReason ?? t('chat.form.startRecording', 'Start recording')}
              >
                <FiMic size={16} />
              </VoiceIconBtn>
            )}
          </VoiceGroup>

          <VoiceGroup>
            <VoiceLabel>{t('chat.form.ttsLabel', 'TTS')}</VoiceLabel>
            <VoiceSelect
              value={selectedSpeechId ?? ''}
              disabled={speechModels.length === 0}
              onChange={(event) => onSelectSpeechModel?.(event.target.value || null)}
              aria-label={t('chat.form.selectSpeechModel', 'Select speech model')}
            >
              {speechModels.length === 0 ? (
                <option value="">{t('chat.form.noSpeechModels', 'No TTS')}</option>
              ) : (
                speechModels.map((model) => (
                  <option key={model.modelId} value={model.modelId}>
                    {model.label}
                  </option>
                ))
              )}
            </VoiceSelect>
            {speechModels.length > 0 && (
              <VoiceInput
                value={selectedVoice ?? ''}
                onChange={(event) => onSelectedVoiceChange?.(event.target.value || null)}
                placeholder={t('chat.form.voicePlaceholder', 'voice')}
                aria-label={t('chat.form.voiceName', 'Voice')}
              />
            )}
            <VoiceToggle
              type="button"
              disabled={!speechReady}
              $active={autoSpeakAssistant}
              onClick={() => onAutoSpeakAssistantChange?.(!autoSpeakAssistant)}
            >
              {t('chat.form.autoSpeak', 'Auto')}
            </VoiceToggle>
            {isSpeaking ? (
              <VoiceIconBtn
                variant="ghost"
                size="sm"
                icon
                type="button"
                onClick={onStopSpeaking}
                aria-label={t('chat.form.stopSpeech', 'Stop speech')}
                title={t('chat.form.stopSpeech', 'Stop speech')}
                $active
              >
                <FiVolumeX size={16} />
              </VoiceIconBtn>
            ) : (
              <VoiceIconBtn
                variant="ghost"
                size="sm"
                icon
                type="button"
                disabled={!canSpeakDraft}
                onClick={() => { void onSpeakText?.(message.trim()); }}
                aria-label={t('chat.form.speakDraft', 'Speak draft')}
                title={t('chat.form.speakDraft', 'Speak draft')}
              >
                <FiVolume2 size={16} />
              </VoiceIconBtn>
            )}
          </VoiceGroup>

          {displayVoiceError && <VoiceStatus $error>{displayVoiceError}</VoiceStatus>}
        </VoiceRow>
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
          disabled={!supportsImageAttachments}
          aria-label={t('chat.form.attachFile', 'Attach file')}
        >
          <FiPaperclip size={17} />
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
          placeholder={inputPlaceholder}
          rows={1}
          autoFocus={autoFocus}
        />
        {isLoading ? (
          <SendBtn
            variant="danger"
            size="sm"
            icon
            type="button"
            onClick={onCancel}
            aria-label={t('chat.form.cancelGeneration', 'Cancel generation')}
          >
            <FiSquare size={15} />
          </SendBtn>
        ) : (
          <SendBtn
            variant="primary"
            size="sm"
            icon
            type="submit"
            disabled={!canSend}
            aria-label={t('chat.form.sendMessage', 'Send message')}
          >
            <FiSend size={15} />
          </SendBtn>
        )}
      </InputRow>

      <BottomAccentLine />
      <HelperText>
        {supportsImageAttachments
          ? t(
              'chat.form.helperWithImages',
              'Enter to send - Shift+Enter for new line - Drag & drop images',
            )
          : t('chat.form.helper', 'Enter to send - Shift+Enter for new line')}
      </HelperText>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => {
          if (e.target.files) addFiles(Array.from(e.target.files));
          e.target.value = '';
        }}
      />
    </Form>
  );
}
