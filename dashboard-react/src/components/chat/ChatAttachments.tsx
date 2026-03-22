/**
 * ChatAttachments
 *
 * Displays attached files as thumbnails or icons with optional remove buttons.
 * Ported from ChatAttachments.svelte.
 */
import React from 'react';
import styled from 'styled-components';
import type { ChatUploadedFile } from '../../types/files';
import { formatFileSize } from '../../types/files';

// ─── Styled components ────────────────────────────────────────────────────────

const List = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
`;

const Item = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border-radius: ${({ theme }) => theme.radius.md};
  background: ${({ theme }) => theme.colors.mediumGray};
  border: 1px solid ${({ theme }) => theme.colors.border};
  max-width: 180px;
  min-width: 0;
`;

const Thumb = styled.img`
  width: 28px;
  height: 28px;
  object-fit: cover;
  border-radius: 2px;
  flex-shrink: 0;
`;

const FileIcon = styled.span`
  font-size: 16px;
  flex-shrink: 0;
`;

const FileMeta = styled.div`
  flex: 1;
  min-width: 0;
`;

const FileName = styled.div`
  font-size: 10px;
  letter-spacing: 0.04em;
  color: ${({ theme }) => theme.colors.foreground};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const FileSize = styled.div`
  font-size: 9px;
  color: ${({ theme }) => theme.colors.mutedForeground};
`;

const RemoveButton = styled.button`
  flex-shrink: 0;
  background: none;
  border: none;
  color: ${({ theme }) => theme.colors.lightGray};
  cursor: pointer;
  font-size: 12px;
  padding: 0;
  line-height: 1;
  &:hover { color: ${({ theme }) => theme.colors.destructive}; }
`;

// ─── Helpers ───────────────────────────────────────────────────────────────────

function getFileIcon(file: ChatUploadedFile): string {
  if (file.type.startsWith('image/')) return '🖼';
  if (file.type === 'application/pdf') return '📕';
  if (file.type.startsWith('audio/')) return '🎵';
  if (file.type.startsWith('text/') || file.type === 'application/json') return '📄';
  return '📎';
}

function truncateName(name: string, maxLen = 20): string {
  if (name.length <= maxLen) return name;
  const dotIdx = name.lastIndexOf('.');
  const ext = dotIdx > 0 ? name.slice(dotIdx) : '';
  const base = dotIdx > 0 ? name.slice(0, dotIdx) : name;
  const available = maxLen - ext.length - 3;
  return base.slice(0, available) + '…' + ext;
}

// ─── Component ─────────────────────────────────────────────────────────────────

export interface ChatAttachmentsProps {
  files: ChatUploadedFile[];
  onRemove?: (id: string) => void;
  readonly?: boolean;
}

export const ChatAttachments: React.FC<ChatAttachmentsProps> = ({
  files,
  onRemove,
  readonly = false,
}) => {
  if (files.length === 0) return null;

  return (
    <List>
      {files.map((file) => (
        <Item key={file.id} title={file.name}>
          {file.preview ? (
            <Thumb src={file.preview} alt={file.name} />
          ) : (
            <FileIcon>{getFileIcon(file)}</FileIcon>
          )}
          <FileMeta>
            <FileName>{truncateName(file.name)}</FileName>
            <FileSize>{formatFileSize(file.size)}</FileSize>
          </FileMeta>
          {!readonly && onRemove && (
            <RemoveButton
              onClick={() => onRemove(file.id)}
              aria-label={`Remove ${file.name}`}
              title="Remove"
            >
              ✕
            </RemoveButton>
          )}
        </Item>
      ))}
    </List>
  );
};
