import styled from 'styled-components';
import type { ChatUploadedFile } from '../../types/chat';
import { getFileIcon, formatFileSize, truncateFileName } from '../../types/chat';
import { Button } from '../common/Button';

export interface ChatAttachmentsProps {
  files: ChatUploadedFile[];
  readonly?: boolean;
  onRemove?: (fileId: string) => void;
}

const Container = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 0 4px;
  margin-bottom: 12px;
`;

const FileCard = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
  border-radius: ${({ theme }) => theme.radii.md};
  transition: all 0.15s;

  &:hover {
    border-color: ${({ theme }) => theme.colors.goldDim};
    box-shadow: 0 0 8px ${({ theme }) => theme.colors.goldBg};
  }
`;

const Thumbnail = styled.img`
  width: 32px;
  height: 32px;
  object-fit: cover;
  border-radius: ${({ theme }) => theme.radii.sm};
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
`;

const IconEmoji = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.xl};
  line-height: 1;
`;

const FileInfo = styled.div`
  display: flex;
  flex-direction: column;
  min-width: 0;
`;

const FileName = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.gold};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 120px;
`;

const FileSize = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const RemoveBtn = styled(Button)`
  &:hover:not(:disabled) {
    color: ${({ theme }) => theme.colors.error};
    background: transparent;
  }
`;

export function ChatAttachments({ files, readonly = false, onRemove }: ChatAttachmentsProps) {
  if (files.length === 0) return null;

  return (
    <Container>
      {files.map((file) => (
        <FileCard key={file.id}>
          {file.preview ? (
            <Thumbnail src={file.preview} alt={file.name} />
          ) : (
            <IconEmoji>{getFileIcon(file.type, file.name)}</IconEmoji>
          )}
          <FileInfo>
            <FileName title={file.name}>{truncateFileName(file.name)}</FileName>
            <FileSize>{formatFileSize(file.size)}</FileSize>
          </FileInfo>
          {!readonly && onRemove && (
            <RemoveBtn variant="ghost" size="sm" icon onClick={() => onRemove(file.id)} aria-label={`Remove ${file.name}`}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M6 18L18 6M6 6l12 12" />
              </svg>
            </RemoveBtn>
          )}
        </FileCard>
      ))}
    </Container>
  );
}
