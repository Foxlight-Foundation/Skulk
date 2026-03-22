/**
 * ModelCard
 *
 * Displays a model with memory requirements, download status, and
 * a launch / download action button.
 * Ported from ModelCard.svelte.
 */
import React from 'react';
import styled, { css } from 'styled-components';
import { useTranslate } from '@tolgee/react';
import type { ModelEntry } from '../../stores/modelsStore';
import { useFavoritesStore } from '../../stores/favoritesStore';

// ─── Styled components ────────────────────────────────────────────────────────

const Card = styled.div<{ $selected?: boolean }>`
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: ${({ theme }) => theme.radius.md};
  border: 1px solid ${({ theme, $selected }) =>
    $selected ? theme.colors.yellow : theme.colors.border};
  background: ${({ theme, $selected }) =>
    $selected ? 'oklch(0.85 0.18 85 / 0.06)' : theme.colors.darkGray};
  cursor: pointer;
  transition: border-color ${({ theme }) => theme.transitions.fast},
    background ${({ theme }) => theme.transitions.fast};
  &:hover {
    border-color: ${({ theme }) => theme.colors.yellowDarker};
    background: oklch(0.85 0.18 85 / 0.04);
  }
`;

const Info = styled.div`
  flex: 1;
  min-width: 0;
`;

const Name = styled.div`
  font-size: 12px;
  letter-spacing: 0.04em;
  color: ${({ theme }) => theme.colors.foreground};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const Meta = styled.div`
  display: flex;
  gap: 8px;
  margin-top: 2px;
`;

const Tag = styled.span`
  font-size: 10px;
  letter-spacing: 0.06em;
  color: ${({ theme }) => theme.colors.mutedForeground};
  text-transform: uppercase;
`;

const ProgressBar = styled.div<{ $pct: number }>`
  height: 2px;
  margin-top: 4px;
  background: ${({ theme }) => theme.colors.mediumGray};
  border-radius: 1px;
  overflow: hidden;
  &::after {
    content: '';
    display: block;
    height: 100%;
    width: ${({ $pct }) => $pct}%;
    background: ${({ theme }) => theme.colors.yellow};
    transition: width 0.3s ease;
  }
`;

const Actions = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
`;

const LaunchBtn = styled.button<{ $downloading?: boolean }>`
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 4px 10px;
  border-radius: ${({ theme }) => theme.radius.sm};
  border: 1px solid ${({ theme, $downloading }) =>
    $downloading ? theme.colors.border : theme.colors.yellow};
  background: ${({ theme, $downloading }) =>
    $downloading ? 'transparent' : 'oklch(0.85 0.18 85 / 0.1)'};
  color: ${({ theme, $downloading }) =>
    $downloading ? theme.colors.mutedForeground : theme.colors.yellow};
  cursor: pointer;
  transition: background ${({ theme }) => theme.transitions.fast};
  &:hover:not(:disabled) {
    background: ${({ theme }) => theme.colors.yellow};
    color: ${({ theme }) => theme.colors.black};
  }
  &:disabled { opacity: 0.4; cursor: default; }
`;

const FavBtn = styled.button<{ $active: boolean }>`
  background: none;
  border: none;
  cursor: pointer;
  font-size: 14px;
  color: ${({ $active, theme }) => ($active ? theme.colors.yellow : theme.colors.border)};
  padding: 2px;
  transition: color ${({ theme }) => theme.transitions.fast};
  &:hover { color: ${({ theme }) => theme.colors.yellow}; }
`;

// ─── Helpers ───────────────────────────────────────────────────────────────────

function formatSize(bytes: number | undefined): string {
  if (!bytes) return '';
  const gb = bytes / (1024 ** 3);
  return gb >= 1 ? `${gb.toFixed(1)}GB` : `${(bytes / (1024 ** 2)).toFixed(0)}MB`;
}

function getShortId(id: string): string {
  return id.split('/').pop() ?? id;
}

// ─── Component ─────────────────────────────────────────────────────────────────

export interface ModelCardProps {
  model: ModelEntry;
  selected?: boolean;
  onLaunch?: (id: string) => void;
  onDownload?: (id: string) => void;
}

export const ModelCard: React.FC<ModelCardProps> = ({
  model,
  selected,
  onLaunch,
  onDownload,
}) => {
  const { t } = useTranslate();
  const isFav = useFavoritesStore((s) => s.isFavorite(model.id));
  const toggleFavorite = useFavoritesStore((s) => s.toggleFavorite);

  const handleAction = () => {
    if (model.isDownloaded) {
      onLaunch?.(model.id);
    } else if (!model.isDownloading) {
      onDownload?.(model.id);
    }
  };

  return (
    <Card $selected={selected} onClick={handleAction}>
      <Info>
        <Name title={model.id}>{getShortId(model.id)}</Name>
        <Meta>
          {model.quantization && <Tag>{model.quantization}</Tag>}
          {model.sizeBytes != null && <Tag>{formatSize(model.sizeBytes)}</Tag>}
          {model.num_params != null && (
            <Tag>{(model.num_params / 1e9).toFixed(1)}B</Tag>
          )}
        </Meta>
        {model.isDownloading && model.downloadProgress != null && (
          <ProgressBar $pct={model.downloadProgress} />
        )}
      </Info>

      <Actions>
        <FavBtn
          $active={isFav}
          onClick={(e) => { e.stopPropagation(); toggleFavorite(model.id); }}
          aria-label={t('models.toggle_favorite')}
          title={t('models.toggle_favorite')}
        >
          {isFav ? '★' : '☆'}
        </FavBtn>

        <LaunchBtn
          $downloading={model.isDownloading}
          disabled={model.isDownloading}
          onClick={(e) => { e.stopPropagation(); handleAction(); }}
          aria-label={model.isDownloaded
            ? t('models.launch_model', { name: getShortId(model.id) })
            : t('models.download_model', { name: getShortId(model.id) })}
        >
          {model.isDownloading
            ? `${model.downloadProgress ?? 0}%`
            : model.isDownloaded
              ? t('models.launch_model', { name: '' }).trim() || 'Launch'
              : t('common.download')}
        </LaunchBtn>
      </Actions>
    </Card>
  );
};
