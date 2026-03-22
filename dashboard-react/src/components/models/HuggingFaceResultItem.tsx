/**
 * HuggingFaceResultItem
 *
 * Displays a single HuggingFace model search result.
 */
import React from 'react';
import styled from 'styled-components';
import { useTranslate } from '@tolgee/react';
import type { HuggingFaceModel } from '../../api/types';

// ─── Styled components ────────────────────────────────────────────────────────

const Item = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  cursor: pointer;
  transition: background ${({ theme }) => theme.transitions.fast};
  &:hover { background: ${({ theme }) => theme.colors.mediumGray}; }
`;

const Info = styled.div`
  flex: 1;
  min-width: 0;
`;

const ModelId = styled.div`
  font-size: 12px;
  letter-spacing: 0.03em;
  color: ${({ theme }) => theme.colors.foreground};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const Stats = styled.div`
  display: flex;
  gap: 10px;
  margin-top: 2px;
`;

const Stat = styled.span`
  font-size: 10px;
  color: ${({ theme }) => theme.colors.mutedForeground};
`;

const AddBtn = styled.button`
  font-size: 10px;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  padding: 3px 9px;
  border-radius: ${({ theme }) => theme.radius.sm};
  border: 1px solid ${({ theme }) => theme.colors.yellow};
  background: transparent;
  color: ${({ theme }) => theme.colors.yellow};
  cursor: pointer;
  transition: background ${({ theme }) => theme.transitions.fast};
  &:hover {
    background: ${({ theme }) => theme.colors.yellow};
    color: ${({ theme }) => theme.colors.black};
  }
`;

// ─── Component ─────────────────────────────────────────────────────────────────

export interface HuggingFaceResultItemProps {
  model: HuggingFaceModel;
  onAdd?: (model: HuggingFaceModel) => void;
}

export const HuggingFaceResultItem: React.FC<HuggingFaceResultItemProps> = ({
  model,
  onAdd,
}) => {
  const { t } = useTranslate();

  return (
    <Item>
      <Info>
        <ModelId title={model.id}>{model.id}</ModelId>
        <Stats>
          {model.pipeline_tag && <Stat>{model.pipeline_tag}</Stat>}
          {model.downloads != null && (
            <Stat>⬇ {(model.downloads / 1000).toFixed(0)}k</Stat>
          )}
          {model.likes != null && <Stat>♥ {model.likes}</Stat>}
        </Stats>
      </Info>
      <AddBtn
        onClick={(e) => { e.stopPropagation(); onAdd?.(model); }}
        aria-label={`Add ${model.id}`}
      >
        {t('models.download_model', { name: '' }).trim() || 'Add'}
      </AddBtn>
    </Item>
  );
};
