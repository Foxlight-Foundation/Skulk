/**
 * ModelPickerGroup
 *
 * Collapsible group of ModelCards for a single model family.
 */
import React, { useState } from 'react';
import styled from 'styled-components';
import { FamilyLogo } from './FamilyLogos';
import { ModelCard } from './ModelCard';
import type { ModelEntry } from '../../stores/modelsStore';

// ─── Styled components ────────────────────────────────────────────────────────

const Group = styled.div`
  display: flex;
  flex-direction: column;
`;

const Header = styled.button`
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 8px 12px;
  background: none;
  border: none;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  cursor: pointer;
  text-align: left;
  transition: background ${({ theme }) => theme.transitions.fast};
  &:hover { background: ${({ theme }) => theme.colors.mediumGray}; }
`;

const FamilyName = styled.span`
  flex: 1;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.foreground};
`;

const Count = styled.span`
  font-size: 10px;
  color: ${({ theme }) => theme.colors.mutedForeground};
`;

const Chevron = styled.span<{ $open: boolean }>`
  font-size: 10px;
  color: ${({ theme }) => theme.colors.lightGray};
  transform: ${({ $open }) => ($open ? 'rotate(90deg)' : 'rotate(0deg)')};
  transition: transform ${({ theme }) => theme.transitions.fast};
`;

const Cards = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px 10px;
`;

// ─── Component ─────────────────────────────────────────────────────────────────

export interface ModelPickerGroupProps {
  family: string;
  models: ModelEntry[];
  defaultOpen?: boolean;
  onLaunch?: (id: string) => void;
  onDownload?: (id: string) => void;
}

export const ModelPickerGroup: React.FC<ModelPickerGroupProps> = ({
  family,
  models,
  defaultOpen = true,
  onLaunch,
  onDownload,
}) => {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <Group>
      <Header onClick={() => setOpen((p) => !p)}>
        <FamilyLogo family={family} size={18} />
        <FamilyName>{family}</FamilyName>
        <Count>{models.length}</Count>
        <Chevron $open={open}>▶</Chevron>
      </Header>
      {open && (
        <Cards>
          {models.map((m) => (
            <ModelCard
              key={m.id}
              model={m}
              onLaunch={onLaunch}
              onDownload={onDownload}
            />
          ))}
        </Cards>
      )}
    </Group>
  );
};
