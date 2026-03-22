/**
 * FamilySidebar
 *
 * Left column of the model picker listing model families for quick navigation.
 */
import React from 'react';
import styled from 'styled-components';
import { FamilyLogo } from './FamilyLogos';

// ─── Styled components ────────────────────────────────────────────────────────

const Sidebar = styled.div`
  width: 160px;
  flex-shrink: 0;
  border-right: 1px solid ${({ theme }) => theme.colors.border};
  overflow-y: auto;
  display: flex;
  flex-direction: column;
`;

const SidebarItem = styled.button<{ $active: boolean }>`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: ${({ theme, $active }) =>
    $active ? 'oklch(0.85 0.18 85 / 0.08)' : 'transparent'};
  border: none;
  border-left: 2px solid ${({ theme, $active }) =>
    $active ? theme.colors.yellow : 'transparent'};
  cursor: pointer;
  width: 100%;
  text-align: left;
  transition: background ${({ theme }) => theme.transitions.fast};
  &:hover { background: ${({ theme }) => theme.colors.mediumGray}; }
`;

const FamilyLabel = styled.span`
  font-size: 11px;
  letter-spacing: 0.05em;
  color: ${({ theme }) => theme.colors.foreground};
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

// ─── Component ─────────────────────────────────────────────────────────────────

export interface FamilySidebarProps {
  families: string[];
  activeFamily: string | null;
  onSelect: (family: string) => void;
}

export const FamilySidebar: React.FC<FamilySidebarProps> = ({
  families,
  activeFamily,
  onSelect,
}) => (
  <Sidebar>
    {families.map((f) => (
      <SidebarItem
        key={f}
        $active={f === activeFamily}
        onClick={() => onSelect(f)}
      >
        <FamilyLogo family={f} size={16} />
        <FamilyLabel>{f}</FamilyLabel>
      </SidebarItem>
    ))}
  </Sidebar>
);
