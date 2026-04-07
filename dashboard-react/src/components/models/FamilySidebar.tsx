import styled, { css } from 'styled-components';
import { FamilyLogos } from './FamilyLogos';

export interface FamilySidebarProps {
  families: string[];
  selectedFamily: string | null;
  hasFavorites: boolean;
  hasRecents: boolean;
  onSelect: (family: string | null) => void;
}

const Sidebar = styled.nav`
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: ${({ theme }) => theme.spacing.sm} 0;
  min-width: 48px;
  overflow-y: auto;
`;

const Divider = styled.hr`
  border: none;
  border-top: 1px solid ${({ theme }) => theme.colors.border};
  margin: ${({ theme }) => theme.spacing.xs} ${({ theme }) => theme.spacing.sm};
`;

const NavButton = styled.button<{ $active: boolean }>`
  all: unset;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  margin: 0 auto;
  border-radius: ${({ theme }) => theme.radii.md};
  cursor: pointer;
  color: ${({ theme }) => theme.colors.textMuted};
  transition: background 0.15s, color 0.15s, border-color 0.15s;
  border-left: 2px solid transparent;

  &:hover {
    background: ${({ theme }) => theme.colors.surfaceHover};
    color: ${({ theme }) => theme.colors.text};
  }

  ${({ $active }) =>
    $active &&
    css`
      background: ${({ theme }) => theme.colors.goldBg};
      color: ${({ theme }) => theme.colors.gold};
      border-left-color: ${({ theme }) => theme.colors.gold};
    `}
`;

const GridIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M3 3h8v8H3V3zm10 0h8v8h-8V3zM3 13h8v8H3v-8zm10 0h8v8h-8v-8z" />
  </svg>
);

export function FamilySidebar({
  families,
  selectedFamily,
  hasFavorites,
  hasRecents,
  onSelect,
}: FamilySidebarProps) {
  return (
    <Sidebar>
      {/* All models */}
      <NavButton
        $active={selectedFamily === null}
        onClick={() => onSelect(null)}
        title="All models"
      >
        <GridIcon />
      </NavButton>

      {/* Favorites */}
      {hasFavorites && (
        <NavButton
          $active={selectedFamily === 'favorites'}
          onClick={() => onSelect('favorites')}
          title="Favorites"
        >
          <FamilyLogos family="favorites" />
        </NavButton>
      )}

      {/* Recents */}
      {hasRecents && (
        <NavButton
          $active={selectedFamily === 'recents'}
          onClick={() => onSelect('recents')}
          title="Recent"
        >
          <FamilyLogos family="recents" />
        </NavButton>
      )}

      {/* HuggingFace Hub */}
      <NavButton
        $active={selectedFamily === 'huggingface'}
        onClick={() => onSelect('huggingface')}
        title="HuggingFace Hub"
      >
        <FamilyLogos family="huggingface" />
      </NavButton>

      {/* Model families */}
      {families.map((f) => (
        <NavButton
          key={f}
          $active={selectedFamily === f}
          onClick={() => onSelect(f)}
          title={f}
        >
          <FamilyLogos family={f} />
        </NavButton>
      ))}
    </Sidebar>
  );
}
