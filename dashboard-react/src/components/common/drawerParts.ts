import styled from 'styled-components';

/** Tab strip styling shared by drawers that host several views. */
export const DrawerTabBar = styled.div`
  display: flex;
  gap: 4px;
  padding: 8px 12px 0;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
`;

/** One tab button in a `DrawerTabBar`. */
export const DrawerTabButton = styled.button<{ $active: boolean }>`
  all: unset;
  cursor: pointer;
  padding: 6px 14px 8px;
  border-radius: ${({ theme }) => theme.radii.sm} ${({ theme }) => theme.radii.sm} 0 0;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ $active, theme }) => ($active ? theme.colors.gold : theme.colors.textSecondary)};
  border-bottom: 2px solid
    ${({ $active, theme }) => ($active ? theme.colors.gold : 'transparent')};
  transition: color 0.15s, border-color 0.15s;

  &:hover {
    color: ${({ theme }) => theme.colors.text};
  }

  /* Visible focus ring for keyboard users; 'all: unset' strips the default. */
  &:focus-visible {
    outline: 2px solid ${({ theme }) => theme.colors.goldDim};
    outline-offset: 2px;
  }
`;

/**
 * Tab content host: a flex column with `overflow: hidden` so each tab decides
 * its own internal layout and scroll surface.
 */
export const DrawerBody = styled.div`
  flex: 1;
  min-height: 0;
  overflow: hidden;
  padding: 16px 18px;
  display: flex;
  flex-direction: column;
`;
