import styled from 'styled-components';

/** Shared menu surface used by Chat and dashboard selectors. */
export const ChoiceMenu = styled.div`
  z-index: 100; width: min(360px, calc(100vw - 24px)); max-height: min(420px, calc(100dvh - 80px)); overflow-y: auto;
  background: ${({ theme }) => theme.colors.surface}; border: 1px solid ${({ theme }) => theme.colors.borderControl};
  border-radius: 12px; padding: 6px; box-shadow: ${({ theme }) => theme.colors.shadowPop};
`;

/** Shared selectable row, including selected and keyboard focus treatments. */
export const ChoiceOption = styled.button<{ $fabric: boolean; $selected: boolean }>`
  display: flex; align-items: center; gap: 10px; width: 100%; padding: 9px 10px;
  border: 1px solid ${({ theme, $selected }) => $selected ? theme.colors.borderLive : 'transparent'};
  background: ${({ theme, $selected }) => $selected ? theme.colors.liveBg : 'transparent'};
  border-radius: 8px; text-align: left; color: ${({ theme }) => theme.colors.text};
  font: 14px ${({ theme }) => theme.fonts.body};
  > svg { flex-shrink: 0; color: ${({ theme, $fabric }) => $fabric ? theme.colors.live : theme.colors.gold}; }
  > span { min-width: 0; flex: 1; overflow-wrap: anywhere; }
  strong { font-weight: 600; } small { display: block; margin-top: 3px; font-size: 12px; line-height: 1.4; color: ${({ theme }) => theme.colors.textSecondary}; }
  &:disabled { opacity: .45; cursor: not-allowed; }
  &:hover:not(:disabled) { background: ${({ theme, $selected }) => $selected ? theme.colors.liveBg : theme.colors.surfaceHover}; }
  &:focus-visible { outline: none; box-shadow: inset 0 0 0 2px ${({ theme }) => theme.colors.gold}; }
`;
