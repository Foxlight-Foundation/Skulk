import styled from 'styled-components';

/** Compact settings switch. Callers supply an accessible label and own the draft value. */
export const Toggle = styled.button.attrs<{ $on: boolean }>(props => ({ type: 'button', role: 'switch', 'aria-checked': props.$on }))`
  all: unset; box-sizing: border-box; cursor: pointer; width: 36px; height: 20px;
  border-radius: 10px; position: relative; flex-shrink: 0;
  background: ${({ $on, theme }) => $on ? theme.colors.gold : theme.colors.surfaceSunken};
  border: 1px solid ${({ $on, theme }) => $on ? theme.colors.gold : theme.colors.border};
  transition: background ${({ theme }) => theme.motion.normal};
  &:focus-visible { box-shadow: ${({ theme }) => theme.colors.focusRing}; }
  &:disabled { opacity: .45; cursor: not-allowed; }
  &::after {
    content: ''; position: absolute; top: 2px; left: ${({ $on }) => $on ? '18px' : '2px'};
    width: 14px; height: 14px; border-radius: 50%;
    background: ${({ $on, theme }) => $on ? theme.colors.surface : theme.colors.selected};
    transition: left ${({ theme }) => theme.motion.normal};
  }
  @media (pointer: coarse) { &::before { content: ''; position: absolute; inset: -12px -4px; } }
`;
