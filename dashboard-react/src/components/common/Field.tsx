import { forwardRef } from 'react';
import styled, { css } from 'styled-components';

export type FieldSize = 'sm' | 'md' | 'lg';

export interface FieldProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'> {
  /** Optional leading icon element (rendered left of the input). */
  icon?: React.ReactNode;
  /** Optional trailing element (rendered right of the input). */
  rightElement?: React.ReactNode;
  /** Visual size preset. */
  size?: FieldSize;
}

const sizeStyles: Record<FieldSize, ReturnType<typeof css>> = {
  sm: css`
    height: 30px;
    font-size: ${({ theme }) => theme.fontSizes.sm};
    padding: 0 8px;
    gap: 6px;
  `,
  md: css`
    height: 36px;
    font-size: 16px;
    padding: 0 10px;
    gap: 8px;
  `,
  lg: css`
    height: 42px;
    font-size: 16px;
    padding: 0 12px;
    gap: 10px;
  `,
};

const Wrapper = styled.label<{ $size: FieldSize; $disabled?: boolean }>`
  display: flex;
  align-items: center;
  background: ${({ theme }) => theme.colors.surfaceHover};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  color: ${({ theme }) => theme.colors.text};
  transition: border-color 0.15s;
  cursor: text;
  ${({ $size }) => sizeStyles[$size]}

  &:focus-within {
    border-color: ${({ theme }) => theme.colors.borderStrong};
    box-shadow: ${({ theme }) => theme.colors.focusRing};
  }

  ${({ $disabled }) =>
    $disabled &&
    css`
      opacity: 0.5;
      cursor: not-allowed;
    `}
`;

const IconSlot = styled.span`
  display: flex;
  align-items: center;
  flex-shrink: 0;
  color: ${({ theme }) => theme.colors.subtleText};
`;

const Input = styled.input`
  all: unset;
  flex: 1;
  min-width: 0;
  color: inherit;
  font: inherit;

  /* The wrapper owns the focus ring; an input outline would cut through it. */
  &:focus-visible { outline: none; }

  &::placeholder {
    color: ${({ theme }) => theme.colors.subtleText};
  }

  &:disabled {
    cursor: not-allowed;
  }
`;

export const Field = forwardRef<HTMLInputElement, FieldProps>(
  ({ icon, rightElement, size = 'md', disabled, className, ...inputProps }, ref) => (
    <Wrapper $size={size} $disabled={disabled} className={className}>
      {icon && <IconSlot>{icon}</IconSlot>}
      <Input ref={ref} disabled={disabled} {...inputProps} />
      {rightElement && <IconSlot>{rightElement}</IconSlot>}
    </Wrapper>
  ),
);

Field.displayName = 'Field';
