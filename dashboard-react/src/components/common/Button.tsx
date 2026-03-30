import { forwardRef } from 'react';
import styled, { css, keyframes } from 'styled-components';

export type ButtonVariant = 'primary' | 'outline' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** Render as icon-only (square, centered content). */
  icon?: boolean;
  /** Show a loading spinner and disable interaction. */
  loading?: boolean;
  /** Full width. */
  block?: boolean;
}

/* ---- size tokens ---- */

const sizeTokens = {
  sm: { height: '30px', padding: '0 10px', iconSize: '30px' },
  md: { height: '36px', padding: '0 14px', iconSize: '36px' },
  lg: { height: '42px', padding: '0 18px', iconSize: '42px' },
};

const sizeFontMap: Record<ButtonSize, string> = {
  sm: 'xs',
  md: 'sm',
  lg: 'nav',
};

/* ---- variant styles ---- */

const variantStyles: Record<ButtonVariant, ReturnType<typeof css>> = {
  primary: css`
    color: #FFD700;
    border: 1px solid rgba(255, 215, 0, 0.4);
    background: transparent;

    &:hover:not(:disabled) {
      background: rgba(255, 215, 0, 0.1);
      border-color: rgba(255, 215, 0, 0.6);
    }

    &:active:not(:disabled) {
      background: rgba(255, 215, 0, 0.15);
    }
  `,
  outline: css`
    color: rgba(179, 179, 179, 0.9);
    border: 1px solid rgba(80, 80, 80, 0.4);
    background: transparent;

    &:hover:not(:disabled) {
      color: #FFD700;
      border-color: rgba(255, 215, 0, 0.4);
    }

    &:active:not(:disabled) {
      background: rgba(255, 215, 0, 0.05);
    }
  `,
  ghost: css`
    color: ${({ theme }) => theme.colors.textMuted};
    border: 1px solid transparent;
    background: transparent;

    &:hover:not(:disabled) {
      color: #FFD700;
      background: rgba(255, 215, 0, 0.08);
    }

    &:active:not(:disabled) {
      background: rgba(255, 215, 0, 0.12);
    }
  `,
  danger: css`
    color: rgba(179, 179, 179, 0.8);
    border: 1px solid rgba(80, 80, 80, 0.4);
    background: transparent;

    &:hover:not(:disabled) {
      color: #ef4444;
      border-color: rgba(239, 68, 68, 0.4);
      background: rgba(239, 68, 68, 0.08);
    }

    &:active:not(:disabled) {
      background: rgba(239, 68, 68, 0.12);
    }
  `,
};

/* ---- spinner ---- */

const spin = keyframes`
  to { transform: rotate(360deg); }
`;

const Spinner = styled.span<{ $size: ButtonSize }>`
  display: inline-block;
  width: ${({ $size }) => ($size === 'sm' ? '10px' : '12px')};
  height: ${({ $size }) => ($size === 'sm' ? '10px' : '12px')};
  border: 1.5px solid currentColor;
  border-top-color: transparent;
  border-radius: 50%;
  animation: ${spin} 0.6s linear infinite;
`;

/* ---- styled button ---- */

const StyledButton = styled.button<{
  $variant: ButtonVariant;
  $size: ButtonSize;
  $icon: boolean;
  $block: boolean;
}>`
  all: unset;
  box-sizing: border-box;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  font-family: ${({ theme }) => theme.fonts.body};
  border-radius: ${({ theme }) => theme.radii.md};
  transition: all 0.15s;
  white-space: nowrap;
  user-select: none;

  /* Size */
  height: ${({ $size }) => sizeTokens[$size].height};
  font-size: ${({ $size, theme }) => theme.fontSizes[sizeFontMap[$size] as keyof typeof theme.fontSizes]};
  ${({ $icon, $size }) =>
    $icon
      ? css`
          width: ${sizeTokens[$size].iconSize};
          padding: 0;
        `
      : css`
          padding: ${sizeTokens[$size].padding};
        `}

  /* Block */
  ${({ $block }) => $block && css`width: 100%;`}

  /* Variant */
  ${({ $variant }) => variantStyles[$variant]}

  /* Disabled */
  &:disabled {
    opacity: 0.88;
    cursor: not-allowed;
  }
`;

/* ---- component ---- */

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = 'outline',
      size = 'md',
      icon = false,
      loading = false,
      block = false,
      disabled,
      children,
      ...rest
    },
    ref,
  ) => (
    <StyledButton
      ref={ref}
      $variant={variant}
      $size={size}
      $icon={icon}
      $block={block}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? <Spinner $size={size} /> : children}
    </StyledButton>
  ),
);

Button.displayName = 'Button';
