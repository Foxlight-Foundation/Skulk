import { forwardRef } from 'react';
import styled, { css, keyframes } from 'styled-components';

export type ButtonVariant = 'primary' | 'outline' | 'ghost' | 'danger' | 'solid' | 'approve';
export type ButtonSize = 'sm' | 'md' | 'lg';

/** Shared action control with semantic treatments and loading state. */
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
  solid: css`
    font-weight: 600;
    color: ${({ theme }) => theme.colors.textOnAccent};
    background: ${({ theme }) => theme.colors.actionFill};
    border: none;
    &:hover:not(:disabled) { filter: brightness(1.08); }
  `,
  approve: css`
    font-weight: 600;
    color: ${({ theme }) => theme.colors.onLive};
    background: ${({ theme }) => theme.colors.approvalFill};
    border: none;
    &:hover:not(:disabled) { background: ${({ theme }) => theme.colors.liveHover}; }
  `,
  primary: css`
    color: ${({ theme }) => theme.colors.accentText};
    border: 1px solid ${({ theme }) => theme.colors.goldDim};
    background: transparent;

    &:hover:not(:disabled) {
      background: ${({ theme }) => theme.colors.goldBg};
      border-color: ${({ theme }) => theme.colors.goldDim};
    }

    &:active:not(:disabled) {
      background: ${({ theme }) => theme.colors.goldBg};
    }
  `,
  outline: css`
    color: ${({ theme }) => theme.colors.textSecondary};
    border: 1px solid ${({ theme }) => theme.colors.border};
    background: transparent;

    &:hover:not(:disabled) {
      color: ${({ theme }) => theme.colors.accentText};
      border-color: ${({ theme }) => theme.colors.goldDim};
    }

    &:active:not(:disabled) {
      background: ${({ theme }) => theme.colors.goldBg};
    }
  `,
  ghost: css`
    color: ${({ theme }) => theme.colors.textMuted};
    border: 1px solid transparent;
    background: transparent;

    &:hover:not(:disabled) {
      color: ${({ theme }) => theme.colors.accentText};
      background: ${({ theme }) => theme.colors.goldBg};
    }

    &:active:not(:disabled) {
      background: ${({ theme }) => theme.colors.goldBg};
    }
  `,
  danger: css`
    color: ${({ theme }) => theme.colors.textSecondary};
    border: 1px solid ${({ theme }) => theme.colors.border};
    background: transparent;

    &:hover:not(:disabled) {
      color: ${({ theme }) => theme.colors.error};
      border-color: ${({ theme }) => theme.colors.borderDanger};
      background: ${({ theme }) => theme.colors.errorBg};
    }

    &:active:not(:disabled) {
      background: ${({ theme }) => theme.colors.errorBg};
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
  transition: color 120ms, background 120ms, border-color 120ms;
  white-space: normal;
  overflow-wrap: anywhere;
  text-align: center;
  min-width: 0;
  max-width: 100%;
  user-select: none;

  /* Minimum height preserves reference sizing while allowing translated labels to wrap. */
  min-height: ${({ $size }) => sizeTokens[$size].height};
  font-size: ${({ $size, theme }) => theme.fontSizes[sizeFontMap[$size] as keyof typeof theme.fontSizes]};
  ${({ $icon, $size, $variant }) =>
    $icon
      ? css`
          width: ${sizeTokens[$size].iconSize};
          flex-shrink: 0;
          padding: 0;
        `
      : css`
          padding: ${$size === 'md' && ($variant === 'solid' || $variant === 'approve') ? '0 16px' : sizeTokens[$size].padding};
        `}

  /* Block */
  ${({ $block }) => $block && css`width: 100%;`}

  /* Variant */
  ${({ $variant }) => variantStyles[$variant]}

  @media (pointer: coarse) {
    min-height: 44px;
    ${({ $icon }) => $icon && css`min-width: 44px;`}
  }

  /* Disabled */
  &:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  /* Keyboard focus — all: unset removes the browser outline. */
  &:focus-visible {
    outline: none;
    box-shadow: ${({ theme }) => theme.colors.focusRing};
  }
`;

/* ---- component ---- */

/** Render an accessible action whose label can wrap without clipping translations. */
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
      aria-busy={loading || undefined}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? <Spinner $size={size} aria-hidden="true" /> : null}
      {children}
    </StyledButton>
  ),
);

Button.displayName = 'Button';
