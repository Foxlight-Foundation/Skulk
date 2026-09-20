import styled, { css } from 'styled-components';

export type SegmentedControlSize = 'sm' | 'md' | 'lg';

export interface SegmentedControlOption<T extends string = string> {
  value: T;
  label: React.ReactNode;
  disabled?: boolean;
}

export interface SegmentedControlProps<T extends string = string> {
  options: SegmentedControlOption<T>[] | T[];
  value: T;
  onChange: (value: T) => void;
  size?: SegmentedControlSize;
  className?: string;
}

const sizeConfig = {
  sm: { padding: '4px 10px', fontSize: '11px' },
  md: { padding: '7px 12px', fontSize: '12px' },
  lg: { padding: '6px 14px', fontSize: '14px' },
};

const Group = styled.div<{ $size: SegmentedControlSize }>`
  display: inline-flex;
  max-width: 100%;
  border: 1px solid ${({ theme, $size }) => $size === 'sm' ? theme.colors.borderStrong : theme.colors.borderControl};
  border-radius: ${({ theme, $size }) => $size === 'sm' ? theme.radii.sm : theme.radii.md};
  overflow-x: auto;
`;

const Segment = styled.button<{
  $active: boolean;
  $size: SegmentedControlSize;
  $disabled?: boolean;
}>`
  all: unset;
  flex-shrink: 0;
  cursor: pointer;
  padding: ${({ $size }) => sizeConfig[$size].padding};
  font-size: ${({ $size }) => sizeConfig[$size].fontSize};
  font-family: ${({ theme }) => theme.fonts.body};
  transition: all 0.15s;
  white-space: nowrap;

  &:focus-visible {
    outline: none;
    box-shadow: inset 0 0 0 2px ${({ theme }) => theme.colors.gold};
  }

  ${({ $active, $size }) =>
    $active
      ? css`
          background: ${({ theme }) => $size === 'sm' ? theme.colors.actionFill : theme.colors.selected};
          color: ${({ theme }) => $size === 'sm' ? theme.colors.textOnAccent : theme.colors.text};
          font-weight: 600;
        `
      : css`
          background: ${({ theme }) => $size === 'sm' ? theme.colors.surfaceSunken : theme.colors.surface};
          color: ${({ theme }) => theme.colors.textSecondary};
          &:hover {
            color: ${({ theme }) => theme.colors.text};
          }
        `}

  ${({ $disabled }) =>
    $disabled &&
    css`
      opacity: 0.35;
      cursor: not-allowed;
      pointer-events: none;
    `}

  /* Subtle divider between inactive segments */
  &:not(:first-child) {
    border-left: 1px solid ${({ theme }) => theme.colors.borderLight};
  }
`;

function normalizeOption<T extends string>(
  opt: SegmentedControlOption<T> | T,
): SegmentedControlOption<T> {
  if (typeof opt === 'string') return { value: opt, label: opt };
  return opt;
}

export function SegmentedControl<T extends string = string>({
  options,
  value,
  onChange,
  size = 'md',
  className,
}: SegmentedControlProps<T>) {
  return (
    <Group $size={size} className={className}>
      {options.map((raw) => {
        const opt = normalizeOption(raw);
        return (
          <Segment
            type="button"
            disabled={opt.disabled}
            key={opt.value}
            $active={value === opt.value}
            $size={size}
            $disabled={opt.disabled}
            onClick={() => !opt.disabled && onChange(opt.value)}
            aria-pressed={value === opt.value}
          >
            {opt.label}
          </Segment>
        );
      })}
    </Group>
  );
}
