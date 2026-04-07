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
  sm: { padding: '3px 8px', fontSize: '10px' },
  md: { padding: '4px 10px', fontSize: '11px' },
  lg: { padding: '6px 14px', fontSize: '12px' },
};

const Group = styled.div<{ $size: SegmentedControlSize }>`
  display: inline-flex;
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
  border-radius: ${({ theme }) => theme.radii.sm};
  overflow: hidden;
`;

const Segment = styled.button<{
  $active: boolean;
  $size: SegmentedControlSize;
  $disabled?: boolean;
}>`
  all: unset;
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

  ${({ $active }) =>
    $active
      ? css`
          background: ${({ theme }) => theme.colors.gold};
          color: ${({ theme }) => theme.colors.textOnAccent};
          font-weight: 600;
        `
      : css`
          background: ${({ theme }) => theme.colors.surfaceSunken};
          color: ${({ theme }) => theme.colors.textMuted};
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
    border-left: 1px solid ${({ theme }) => theme.colors.goldBg};
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
