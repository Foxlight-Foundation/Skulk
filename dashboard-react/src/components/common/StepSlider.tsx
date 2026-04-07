import styled from 'styled-components';

export interface StepSliderOption<T extends string | number = number> {
  value: T;
  label: string;
}

export interface StepSliderProps<T extends string | number = number> {
  options: StepSliderOption<T>[] | T[];
  value: T;
  onChange: (value: T) => void;
  /** Track line thickness in px. Default 2. */
  trackWidth?: number;
  /** Active dot diameter in px. Default 18. */
  dotSize?: number;
  /** Inactive dot diameter in px. Default 14. */
  inactiveDotSize?: number;
  className?: string;
}

/* ---- styles ---- */

const Container = styled.div`
  position: relative;
  user-select: none;
`;

const TrackLine = styled.div<{ $inset: number; $height: number; $dotCenter: number }>`
  position: absolute;
  top: ${({ $dotCenter }) => $dotCenter}px;
  left: ${({ $inset }) => $inset}px;
  right: ${({ $inset }) => $inset}px;
  height: ${({ $height }) => $height}px;
  margin-top: ${({ $height }) => -$height / 2}px;
  background: ${({ theme }) => theme.colors.borderStrong};
  border-radius: ${({ $height }) => $height / 2}px;
`;

const ActiveLine = styled.div<{ $inset: number; $pct: number; $height: number; $dotCenter: number }>`
  position: absolute;
  top: ${({ $dotCenter }) => $dotCenter}px;
  left: ${({ $inset }) => $inset}px;
  height: ${({ $height }) => $height}px;
  margin-top: ${({ $height }) => -$height / 2}px;
  width: calc((100% - ${({ $inset }) => $inset * 2}px) * ${({ $pct }) => $pct / 100});
  background: ${({ theme }) => theme.colors.gold};
  border-radius: ${({ $height }) => $height / 2}px;
  transition: width 0.15s ease-out;
`;

const Steps = styled.div`
  position: relative;
  display: flex;
  justify-content: space-between;
`;

const StepCol = styled.button`
  all: unset;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  z-index: 1;
`;

const Dot = styled.span<{ $active: boolean; $size: number; $inactiveSize: number }>`
  width: ${({ $active, $size, $inactiveSize }) => ($active ? $size : $inactiveSize)}px;
  height: ${({ $active, $size, $inactiveSize }) => ($active ? $size : $inactiveSize)}px;
  margin: ${({ $active, $size, $inactiveSize }) => {
    const maxSize = $size;
    const thisSize = $active ? $size : $inactiveSize;
    const pad = (maxSize - thisSize) / 2;
    return `${pad}px 0`;
  }};
  border-radius: 50%;
  background: ${({ $active, theme }) => ($active ? theme.colors.gold : theme.colors.borderStrong)};
  transition: all 0.15s ease-out;
  box-shadow: ${({ $active, theme }) => ($active ? `0 0 8px ${theme.colors.goldDim}` : 'none')};

  ${StepCol}:hover & {
    background: ${({ $active, theme }) => ($active ? theme.colors.gold : theme.colors.textMuted)};
  }
`;

const Label = styled.span<{ $active: boolean }>`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ $active, theme }) => ($active ? theme.colors.gold : theme.colors.textMuted)};
  font-weight: ${({ $active }) => ($active ? 600 : 400)};
  transition: color 0.15s;
`;

/* ---- component ---- */

function normalizeOption<T extends string | number>(
  opt: StepSliderOption<T> | T,
): StepSliderOption<T> {
  if (typeof opt === 'object' && opt !== null && 'value' in opt) return opt;
  return { value: opt as T, label: String(opt) };
}

export function StepSlider<T extends string | number = number>({
  options: rawOptions,
  value,
  onChange,
  trackWidth = 2,
  dotSize = 18,
  inactiveDotSize = 14,
  className,
}: StepSliderProps<T>) {
  const options = rawOptions.map(normalizeOption);
  const activeIndex = options.findIndex((o) => o.value === value);
  const pct = options.length <= 1 ? 0 : (activeIndex / (options.length - 1)) * 100;

  // Inset track lines to start/end at dot centers.
  const trackInset = dotSize / 2;
  const dotCenter = dotSize / 2;

  return (
    <Container className={className}>
      <TrackLine $inset={trackInset} $height={trackWidth} $dotCenter={dotCenter} />
      <ActiveLine $inset={trackInset} $pct={pct} $height={trackWidth} $dotCenter={dotCenter} />
      <Steps>
        {options.map((opt, i) => (
          <StepCol
            key={String(opt.value)}
            onClick={() => onChange(opt.value)}
            aria-label={opt.label}
          >
            <Dot $active={i === activeIndex} $size={dotSize} $inactiveSize={inactiveDotSize} />
            <Label $active={i === activeIndex}>{opt.label}</Label>
          </StepCol>
        ))}
      </Steps>
    </Container>
  );
}
