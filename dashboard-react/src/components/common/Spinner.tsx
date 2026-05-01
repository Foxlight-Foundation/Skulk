import styled, { keyframes } from 'styled-components';

/**
 * Spinning indicator for in-flight loads. The host component is responsible
 * for placement; pair with `CenteredSpinner` when the loading state should
 * fill the available space and center the indicator.
 */
const spin = keyframes`
  to { transform: rotate(360deg); }
`;

const SpinnerEl = styled.div<{ $size: number }>`
  width: ${({ $size }) => $size}px;
  height: ${({ $size }) => $size}px;
  border: 2px solid ${({ theme }) => theme.colors.borderStrong};
  border-top-color: ${({ theme }) => theme.colors.gold};
  border-radius: 50%;
  animation: ${spin} 0.7s linear infinite;
`;

export interface SpinnerProps {
  /** Diameter in pixels. Defaults to 28px. */
  size?: number;
  className?: string;
}

export function Spinner({ size = 28, className }: SpinnerProps) {
  return <SpinnerEl $size={size} className={className} role="status" aria-label="Loading" />;
}

/**
 * Flex container that fills its parent and centers its content along both
 * axes. Drop a `<Spinner />` inside to get a centered loading indicator with
 * no surrounding box, text, or dressing — the spinner is enough to
 * communicate that something is in flight.
 */
export const CenteredSpinner = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
  min-height: 0;
`;
