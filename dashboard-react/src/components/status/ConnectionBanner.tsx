import styled, { keyframes } from 'styled-components';

export interface ConnectionBannerProps {
  connected: boolean;
}

const pulse = keyframes`
  0%, 100% { opacity: 1; }
  50%      { opacity: 0.4; }
`;

const Banner = styled.div`
  position: relative;
  z-index: 50;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 8px 16px;
  background: rgba(69, 10, 10, 0.8);
  border-bottom: 1px solid ${({ theme }) => theme.colors.errorBg};
`;

const Dot = styled.div`
  width: 8px;
  height: 8px;
  background: ${({ theme }) => theme.colors.error};
  border-radius: 50%;
  animation: ${pulse} 1.5s ease-in-out infinite;
`;

const Text = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.errorText};
`;

export function ConnectionBanner({ connected }: ConnectionBannerProps) {
  if (connected) return null;

  return (
    <Banner role="alert" aria-live="assertive">
      <Dot />
      <Text>Connection lost — Reconnecting to backend…</Text>
    </Banner>
  );
}
