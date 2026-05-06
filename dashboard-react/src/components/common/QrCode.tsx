import { QRCodeSVG } from 'qrcode.react';
import styled from 'styled-components';

type QrCodeLevel = 'L' | 'M' | 'Q' | 'H';

interface QrCodeProps {
  value: string;
  alt: string;
  size?: number;
  level?: QrCodeLevel;
  logoSrc?: string;
  logoSize?: number;
  marginSize?: number;
  className?: string;
}

const Frame = styled.div`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: #ffffff;
  padding: 10px;
  line-height: 0;
`;

export function QrCode({
  value,
  alt,
  size = 220,
  level = 'H',
  logoSrc,
  logoSize,
  marginSize = 1,
  className,
}: QrCodeProps) {
  const resolvedLogoSize = logoSize ?? Math.round(size * 0.2);

  return (
    <Frame className={className}>
      <QRCodeSVG
        value={value}
        size={size}
        level={level}
        marginSize={marginSize}
        role="img"
        aria-label={alt}
        imageSettings={logoSrc
          ? {
              src: logoSrc,
              height: resolvedLogoSize,
              width: resolvedLogoSize,
              excavate: true,
            }
          : undefined}
      />
    </Frame>
  );
}
