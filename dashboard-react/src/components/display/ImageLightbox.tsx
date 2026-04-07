import { useCallback, useEffect } from 'react';
import styled, { keyframes } from 'styled-components';
import { Button } from '../common/Button';

export interface ImageLightboxProps {
  src: string | null;
  onClose: () => void;
}

/* ---- helpers ---- */

function getExtension(src: string): string {
  const dataMatch = src.match(/^data:image\/(\w+)/);
  if (dataMatch) return dataMatch[1];
  const urlMatch = src.match(/\.(\w+)(?:\?|$)/);
  if (urlMatch) return urlMatch[1];
  return 'png';
}

function downloadImage(src: string) {
  const ext = getExtension(src);
  const a = document.createElement('a');
  a.href = src;
  a.download = `image-${Date.now()}.${ext}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

/* ---- styles ---- */

const fadeIn = keyframes`
  from { opacity: 0; }
  to   { opacity: 1; }
`;

const slideUp = keyframes`
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
`;

const Overlay = styled.div`
  position: fixed;
  inset: 0;
  z-index: 50;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.9);
  backdrop-filter: blur(4px);
  animation: ${fadeIn} 0.2s ease-out;
`;

const Image = styled.img`
  max-width: 90vw;
  max-height: 90vh;
  object-fit: contain;
  border-radius: ${({ theme }) => theme.radii.lg};
  box-shadow: 0 25px 50px ${({ theme }) => theme.colors.shadowStrong};
  animation: ${slideUp} 0.3s cubic-bezier(0.33, 1, 0.68, 1);
`;

const ButtonBar = styled.div`
  position: absolute;
  top: 16px;
  right: 16px;
  display: flex;
  gap: 8px;
  z-index: 10;
`;

const IconButton = styled(Button)`
  background: rgba(30, 30, 30, 0.8);
  &:hover:not(:disabled) {
    background: rgba(40, 40, 40, 0.9);
  }
`;

/* ---- component ---- */

export function ImageLightbox({ src, onClose }: ImageLightboxProps) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    },
    [onClose],
  );

  useEffect(() => {
    if (!src) return;
    document.addEventListener('keydown', handleKeyDown);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = prev;
    };
  }, [src, handleKeyDown]);

  if (!src) return null;

  return (
    <Overlay onClick={onClose}>
      <ButtonBar>
        <IconButton
          variant="primary"
          size="lg"
          icon
          onClick={(e) => { e.stopPropagation(); downloadImage(src); }}
          aria-label="Download image"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
        </IconButton>
        <IconButton
          variant="primary"
          size="lg"
          icon
          onClick={(e) => { e.stopPropagation(); onClose(); }}
          aria-label="Close lightbox"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M6 18L18 6M6 6l12 12" />
          </svg>
        </IconButton>
      </ButtonBar>
      <Image
        src={src}
        alt="Full size preview"
        onClick={(e) => e.stopPropagation()}
      />
    </Overlay>
  );
}
