/**
 * ImageLightbox
 *
 * Full-screen image preview rendered via React portal.
 * Ported from ImageLightbox.svelte.
 */
import React, { useEffect } from 'react';
import { createPortal } from 'react-dom';
import styled from 'styled-components';
import { useTranslate } from '@tolgee/react';

// ─── Styled components ────────────────────────────────────────────────────────

const Backdrop = styled.div`
  position: fixed;
  inset: 0;
  z-index: 9990;
  background: oklch(0 0 0 / 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: zoom-out;
`;

const Img = styled.img`
  max-width: 90vw;
  max-height: 90vh;
  object-fit: contain;
  border-radius: ${({ theme }) => theme.radius.md};
  box-shadow: 0 0 60px oklch(0 0 0 / 0.8);
  cursor: default;
`;

const CloseButton = styled.button`
  position: fixed;
  top: 20px;
  right: 20px;
  background: ${({ theme }) => theme.colors.darkGray};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radius.full};
  color: ${({ theme }) => theme.colors.foreground};
  width: 36px;
  height: 36px;
  font-size: 18px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background ${({ theme }) => theme.transitions.fast};
  &:hover { background: ${({ theme }) => theme.colors.mediumGray}; }
`;

// ─── Component ─────────────────────────────────────────────────────────────────

export interface ImageLightboxProps {
  src: string;
  alt?: string;
  onClose: () => void;
}

export const ImageLightbox: React.FC<ImageLightboxProps> = ({ src, alt = '', onClose }) => {
  const { t } = useTranslate();

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  return createPortal(
    <Backdrop onClick={onClose} role="dialog" aria-modal aria-label="Image preview">
      <Img
        src={src}
        alt={alt}
        onClick={(e) => e.stopPropagation()}
      />
      <CloseButton onClick={onClose} aria-label={t('common.close')}>✕</CloseButton>
    </Backdrop>,
    document.body,
  );
};
