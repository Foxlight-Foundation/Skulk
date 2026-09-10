import { useCallback, useEffect, useRef, type ReactNode } from 'react';
import styled, { keyframes } from 'styled-components';
import { FiX } from 'react-icons/fi';
import { MOBILE_BREAKPOINT_PX } from '../../hooks/useMediaQuery';
import { Button } from './Button';
import { useSkulkTranslation } from '../../i18n/tolgee';

/** Props for the shared right-hand drawer chrome. */
export interface RightDrawerProps {
  /** Whether the drawer is shown; nothing renders while false. */
  open: boolean;
  /** Closes the drawer (backdrop click, close button, Escape). */
  onClose: () => void;
  /** Heading shown in the drawer header. */
  title: ReactNode;
  /** Accessible name of the drawer landmark. */
  ariaLabel: string;
  /** DOM id for the drawer element, when other components target it. */
  id?: string;
  /** Current width in pixels. */
  width: number;
  /** Lower bound the drag handle honors. */
  minWidth: number;
  /** Upper bound the drag handle honors. */
  maxWidth: number;
  /** Commits a new width after a drag ends. */
  onWidthChange: (width: number) => void;
  /** Accessible names for the resize handle and close button. */
  resizeLabel: string;
  closeLabel: string;
  /** Drawer content below the header (tabs, body). */
  children: ReactNode;
}

const fadeIn = keyframes`
  from { opacity: 0; }
  to   { opacity: 1; }
`;

const slideIn = keyframes`
  from { transform: translateX(100%); }
  to   { transform: translateX(0); }
`;

/**
 * Click-to-close dim + blur layer behind the drawer. Matches `SettingsPanel`
 * (z=40 backdrop / z=50 drawer) so the visual treatment is consistent across
 * modal-style panels in the dashboard.
 */
const Backdrop = styled.div`
  position: fixed;
  inset: 0;
  z-index: 40;
  background: ${({ theme }) => theme.colors.shadowStrong};
  backdrop-filter: blur(2px);
  animation: ${fadeIn} 0.2s ease-out;
`;

const Aside = styled.aside<{ $width: number }>`
  position: fixed;
  top: 0;
  right: 0;
  /* dvh tracks the ACTUAL visible viewport on mobile browsers; 100vh
   * includes the area behind Safari's URL/tool bars, which pushed the
   * panel's bottom off screen. Plain vh stays as the fallback for engines
   * without dvh. */
  height: 100vh;
  height: 100dvh;
  width: ${({ $width }) => $width}px;
  background: ${({ theme }) => theme.colors.surfaceElevated};
  border-left: 1px solid ${({ theme }) => theme.colors.borderStrong};
  box-shadow: -18px 0 48px ${({ theme }) => theme.colors.shadowStrong};
  display: flex;
  flex-direction: column;
  z-index: 50;
  animation: ${slideIn} 0.25s cubic-bezier(0.33, 1, 0.68, 1);

  /* Phone width: the drawer becomes a full-viewport sheet. The persisted
   * desktop width (which can exceed a phone viewport) is ignored; !important
   * beats the drag-resize inline style if one was left behind by a desktop
   * session. */
  @media (max-width: ${MOBILE_BREAKPOINT_PX}px) {
    width: 100vw !important;
    border-left: none;
  }
`;

/**
 * Drag handle on the left edge of the drawer. Sits in front of `Aside`'s
 * left border with a slightly wider hit area than its visible footprint so
 * the cursor catches it reliably.
 */
const ResizeHandle = styled.div`
  position: absolute;
  left: -4px;
  top: 0;
  bottom: 0;
  width: 8px;
  cursor: ew-resize;
  &:hover {
    background: ${({ theme }) => theme.colors.goldDim};
    opacity: 0.4;
  }

  /* No drag-resize on a full-viewport phone sheet. */
  @media (max-width: ${MOBILE_BREAKPOINT_PX}px) {
    display: none;
  }
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px 10px;
  gap: 12px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
`;

const Title = styled.h2`
  margin: 0;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.lg};
  color: ${({ theme }) => theme.colors.text};
`;

/**
 * Resizable right-hand drawer shared by the observability and capability
 * panels: dimmed backdrop, slide-in aside, drag handle on the left edge,
 * header with a close button, and Escape to close. Width is owned by the
 * caller so each panel can persist its own; live drag previews write the
 * width straight to the DOM and commit once on pointer up.
 */
export function RightDrawer({
  open,
  onClose,
  title,
  ariaLabel,
  id,
  width,
  minWidth,
  maxWidth,
  onWidthChange,
  resizeLabel,
  closeLabel,
  children,
}: RightDrawerProps) {
  const { t } = useSkulkTranslation();
  // Drag-to-resize: capture pointer at the handle; resizing computes width
  // from the cursor's distance to the right edge of the viewport. The aside
  // ref keeps the hot path free of DOM lookups and the contract local.
  const asideRef = useRef<HTMLElement | null>(null);
  const draggingRef = useRef(false);
  const dragWidthRef = useRef<number>(width);

  const onResizeStart = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      event.preventDefault();
      draggingRef.current = true;
      dragWidthRef.current = width;
      (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    },
    [width],
  );

  useEffect(() => {
    if (!open) return;
    const onMove = (event: PointerEvent) => {
      if (!draggingRef.current) return;
      // Clamp during the live preview too, so a pointer leaving the viewport
      // never flickers the drawer off-screen before the commit clamps.
      const raw = window.innerWidth - event.clientX;
      const next = Math.max(minWidth, Math.min(maxWidth, raw));
      dragWidthRef.current = next;
      if (asideRef.current) asideRef.current.style.width = `${next}px`;
    };
    const onUp = () => {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      // Clear the inline style BEFORE committing: inline widths beat the
      // styled-component rule on specificity and would override every later
      // state-driven width change.
      if (asideRef.current) asideRef.current.style.width = '';
      onWidthChange(dragWidthRef.current);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
    };
  }, [open, minWidth, maxWidth, onWidthChange]);

  // Esc closes the drawer; operators expect this for any modal-like surface.
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <>
      <Backdrop data-testid="right-drawer-backdrop" onClick={onClose} />
      <Aside $width={width} ref={asideRef} id={id} aria-label={ariaLabel}>
        <ResizeHandle
          onPointerDown={onResizeStart}
          role="separator"
          aria-orientation="vertical"
          aria-label={resizeLabel}
        />
        <Header>
          <Title>{title}</Title>
          <Button variant="ghost" size="sm" onClick={onClose} aria-label={closeLabel || t('common.close', 'Close')}>
            <FiX size={16} />
          </Button>
        </Header>
        {children}
      </Aside>
    </>
  );
}
