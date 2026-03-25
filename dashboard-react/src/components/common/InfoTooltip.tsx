import { useState, useRef } from 'react';
import {
  useFloating,
  autoUpdate,
  offset,
  flip,
  shift,
  arrow,
  useHover,
  useFocus,
  useDismiss,
  useRole,
  useInteractions,
  FloatingPortal,
  FloatingArrow,
  type Placement,
} from '@floating-ui/react';
import styled from 'styled-components';

export interface InfoTooltipProps {
  /** Tooltip content — string or JSX. */
  content: React.ReactNode;
  /** Preferred placement. Default 'top'. */
  placement?: Placement;
  /** Hover delay in ms before showing. Default 200. */
  delay?: number;
  /** Fill the default info icon. Default false (outline only). */
  filled?: boolean;
  /** Custom trigger element. Defaults to info icon. */
  children?: React.ReactNode;
  className?: string;
}

const Trigger = styled.span`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: help;
  color: ${({ theme }) => theme.colors.textMuted};
  transition: color 0.15s;

  &:hover {
    color: ${({ theme }) => theme.colors.textSecondary};
  }
`;

const TooltipBox = styled.div`
  max-width: 280px;
  padding: 8px 12px;
  background: #1a1a1a;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: ${({ theme }) => theme.radii.md};
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
  font-size: 12px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.textSecondary};
  line-height: 1.5;
  z-index: 9999;
`;

const ARROW_SIZE = 8;

const InfoIcon = ({ filled = false, size = 16 }: { filled?: boolean; size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={filled ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <path d="M12 16v-4" stroke={filled ? '#000' : 'currentColor'} />
    <path d="M12 8h.01" stroke={filled ? '#000' : 'currentColor'} />
  </svg>
);

export function InfoTooltip({
  content,
  placement = 'top',
  delay = 200,
  filled = false,
  children,
  className,
}: InfoTooltipProps) {
  const [isOpen, setIsOpen] = useState(false);
  const arrowRef = useRef(null);

  const { refs, floatingStyles, context } = useFloating({
    open: isOpen,
    onOpenChange: setIsOpen,
    placement,
    whileElementsMounted: autoUpdate,
    middleware: [
      offset(ARROW_SIZE + 4),
      flip(),
      shift({ padding: 8 }),
      arrow({ element: arrowRef }),
    ],
  });

  const hover = useHover(context, { delay: { open: delay, close: 0 } });
  const focus = useFocus(context);
  const dismiss = useDismiss(context);
  const role = useRole(context, { role: 'tooltip' });

  const { getReferenceProps, getFloatingProps } = useInteractions([
    hover,
    focus,
    dismiss,
    role,
  ]);

  return (
    <>
      <Trigger
        ref={refs.setReference}
        {...getReferenceProps()}
        className={className}
        tabIndex={0}
      >
        {children ?? <InfoIcon filled={filled} />}
      </Trigger>

      {isOpen && (
        <FloatingPortal>
          <TooltipBox
            ref={refs.setFloating}
            style={floatingStyles}
            {...getFloatingProps()}
          >
            <FloatingArrow
              ref={arrowRef}
              context={context}
              width={ARROW_SIZE * 2}
              height={ARROW_SIZE}
              fill="#1a1a1a"
              stroke="rgba(255, 255, 255, 0.1)"
              strokeWidth={1}
            />
            {content}
          </TooltipBox>
        </FloatingPortal>
      )}
    </>
  );
}
