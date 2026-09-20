import { useLayoutEffect, useRef, type RefObject } from 'react';

let activeModal: { identity: symbol; close: () => void; element: HTMLElement; returnFocus: Element | null; bodyOverflow: string } | null = null;

/** Own modal focus, dismissal and focus return while allowing only one active surface. */
export function useModalFocus(open: boolean, elementRef: RefObject<HTMLElement | null>, onClose: () => void): void {
  const identity = useRef(Symbol('modal'));
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useLayoutEffect(() => {
    if (!open) return;
    const element = elementRef.current;
    if (!element) return;
    const owner = identity.current;
    const previous = activeModal;
    const previousFocus = previous?.element.contains(document.activeElement) ? previous.returnFocus : document.activeElement;
    const bodyOverflow = previous?.bodyOverflow ?? document.body.style.overflow;
    if (previous && previous.identity !== owner) previous.close();
    activeModal = { identity: owner, close: () => closeRef.current(), element, returnFocus: previousFocus, bodyOverflow };
    document.body.style.overflow = 'hidden';
    element.focus();
    const keydown = (event: KeyboardEvent) => {
      if (activeModal?.identity !== owner) return;
      if (event.key === 'Escape' && !event.defaultPrevented) { event.preventDefault(); closeRef.current(); return; }
      if (event.key !== 'Tab') return;
      const controls = Array.from(element.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href], summary, [tabindex="0"]')).filter(control => control.tabIndex >= 0 && control.getClientRects().length > 0);
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (!first || !last) { event.preventDefault(); element.focus(); return; }
      if (event.shiftKey && (document.activeElement === first || document.activeElement === element)) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && (document.activeElement === last || document.activeElement === element)) { event.preventDefault(); first.focus(); }
    };
    const containFocus = (event: FocusEvent) => {
      if (activeModal?.identity === owner && event.target instanceof Node && !element.contains(event.target)) element.focus();
    };
    document.addEventListener('focusin', containFocus);
    window.addEventListener('keydown', keydown);
    return () => {
      window.removeEventListener('keydown', keydown);
      document.removeEventListener('focusin', containFocus);
      if (activeModal?.identity !== owner) return;
      activeModal = null;
      document.body.style.overflow = bodyOverflow;
      if (previousFocus instanceof HTMLElement && previousFocus.isConnected) {
        // Focus-to-open prompts must distinguish restoration from a new visit.
        previousFocus.dataset.skulkFocusReturn = 'true';
        previousFocus.focus();
        delete previousFocus.dataset.skulkFocusReturn;
      }
    };
  }, [open, elementRef]);
}
