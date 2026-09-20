import { Children, isValidElement, useId, useRef, useState, type ButtonHTMLAttributes, type ReactNode } from 'react';
import { autoUpdate, flip, offset, shift, size, useDismiss, useFloating, useInteractions, useListNavigation, useTypeahead, FloatingFocusManager, FloatingPortal } from '@floating-ui/react';
import { FiCheck, FiChevronDown } from 'react-icons/fi';
import styled from 'styled-components';
import { ChoiceMenu, ChoiceOption } from './ChoiceMenu.styles';

const Trigger = styled.button`
  display: inline-flex; align-items: center; justify-content: space-between; gap: 8px;
  min-width: 0; max-width: 100%; min-height: 34px; padding: 6px 10px;
  background: ${({ theme }) => theme.colors.surface}; color: ${({ theme }) => theme.colors.text};
  border: 1px solid ${({ theme }) => theme.colors.borderControl}; border-radius: 8px;
  font: 13px ${({ theme }) => theme.fonts.body}; text-align: left; cursor: pointer;
  > span { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  > svg { flex-shrink: 0; }
  &:focus-visible { outline: none; box-shadow: ${({ theme }) => theme.colors.focusRing}; }
  &:disabled { opacity: .5; cursor: not-allowed; }
`;

/** Controlled, single-value selector using Chat's menu presentation and declarative option children. */
export interface SelectProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'value' | 'onChange'> {
  value?: string | number;
  onValueChange?: (value: string) => void;
  required?: boolean;
}

function optionText(node: ReactNode): string {
  return Children.toArray(node).map(child => isValidElement<{ children?: ReactNode }>(child) ? optionText(child.props.children) : String(child)).join('');
}

/** Render a themed listbox, keeping focus and Escape dismissal local to an enclosing drawer. */
export function Select({ value, onValueChange, children, required, disabled, id, name, ...props }: SelectProps) {
  const generatedId = useId();
  const triggerId = id ?? generatedId;
  const menuId = `${triggerId}-options`;
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const listRef = useRef<Array<HTMLElement | null>>([]);
  const textRef = useRef<Array<string | null>>([]);
  const [localValue, setLocalValue] = useState<string | undefined>();
  const effectiveValue = value ?? localValue;
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [portalRoot, setPortalRoot] = useState<HTMLElement | null>(null);
  const options = Children.toArray(children).flatMap(child => {
    if (!isValidElement<{ value?: string | number; children?: ReactNode; disabled?: boolean }>(child)) return [];
    return [{ value: String(child.props.value ?? optionText(child.props.children)), label: child.props.children, disabled: child.props.disabled }];
  });
  const selectedIndex = options.findIndex(option => option.value === String(effectiveValue ?? ''));
  const changeOpen = (next: boolean) => {
    if (next && (disabled || !options.length)) return;
    if (next) setPortalRoot(triggerRef.current?.closest<HTMLElement>('[role="dialog"]') ?? null);
    setOpen(next);
  };
  const { refs, floatingStyles, context } = useFloating({
    open: open && !disabled, onOpenChange: changeOpen, placement: 'bottom-start', strategy: 'fixed', whileElementsMounted: autoUpdate,
    middleware: [offset(8), flip(), shift({ padding: 12 }), size({ padding: 12, apply({ availableHeight, elements }) {
      elements.floating.style.maxHeight = `${Math.max(0, Math.min(420, availableHeight))}px`;
    } })],
  });
  const dismiss = useDismiss(context, { bubbles: { escapeKey: false } });
  const navigation = useListNavigation(context, { listRef, activeIndex, selectedIndex: selectedIndex < 0 ? null : selectedIndex, onNavigate: setActiveIndex, loop: true, disabledIndices: options.flatMap((option, index) => option.disabled ? [index] : []) });
  const typeahead = useTypeahead(context, { listRef: textRef, activeIndex, onMatch: setActiveIndex, enabled: open });
  const { getReferenceProps, getFloatingProps, getItemProps } = useInteractions([dismiss, navigation, typeahead]);
  return <>
    <Trigger {...props} {...getReferenceProps()} onClick={() => changeOpen(!open)} id={triggerId} type="button" disabled={disabled}
      ref={element => { triggerRef.current = element; refs.setReference(element); }}
      aria-haspopup="listbox" aria-expanded={open && !disabled} aria-controls={open && !disabled ? menuId : undefined}>
      <span>{options[selectedIndex]?.label ?? options[0]?.label}</span><FiChevronDown aria-hidden />
    </Trigger>
    {/* Keep required-field validation and form submission without exposing a native select. */}
    {(required || name) && <input tabIndex={-1} aria-hidden name={name} value={effectiveValue ?? ''} disabled={disabled} required={required} onChange={() => {}}
      style={{ position: 'absolute', width: 1, height: 1, opacity: 0, pointerEvents: 'none' }}
      onInvalid={event => { event.preventDefault(); triggerRef.current?.focus(); changeOpen(true); }} />}
    {open && !disabled && <FloatingPortal root={portalRoot ?? undefined}><FloatingFocusManager context={context} modal={false} initialFocus={-1} returnFocus>
      <ChoiceMenu {...getFloatingProps()} ref={element => refs.setFloating(element)} style={floatingStyles} id={menuId} role="listbox" aria-labelledby={triggerId} aria-required={required}>
        {options.map((option, index) => <ChoiceOption key={option.value} {...getItemProps()} onClick={() => { setLocalValue(option.value); onValueChange?.(option.value); setOpen(false); triggerRef.current?.focus(); }}
          ref={element => { listRef.current[index] = element; textRef.current[index] = option.disabled ? null : optionText(option.label); }}
          type="button" role="option" disabled={option.disabled} tabIndex={activeIndex === index ? 0 : -1}
          aria-selected={index === selectedIndex} $selected={index === selectedIndex} $fabric={false}>
          <span>{option.label}</span>{index === selectedIndex && <FiCheck aria-hidden />}
        </ChoiceOption>)}
      </ChoiceMenu>
    </FloatingFocusManager></FloatingPortal>}
  </>;
}
