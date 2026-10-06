import { useId, useRef, useState, type KeyboardEvent } from 'react';
import { autoUpdate, flip, offset, shift, useDismiss, useFloating, useInteractions, FloatingFocusManager, FloatingPortal, type Placement } from '@floating-ui/react';
import styled from 'styled-components';
import { ChoiceMenu as Menu, ChoiceOption as Option } from '../common/ChoiceMenu.styles';
import { FiCheck, FiChevronDown } from 'react-icons/fi';
import { MdAutoAwesome } from 'react-icons/md';
import { useSkulkTranslation } from '../../i18n/tolgee';

const Trigger = styled.button<{ $fabric: boolean }>`
  min-width: 0; max-width: 100%; padding: 0; display: inline-flex; align-items: center; gap: 5px;
  border: 0;
  border-radius: 4px; background: transparent;
  color: ${({ theme, $fabric }) => $fabric ? theme.colors.live : theme.colors.text};
  font: 600 13px ${({ theme }) => theme.fonts.body}; text-align: left;
  span { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  svg { flex-shrink: 0; }
`;
const GroupLabel = styled.div`
  padding: 8px 10px 4px; font: 600 10px ${({ theme }) => theme.fonts.mono};
  letter-spacing: .14em; color: ${({ theme }) => theme.colors.textMuted}; text-transform: uppercase;
`;
const ReadyDot = styled.i`width: 6px; height: 6px; border-radius: 50%; background: ${({ theme }) => theme.colors.healthy}; flex-shrink: 0;`;

/** Shared controlled model chooser. Opening and keyboard focus never change or submit the selection. */
export function ReadyModelSelect({ value, onChange, models, fabricEnabled, ariaLabel, className, placement = 'top-start' }: {
  value: string | null;
  onChange: (id: string) => void;
  models: { modelId: string }[];
  fabricEnabled: boolean;
  /** Context-specific accessible name; Chat remains the default. */
  ariaLabel?: string;
  /** Allows the trigger to fit a containing form without changing menu behavior. */
  className?: string;
  /** Prefer opening below form fields; Chat opens above its composer. */
  placement?: Placement;
}) {
  const { t } = useSkulkTranslation();
  const [open, setOpen] = useState(false);
  const [portalRoot, setPortalRoot] = useState<HTMLElement | null>(null);
  const menuId = useId();
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const changeOpen = (next: boolean) => {
    if (next) setPortalRoot(triggerRef.current?.closest<HTMLElement>('[role="dialog"]') ?? null);
    setOpen(next);
  };
  const ids = Array.from(new Set(models.map(model => model.modelId))).filter(id => id !== 'skulk/steward');
  const options = fabricEnabled ? ['skulk/steward', ...ids] : ids;
  const { refs, floatingStyles, context } = useFloating({ open, onOpenChange: changeOpen, placement, strategy: 'fixed', whileElementsMounted: autoUpdate, middleware: [offset(8), flip(), shift({ padding: 12 })] });
  const dismiss = useDismiss(context);
  const { getReferenceProps, getFloatingProps } = useInteractions([dismiss]);
  const selectedIndex = Math.max(0, options.indexOf(value ?? ''));
  const label = ariaLabel ?? t('chat.view.selectModel', 'Select chat model');
  const choose = (id: string) => { onChange(id); setOpen(false); triggerRef.current?.focus(); };
  const navigate = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let next: number;
    switch (event.key) {
      case 'ArrowDown': next = (index + 1) % options.length; break;
      case 'ArrowUp': next = (index - 1 + options.length) % options.length; break;
      case 'Home': next = 0; break;
      case 'End': next = options.length - 1; break;
      default: {
        if (event.key.length !== 1 || event.ctrlKey || event.metaKey || event.altKey) return;
        const match = options.findIndex(id => (id === 'skulk/steward' ? 'Skulk' : id.split('/').pop() ?? id).toLowerCase().startsWith(event.key.toLowerCase()));
        if (match < 0) return;
        next = match;
      }
    }
    event.preventDefault(); optionRefs.current[next]?.focus();
  };
  const option = (id: string) => {
    const fabric = id === 'skulk/steward';
    const index = options.indexOf(id);
    return <Option key={id} type="button" role="option" tabIndex={index === selectedIndex ? 0 : -1} aria-selected={value === id} $fabric={fabric} $selected={value === id}
      ref={element => { optionRefs.current[index] = element; }} onKeyDown={event => navigate(event, index)} onClick={() => choose(id)}>
      {fabric ? <MdAutoAwesome aria-hidden /> : <ReadyDot aria-hidden />}
      <span><strong>{fabric ? 'Skulk' : id.split('/').pop()}</strong><small>{fabric ? t('chat.fabricDescription', 'Asks the cluster itself · prepares actions for approval') : id}</small></span>
      {value === id && <FiCheck aria-hidden />}
    </Option>;
  };
  return <>
    <Trigger className={className} {...getReferenceProps()} onClick={() => changeOpen(!open)} onKeyDown={(event: KeyboardEvent) => { if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); changeOpen(true); } }}
      ref={element => { triggerRef.current = element; refs.setReference(element); }} type="button" $fabric={value === 'skulk/steward'} aria-label={label} aria-haspopup="listbox" aria-expanded={open} aria-controls={open ? menuId : undefined}>
      {value === 'skulk/steward' && <MdAutoAwesome aria-hidden />}<span>{value === 'skulk/steward' ? 'Skulk' : value?.split('/').pop() ?? label}</span><FiChevronDown aria-hidden />
    </Trigger>
    {open && <FloatingPortal root={portalRoot ?? undefined}><FloatingFocusManager context={context} modal={false} initialFocus={0} returnFocus>
      <Menu {...getFloatingProps()} onKeyDown={(event: KeyboardEvent) => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setOpen(false); triggerRef.current?.focus(); } }} ref={element => refs.setFloating(element)} style={floatingStyles} id={menuId} role="listbox" aria-label={label}>
        {fabricEnabled && <div role="group" aria-label={t('chat.fabricGroup', 'THE FABRIC')}><GroupLabel>{t('chat.fabricGroup', 'THE FABRIC')}</GroupLabel>{option('skulk/steward')}</div>}
        <div role="group" aria-label={t('chat.readyGroup', 'READY MODELS')}><GroupLabel>{t('chat.readyGroup', 'READY MODELS')}</GroupLabel>{ids.map(option)}{ids.length === 0 && <GroupLabel>{t('chat.noReadyModels', 'No ready models')}</GroupLabel>}</div>
      </Menu>
    </FloatingFocusManager></FloatingPortal>}
  </>;
}
