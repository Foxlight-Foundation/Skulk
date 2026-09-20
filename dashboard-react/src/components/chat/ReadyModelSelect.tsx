import { useId, useRef, useState, type KeyboardEvent } from 'react';
import { autoUpdate, flip, offset, shift, useDismiss, useFloating, useInteractions, FloatingFocusManager, FloatingPortal } from '@floating-ui/react';
import styled from 'styled-components';
import { FiCheck, FiChevronDown } from 'react-icons/fi';
import { MdAutoAwesome } from 'react-icons/md';
import { useSkulkTranslation } from '../../i18n/tolgee';

const Trigger = styled.button<{ $fabric: boolean }>`
  min-width: 0; max-width: 100%; padding: 8px 12px; display: inline-flex; align-items: center; gap: 8px;
  border: 1px solid ${({ theme, $fabric }) => $fabric ? theme.colors.borderLive : theme.colors.borderControl};
  border-radius: 8px; background: ${({ theme, $fabric }) => $fabric ? theme.colors.liveBg : theme.colors.surface};
  color: ${({ theme, $fabric }) => $fabric ? theme.colors.live : theme.colors.text};
  font: 14px ${({ theme }) => theme.fonts.body}; text-align: left;
  span { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  svg { flex-shrink: 0; }
`;
const Menu = styled.div`
  z-index: 100; width: min(360px, calc(100vw - 24px)); max-height: min(420px, calc(100dvh - 80px)); overflow-y: auto;
  background: ${({ theme }) => theme.colors.surfaceElevated}; border: 1px solid ${({ theme }) => theme.colors.borderControl};
  border-radius: 12px; padding: 6px; box-shadow: ${({ theme }) => theme.colors.shadowPop};
`;
const GroupLabel = styled.div`
  padding: 8px 10px 6px; font: 600 10px ${({ theme }) => theme.fonts.mono};
  letter-spacing: .14em; color: ${({ theme }) => theme.colors.textMuted}; text-transform: uppercase;
`;
const Option = styled.button<{ $fabric: boolean; $selected: boolean }>`
  display: flex; align-items: center; gap: 10px; width: 100%; padding: 10px;
  border: 1px solid ${({ theme, $fabric, $selected }) => $selected ? ($fabric ? theme.colors.borderLive : theme.colors.borderStrong) : 'transparent'};
  background: ${({ theme, $fabric, $selected }) => $selected ? ($fabric ? theme.colors.liveBg : theme.colors.goldBg) : 'transparent'};
  border-radius: 8px; text-align: left; color: ${({ theme }) => theme.colors.text};
  font: 14px ${({ theme }) => theme.fonts.body};
  > svg { flex-shrink: 0; color: ${({ theme, $fabric }) => $fabric ? theme.colors.live : theme.colors.gold}; }
  > span { min-width: 0; flex: 1; overflow-wrap: anywhere; }
  strong { font-weight: 600; } small { display: block; margin-top: 3px; font-size: 11.5px; line-height: 1.4; color: ${({ theme }) => theme.colors.textSecondary}; }
  &:hover { background: ${({ theme }) => theme.colors.surfaceSunken}; }
  &:focus-visible { outline: none; box-shadow: inset 0 0 0 2px ${({ theme }) => theme.colors.gold}; }
`;
const ReadyDot = styled.i`width: 6px; height: 6px; border-radius: 50%; background: ${({ theme }) => theme.colors.healthy}; flex-shrink: 0;`;

/** Shared controlled model chooser. Opening and keyboard focus never change or submit the selection. */
export function ReadyModelSelect({ value, onChange, models, fabricEnabled }: {
  value: string | null;
  onChange: (id: string) => void;
  models: { modelId: string }[];
  fabricEnabled: boolean;
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
  const { refs, floatingStyles, context } = useFloating({ open, onOpenChange: changeOpen, placement: 'top-start', strategy: 'fixed', whileElementsMounted: autoUpdate, middleware: [offset(8), flip(), shift({ padding: 12 })] });
  const dismiss = useDismiss(context);
  const { getReferenceProps, getFloatingProps } = useInteractions([dismiss]);
  const selectedIndex = Math.max(0, options.indexOf(value ?? ''));
  const label = t('chat.view.selectModel', 'Select chat model');
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
    <Trigger {...getReferenceProps()} onClick={() => changeOpen(!open)} onKeyDown={(event: KeyboardEvent) => { if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); changeOpen(true); } }}
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
