import { useCallback, useEffect, useRef, useState } from 'react';
import styled from 'styled-components';
import { FiMoreHorizontal, FiPackage } from 'react-icons/fi';
import { Button } from './Button';
import { Monogram, StatusPill, type StatusTone } from './Surfaces';
import { useSkulkTranslation } from '../../i18n/tolgee';

/** An existing plugin operation with its caller-owned permission and request fences. */
export interface PluginCardAction {
  id: string;
  label: string;
  onSelect: () => void;
  disabled?: boolean;
  danger?: boolean;
  separatorBefore?: boolean;
}
/** Display metadata and observed health for one plugin, independent of its operations. */
export interface PluginSummaryCardProps {
  name: string;
  pluginId?: string;
  description?: string;
  health: string;
  tone: StatusTone;
  release: string;
  releaseNote?: string;
  nodes: string[];
  muted?: boolean;
  actions?: PluginCardAction[];
  primaryLabel?: string;
  onOpen: () => void;
}
const Card = styled.article<{ $muted: boolean }>`
  padding: 16px 18px; margin: 10px 0; border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 14px;
  background: ${({ theme }) => theme.colors.surface}; display: grid; grid-template-columns: 44px minmax(0, 1fr) auto auto; gap: 16px; align-items: center;
  color: ${({ theme }) => theme.colors.text};
  > :not(:last-child) { opacity: ${({ $muted }) => $muted ? .65 : 1}; }
  h2 { font-size: 16px; font-weight: 600; margin: 0; overflow-wrap: anywhere; }
  @container(max-width: 700px) { grid-template-columns: 44px minmax(0, 1fr) auto; align-items: start; > div:nth-child(3) { grid-column: 2 / -1; grid-row: 2; } > div:last-child { grid-column: 3; grid-row: 1; } }
  @media(max-width: 600px) { grid-template-columns: 44px minmax(0, 1fr) auto; align-items: start; gap: 12px; padding: 16px; > div:nth-child(3) { grid-column: 2 / -1; grid-row: 2; } > div:last-child { grid-column: 3; grid-row: 1; } }
`;
const Mark = styled(Monogram)`border-radius: 12px; border: 1px solid ${({ theme }) => theme.colors.border}; font-size: 14px;`;
const Description = styled.p`margin: 3px 0 0; color: ${({ theme }) => theme.colors.textSecondary}; font-size: 13px; line-height: 1.45;`;
const Metadata = styled.div`
  display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 8px; font: 11.5px ${({ theme }) => theme.fonts.mono}; color: ${({ theme }) => theme.colors.metadataText};
  > span { overflow-wrap: anywhere; min-width: 0; } b { font-weight: 400; color: ${({ theme }) => theme.colors.textSecondary}; }
`;
const Row = styled.div`display: flex; align-items: center; flex-wrap: wrap; gap: 10px; min-width: 0;`;
const Chip = styled.span`padding: 3px 8px; border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 6px; background: ${({ theme }) => theme.colors.surfaceHover}; font: 12px ${({ theme }) => theme.fonts.mono}; overflow-wrap: anywhere;`;
const Nodes = styled(Row)`max-width: 200px; gap: 6px; min-width: 0;`;
const Actions = styled.div`display: flex; align-items: center; gap: 6px;`;
const Primary = styled(Button)`@media(max-width: 600px) { display: none; } @container(max-width: 700px) { display: none; }`;
const Menu = styled.details<{ $above: boolean; $availableHeight: number }>`
  position: relative;
  summary { display: flex; align-items: center; justify-content: center; width: 32px; height: 32px; border: 1px solid transparent; border-radius: 8px; cursor: pointer; list-style: none; color: ${({ theme }) => theme.colors.textSecondary}; }
  summary::-webkit-details-marker { display: none; }
  &[open] > summary, summary:hover { border-color: ${({ theme }) => theme.colors.borderStrong}; background: ${({ theme }) => theme.colors.surfaceHover}; }
  > div { position: absolute; right: 0; top: ${({ $above }) => $above ? 'auto' : '38px'}; bottom: ${({ $above }) => $above ? '38px' : 'auto'}; max-height: ${({ $availableHeight }) => $availableHeight}px; overflow-y: auto; z-index: 5; width: 240px; max-width: calc(100vw - 48px); background: ${({ theme }) => theme.colors.surfaceElevated}; border: 1px solid ${({ theme }) => theme.colors.borderControl}; padding: 6px; border-radius: 10px; box-shadow: ${({ theme }) => theme.colors.shadowPop}; }
  hr { border: 0; border-top: 1px solid ${({ theme }) => theme.colors.border}; margin: 4px 6px; }
  p { padding: 6px 10px 4px; font-size: 11px; line-height: 1.4; color: ${({ theme }) => theme.colors.metadataText}; }
`;
const MenuAction = styled.button<{ $danger?: boolean }>`
  display: block; width: 100%; text-align: left; padding: 8px 10px; border-radius: 6px; font: 13.5px ${({ theme }) => theme.fonts.body};
  color: ${({ theme, $danger }) => $danger ? theme.colors.error : theme.colors.text}; overflow-wrap: break-word;
  &:hover:not(:disabled) { background: ${({ theme }) => theme.colors.surfaceHover}; }
  &:disabled { opacity: .45; cursor: not-allowed; }
`;
/** Compact plugin card; every action retains its existing caller-owned operation guards. */
export function PluginSummaryCard({ name, pluginId, description, health, tone, release, releaseNote, nodes, onOpen, actions, primaryLabel, muted = false }: PluginSummaryCardProps) {
  const { t } = useSkulkTranslation();
  const menu = useRef<HTMLDetailsElement>(null);
  const [menuPlacement, setMenuPlacement] = useState({ above: false, availableHeight: 360 });
  // A card near the bottom of a phone must keep every operation reachable.
  const positionMenu = useCallback(() => {
    const details = menu.current;
    if (!details?.open) return;
    const anchor = details.querySelector('summary')?.getBoundingClientRect();
    const content = details.querySelector<HTMLDivElement>(':scope > div');
    if (!anchor || !content) return;
    const below = window.innerHeight - anchor.bottom - 14;
    const aboveSpace = anchor.top - 14;
    const above = below < content.scrollHeight && aboveSpace > below;
    setMenuPlacement({ above, availableHeight: Math.max(48, above ? aboveSpace : below) });
  }, []);
  useEffect(() => {
    window.addEventListener('resize', positionMenu);
    return () => window.removeEventListener('resize', positionMenu);
  }, [positionMenu]);
  const choose = (action: () => void) => { if (menu.current?.open) { menu.current.open = false; menu.current.querySelector('summary')?.focus(); } action(); };
  const items = actions ?? [{ id: 'configure', label: t('plugins.configureSettings', 'Configure settings'), onSelect: onOpen }];
  const words = name.trim().split(/\s+/);
  const initials = words.length > 1 ? words.slice(0, 2).map(word => word[0]).join('').toUpperCase() : name.slice(0, 2).toUpperCase();
  return <Card $muted={muted}>
    <Mark aria-hidden="true">{name.startsWith('managed.') ? <FiPackage size={20} /> : initials}</Mark>
    <div><Row><h2>{name}</h2><StatusPill tone={tone}>{health}</StatusPill></Row>
      {description && <Description>{description}</Description>}
      <Metadata>{pluginId && <span title={pluginId}>{pluginId.length > 30 ? `${pluginId.slice(0, 16)}…${pluginId.slice(-6)}` : pluginId}</span>}<span>{t('plugins.release', 'Release')} <b title={release}>{release}</b></span>{releaseNote && <span>{releaseNote}</span>}</Metadata>
    </div>
    <Nodes>{nodes.map((node, index) => <Chip key={`${node}-${index}`}>{node}</Chip>)}</Nodes>
    <Actions><Primary variant="outline" size="sm" onClick={() => choose(onOpen)}>{primaryLabel ?? t('plugins.configure', 'Configure')}</Primary>
    <Menu ref={menu} $above={menuPlacement.above} $availableHeight={menuPlacement.availableHeight} onToggle={positionMenu} onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget)) event.currentTarget.open = false; }} onKeyDown={event => { if (event.key === 'Escape' && menu.current?.open) { event.preventDefault(); event.stopPropagation(); menu.current.open = false; menu.current.querySelector('summary')?.focus(); } }}>
      <summary aria-label={t('plugins.more', 'More plugin actions')}><FiMoreHorizontal /></summary>
      <div>{items.map(item => <div key={item.id}>{item.separatorBefore && <hr />}<MenuAction type="button" $danger={item.danger} disabled={item.disabled} onClick={() => choose(item.onSelect)}>{item.label}</MenuAction></div>)}
        {actions && <p>{t('plugins.uninstallNote', 'Uninstall is not a data purge — credentials and cleanup records are retained.')}</p>}
      </div>
    </Menu></Actions>
  </Card>;
}
