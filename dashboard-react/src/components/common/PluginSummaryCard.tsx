import { useRef } from 'react';
import styled from 'styled-components';
import { FiMoreHorizontal } from 'react-icons/fi';
import { Button } from './Button';
import { Monogram, StatusPill, type StatusTone } from './Surfaces';
import { useSkulkTranslation } from '../../i18n/tolgee';

/** Display metadata and observed health for one plugin, independent of its operations. */
export interface PluginSummaryCardProps {
  name: string;
  description?: string;
  health: string;
  tone: StatusTone;
  release: string;
  nodes: string[];
  onOpen: () => void;
}
const Card = styled.article`
  padding: 16px 18px; margin: 10px 0; border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 14px;
  background: ${({ theme }) => theme.colors.surface}; display: grid; grid-template-columns: 44px minmax(0, 1fr) auto auto; gap: 16px; align-items: center;
  h2 { font-size: 16px; font-weight: 600; color: ${({ theme }) => theme.colors.text}; margin: 0; overflow-wrap: anywhere; }
  p { margin: 8px 0; color: ${({ theme }) => theme.colors.textSecondary}; font-size: 13px; }
  @media(max-width: 600px) { grid-template-columns: 44px minmax(0, 1fr); align-items: start; > div:nth-child(3), > div:nth-child(4) { grid-column: 2; } }
`;
const Metadata = styled.p`font: 11.5px ${({ theme }) => theme.fonts.mono}; span { color: ${({ theme }) => theme.colors.body}; }`;
const Row = styled.div`display: flex; align-items: center; flex-wrap: wrap; gap: 10px;`;
const Chip = styled.span`padding: 3px 8px; border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 6px; background: ${({ theme }) => theme.colors.surfaceHover}; font: 12px ${({ theme }) => theme.fonts.mono}; overflow-wrap: anywhere;`;
const Nodes = styled(Row)`max-width: 200px; gap: 6px; min-width: 0;`;
const Actions = styled.div`display: flex; align-items: center; gap: 6px;`;
const Menu = styled.details`
  margin-left: auto; position: relative;
  summary { display: flex; padding: 8px; cursor: pointer; list-style: none; }
  summary::-webkit-details-marker { display: none; }
  > div { position: absolute; right: 0; z-index: 5; background: ${({ theme }) => theme.colors.surfaceElevated}; border: 1px solid ${({ theme }) => theme.colors.border}; padding: 8px; border-radius: 8px; white-space: nowrap; box-shadow: ${({ theme }) => theme.colors.shadowPop}; }
`;
/** Compact plugin card; detailed changes remain behind the existing fenced workflows. */
export function PluginSummaryCard({ name, description, health, tone, release, nodes, onOpen }: PluginSummaryCardProps) {
  const { t } = useSkulkTranslation();
  const menu = useRef<HTMLDetailsElement>(null);
  const openDetails = () => { if (menu.current?.open) { menu.current.open = false; menu.current.querySelector('summary')?.focus(); } onOpen(); };
  return <Card>
    <Monogram>{name.slice(0, 2).toUpperCase()}</Monogram>
    <div><Row><h2>{name}</h2><StatusPill tone={tone}>{health}</StatusPill></Row>
      {description && <p>{description}</p>}
      <Metadata>{t('plugins.release', 'Release')} · <span>{release}</span></Metadata>
    </div>
    <Nodes>{nodes.map((node, index) => <Chip key={`${node}-${index}`}>{node}</Chip>)}</Nodes>
    <Actions><Button variant="outline" size="sm" onClick={openDetails}>{t('plugins.configure', 'Configure')}</Button>
    <Menu ref={menu} onKeyDown={event => { if (event.key === 'Escape' && menu.current?.open) { event.preventDefault(); event.stopPropagation(); menu.current.open = false; menu.current.querySelector('summary')?.focus(); } }}><summary aria-label={t('plugins.more', 'More plugin actions')}><FiMoreHorizontal /></summary><div><Button variant="ghost" onClick={openDetails}>{t('plugins.manage', 'Manage plugin')}</Button></div></Menu></Actions>
  </Card>;
}
