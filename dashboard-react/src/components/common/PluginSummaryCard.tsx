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
  padding: 16px 18px; margin: 12px 0; border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 14px;
  background: ${({ theme }) => theme.colors.surface}; display: grid; grid-template-columns: 44px minmax(0, 1fr) auto auto; gap: 14px; align-items: start;
  h2 { font-size: 16px; margin: 0; overflow-wrap: anywhere; }
  p { margin: 8px 0; color: ${({ theme }) => theme.colors.textSecondary}; font-size: 13px; }
  @media(max-width: 600px) { grid-template-columns: 44px minmax(0, 1fr) auto; > button { grid-column: 2; grid-row: 2; justify-self: start; } > details { grid-column: 3; grid-row: 1; } }
`;
const Row = styled.div`display: flex; align-items: center; flex-wrap: wrap; gap: 10px;`;
const Chip = styled.span`padding: 4px 8px; border-radius: 6px; background: ${({ theme }) => theme.colors.selected}; font: 11px ${({ theme }) => theme.fonts.mono}; overflow-wrap: anywhere;`;
const Menu = styled.details`
  margin-left: auto; position: relative;
  summary { display: flex; padding: 8px; cursor: pointer; list-style: none; }
  summary::-webkit-details-marker { display: none; }
  > div { position: absolute; right: 0; z-index: 1; background: ${({ theme }) => theme.colors.surface}; border: 1px solid ${({ theme }) => theme.colors.border}; padding: 8px; border-radius: 8px; white-space: nowrap; box-shadow: ${({ theme }) => theme.colors.shadowPop}; }
`;
/** Compact plugin card; detailed changes remain behind the existing fenced workflows. */
export function PluginSummaryCard({ name, description, health, tone, release, nodes, onOpen }: PluginSummaryCardProps) {
  const { t } = useSkulkTranslation();
  return <Card>
    <Monogram>{name.slice(0, 2).toUpperCase()}</Monogram>
    <div><Row><h2>{name}</h2><StatusPill tone={tone}>{health}</StatusPill></Row>
      {description && <p>{description}</p>}
      <p>{t('plugins.release', 'Release')} · {release}</p>
      <Row>{nodes.map((node, index) => <Chip key={`${node}-${index}`}>{node}</Chip>)}</Row>
    </div>
    <Button onClick={onOpen}>{t('plugins.configure', 'Configure')}</Button>
    <Menu><summary aria-label={t('plugins.more', 'More plugin actions')}><FiMoreHorizontal /></summary><div><Button variant="ghost" onClick={onOpen}>{t('plugins.manage', 'Manage plugin')}</Button></div></Menu>
  </Card>;
}
