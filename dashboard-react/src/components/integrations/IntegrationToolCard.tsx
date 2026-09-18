import styled from 'styled-components';
import { Monogram } from '../common/Surfaces';
import { useSkulkTranslation } from '../../i18n/tolgee';

const Card = styled.button`
  min-width: 0; min-height: 150px; display: flex; flex-direction: column;
  gap: 10px; padding: 16px; text-align: left; border-radius: 14px;
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: ${({ theme }) => theme.colors.surface};
  transition: background 120ms, border-color 120ms;
  &:hover { background: ${({ theme }) => theme.colors.surfaceHover}; border-color: ${({ theme }) => theme.colors.borderStrong}; }
  strong { font-size: 15px; font-weight: 600; color: ${({ theme }) => theme.colors.text}; }
  p { font-size: 12.5px; line-height: 1.5; color: ${({ theme }) => theme.colors.textSecondary}; }
  footer { margin-top: auto; font: 11px ${({ theme }) => theme.fonts.mono}; color: ${({ theme }) => theme.colors.subtleText}; }
`;
const Top = styled.div`display: flex; align-items: center; justify-content: space-between; gap: 8px; width: 100%; span:last-child { padding: 2px 6px; border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 999px; font-family: ${({ theme }) => theme.fonts.mono}; font-size: 10px; color: ${({ theme }) => theme.colors.subtleText}; }`;

/** Tool discovery card; setup availability is not presented as connection evidence. */
export function IntegrationToolCard({ name, monogram, description, method, onOpen }: {
  name: string; monogram: string; description: string; method: string; onOpen: () => void;
}) {
  const { t } = useSkulkTranslation();
  return <Card type="button" onClick={onOpen} aria-label={name}>
    <Top><Monogram $size={36}>{monogram}</Monogram><span>{t('integrations.setUp', 'set up')}</span></Top>
    <strong>{name}</strong><p>{description}</p><footer>{method}</footer>
  </Card>;
}
