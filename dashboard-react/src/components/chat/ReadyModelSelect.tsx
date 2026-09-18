import styled from 'styled-components';
import { useSkulkTranslation } from '../../i18n/tolgee';

const Select = styled.select<{ $fabric: boolean }>`
  min-width: 0; max-width: 100%; padding: 8px 12px;
  border: 1px solid ${({ theme, $fabric }) => $fabric ? theme.colors.borderLive : theme.colors.borderControl};
  border-radius: 8px; background: ${({ theme, $fabric }) => $fabric ? theme.colors.liveBg : theme.colors.surface};
  color: ${({ theme, $fabric }) => $fabric ? theme.colors.liveText : theme.colors.text};
  font: 14px ${({ theme }) => theme.fonts.body};
  option, optgroup { background: ${({ theme }) => theme.colors.surface}; color: ${({ theme }) => theme.colors.text}; }
`;

/** Shared native keyboard-accessible model chooser across ordinary and fabric chat. */
export function ReadyModelSelect({ value, onChange, models, fabricEnabled }: {
  value: string | null;
  onChange: (id: string) => void;
  models: { modelId: string }[];
  fabricEnabled: boolean;
}) {
  const { t } = useSkulkTranslation();
  const ids = Array.from(new Set(models.map(model => model.modelId)));
  return <Select $fabric={value === 'skulk/steward'} value={value ?? ''} onChange={event => onChange(event.target.value)} aria-label={t('chat.view.selectModel', 'Select chat model')}>
    {!value && <option value="" disabled>{t('chat.view.selectModel', 'Select chat model')}</option>}
    {fabricEnabled && <optgroup label={t('chat.fabricGroup', 'THE FABRIC')}><option value="skulk/steward">✦ Skulk</option></optgroup>}
    <optgroup label={t('chat.readyGroup', 'READY MODELS')}>{ids.map(id => <option key={id} value={id}>{id.split('/').pop()}</option>)}</optgroup>
  </Select>;
}
