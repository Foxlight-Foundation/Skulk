import styled from 'styled-components';
import { FiZap } from 'react-icons/fi';
import { InfoTooltip } from '../common/InfoTooltip';
import { formatBytes } from '../../utils/format';
import { useSkulkTranslation } from '../../i18n/tolgee';
import type { BurstInfo } from './burst';

/* Icon-only, inked in the brand accent like the row's other quiet
 * affordances; the name and reason live in the tooltip. */
const Chip = styled.span`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  flex-shrink: 0;
  border-radius: 50%;
  color: ${({ theme }) => theme.colors.gold};
  background: ${({ theme }) => theme.colors.goldBg};
`;

/** Brand-accent burst marker; the name and reason live in its tooltip. */
export function BurstChip({ info, fleetMemoryBytes }: {
  info: BurstInfo;
  fleetMemoryBytes?: number;
}) {
  const { t } = useSkulkTranslation();
  const needed = info.neededBytes !== undefined
    ? `${info.estimated ? '~' : ''}${formatBytes(info.neededBytes)}`
    : undefined;
  const reason = info.reason === 'engine'
    ? t(
        'burst.engineReason',
        'No node in your fleet runs {format} artifacts. Serve it with burst capacity or an added node.',
        { format: info.format ?? t('burst.thisFormat', 'this format') },
      )
    : t(
        'burst.sizeReason',
        'Larger than your fleet’s total memory ({needed} of weights vs {available} fleet-wide). Serve it with burst capacity or added nodes.',
        {
          needed: needed ?? t('burst.unknownSize', 'unknown size'),
          available: fleetMemoryBytes !== undefined ? formatBytes(fleetMemoryBytes) : '?',
        },
      );
  return (
    <InfoTooltip
      placement="left"
      delay={100}
      content={
        <div style={{ maxWidth: 260 }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>
            {t('burst.chip', 'Burst')}
            {info.estimated ? ` (${t('burst.estimated', 'estimated')})` : ''}
          </div>
          {reason}
        </div>
      }
    >
      <Chip aria-label={t('burst.chip', 'Burst')} role="img">
        <FiZap size={12} />
      </Chip>
    </InfoTooltip>
  );
}
