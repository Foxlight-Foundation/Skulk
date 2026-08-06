import styled from 'styled-components';
import { FiZap } from 'react-icons/fi';
import { InfoTooltip } from '../common/InfoTooltip';
import { formatBytes } from '../../utils/format';
import { useSkulkTranslation } from '../../i18n/tolgee';
import type { BurstInfo } from './burst';

const Chip = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.warningOnSurface};
  background: ${({ theme }) => theme.colors.warningBg};
  border: 1px solid ${({ theme }) => theme.colors.warningBg};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 2px 8px;
`;

/** Amber chip naming the burst requirement, with the reason in a tooltip. */
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
    <InfoTooltip placement="left" delay={100} content={<div style={{ maxWidth: 260 }}>{reason}</div>}>
      <Chip>
        <FiZap size={11} />
        {t('burst.chip', 'Burst')}
        {info.estimated ? ' ~' : ''}
      </Chip>
    </InfoTooltip>
  );
}
