import styled from 'styled-components';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { useGetBurstBudgetQuery } from '../../store/endpoints/burst';
import { formatCents, positiveUsd, spentOfLimit } from '../models/burstPresentation';

/** How often the spend line reads the limit again while it is shown. */
const BUDGET_POLL_MS = 30_000;

const Line = styled.p`
  margin: 6px 0 0;
  font-size: 12px;
  line-height: 1.4;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

/** Props for the compact spend line. */
export interface BurstSpendLineProps {
  pluginId: string;
  nodeId: string;
}

/**
 * One or two lines for a burst provider's satellite flyout: what has been
 * spent of the limit and, while machines run, what they hold. Says nothing
 * while the limit is unknown, so the flyout never shows a guess.
 */
export function BurstSpendLine({ pluginId, nodeId }: BurstSpendLineProps) {
  const { t } = useSkulkTranslation();
  const budget = useGetBurstBudgetQuery({ pluginId, nodeId }, { pollingInterval: BUDGET_POLL_MS, skipPollingIfUnfocused: true }).currentData;
  const spent = spentOfLimit(budget, t);
  if (!spent || !budget) return null;
  return (
    <Line data-burst-spend>
      {spent}
      {positiveUsd(budget.heldUsd) && budget.heldUsd !== null ? (
        <>
          <br />
          {t('burst.rentals.heldNow', '{held} held for machines running now', { held: formatCents(budget.heldUsd) })}
        </>
      ) : null}
    </Line>
  );
}
