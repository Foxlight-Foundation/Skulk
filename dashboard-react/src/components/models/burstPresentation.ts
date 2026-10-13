import type { SkulkTranslate } from '../../i18n/tolgee';
import type { BurstBudget, BurstPhase } from '../../store/endpoints/burst';

/**
 * Show an exact six-decimal dollar string as dollars and cents.
 *
 * Spending rounds up, so nothing reads as cheaper than it could be; what is
 * left rounds down, so it never promises more than the provider allows.
 */
export function formatCents(value: string, rounding: 'up' | 'down' = 'up'): string {
  const [wholeText, fractionText = ''] = value.split('.');
  const whole = Number.parseInt(wholeText || '0', 10);
  const micro = Number.parseInt(fractionText.padEnd(6, '0').slice(0, 6), 10);
  const extra = rounding === 'up' && micro % 10_000 !== 0 ? 1 : 0;
  const cents = whole * 100 + Math.floor(micro / 10_000) + extra;
  const dollars = Math.floor(cents / 100)
    .toString()
    .replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return `$${dollars}.${(cents % 100).toString().padStart(2, '0')}`;
}

/** Whether an exact dollar string is more than zero. */
export function positiveUsd(value: string | null): boolean {
  return value !== null && /[1-9]/.test(value);
}

/** The spent-of-limit sentence for one budget, or null when it cannot be said. */
export function spentOfLimit(budget: BurstBudget | undefined, t: SkulkTranslate): string | null {
  if (!budget || budget.state !== 'available' || budget.spentUsd === null || budget.limitUsd === null) return null;
  return t('burst.rentals.spentOfLimit', '{spent} of {limit} spent', {
    spent: formatCents(budget.spentUsd),
    limit: formatCents(budget.limitUsd),
  });
}

/** Plain words for each rental phase, shared by every view of a rental. */
export function burstPhaseLabels(t: SkulkTranslate): Record<BurstPhase, string> {
  return {
    approving: t('burst.phase.approving', 'Approving'),
    refused: t('burst.phase.refused', 'Refused'),
    attention: t('burst.phase.attention', 'Needs attention'),
    uncertain: t('burst.phase.uncertain', 'Uncertain'),
    starting: t('burst.phase.starting', 'Starting the machine'),
    installing: t('burst.phase.installing', 'Installing Skulk'),
    joining: t('burst.phase.joining', 'Joining the cluster'),
    downloading: t('burst.phase.downloading', 'Downloading the model'),
    loading: t('burst.phase.loading', 'Loading the model'),
    ready: t('burst.phase.ready', 'Ready'),
    releasing: t('burst.phase.releasing', 'Releasing'),
    released: t('burst.phase.released', 'Released'),
  };
}
