import { expect, it } from 'vitest';
import type { SkulkTranslate } from '../../i18n/tolgee';
import type { BurstBudget } from '../../store/endpoints/burst';
import { formatCents, positiveUsd, spentOfLimit } from './burstPresentation';

it('shows spending in dollars and cents, never cheaper than it could be', () => {
  expect(formatCents('2.500000')).toBe('$2.50');
  expect(formatCents('0.002750')).toBe('$0.01');
  expect(formatCents('1.527800')).toBe('$1.53');
  expect(formatCents('0.000000')).toBe('$0.00');
  expect(formatCents('1000.000000')).toBe('$1,000.00');
});

it('shows what is left rounded down, so it never promises more', () => {
  expect(formatCents('7.499999', 'down')).toBe('$7.49');
  expect(formatCents('7.500000', 'down')).toBe('$7.50');
});

it('says spending of the limit with the limit rounded down and spending rounded up', () => {
  const t = ((_key: string, fallback: string, params?: Record<string, unknown>) =>
    fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? `{${name}}`))) as unknown as SkulkTranslate;
  const budget: BurstBudget = {
    pluginId: 'managed.runpod',
    nodeId: 'gpu',
    state: 'available',
    message: null,
    limitUsd: '10.000001',
    spentUsd: '2.500001',
    heldUsd: null,
    leftUsd: '7.500000',
    rentals: 1,
    countingSince: null,
    changeLimitAction: null,
    startOverAction: null,
  };
  expect(spentOfLimit(budget, t)).toBe('$2.51 of $10.00 spent');
});

it('tells a positive amount from zero', () => {
  expect(positiveUsd('0.000000')).toBe(false);
  expect(positiveUsd('0.000001')).toBe(true);
  expect(positiveUsd(null)).toBe(false);
});
