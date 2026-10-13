import { expect, it } from 'vitest';
import { amountFits, limitField, takesNoValues } from './limitField';

it('finds the one required dollar field and its bounds', () => {
  const field = limitField({
    type: 'object',
    required: ['spending_limit_usd'],
    properties: {
      spending_limit_usd: { type: 'number', exclusiveMinimum: 0, maximum: 1000, 'x-skulk-unit': 'usd' },
      maximum_rentals: { type: 'integer' },
    },
  });
  expect(field).toEqual({ name: 'spending_limit_usd', minimum: null, exclusiveMinimum: 0, maximum: 1000, wholeDollars: false });
  expect(amountFits(field!, 20)).toBe(true);
  expect(amountFits(field!, 0)).toBe(false);
  expect(amountFits(field!, 1000.01)).toBe(false);
});

it('accepts an unmarked form with a single required number, keeping whole dollars whole', () => {
  const field = limitField({ required: ['cap'], properties: { cap: { type: 'integer', minimum: 1 } } });
  expect(field?.name).toBe('cap');
  expect(field?.wholeDollars).toBe(true);
  expect(amountFits(field!, 10)).toBe(true);
  expect(amountFits(field!, 10.5)).toBe(false);
});

it('runs an action from one button only when its form takes no values', () => {
  expect(takesNoValues({ type: 'object', properties: {} })).toBe(true);
  expect(takesNoValues({ type: 'object', properties: { note: { type: 'string' } }, required: [] })).toBe(true);
  expect(takesNoValues({ type: 'object', required: ['confirm'], properties: { confirm: { type: 'boolean' } } })).toBe(false);
  expect(takesNoValues(null)).toBe(false);
});

it('refuses a form one amount cannot fill', () => {
  expect(limitField(null)).toBeNull();
  expect(limitField({ properties: {} })).toBeNull();
  expect(limitField({ required: ['cap', 'note'], properties: { cap: { type: 'number' }, note: { type: 'string' } } })).toBeNull();
  expect(limitField({ required: ['a', 'b'], properties: { a: { type: 'number' }, b: { type: 'number' } } })).toBeNull();
  expect(limitField({ required: ['label'], properties: { label: { type: 'string' } } })).toBeNull();
});
