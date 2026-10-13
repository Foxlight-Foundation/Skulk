/**
 * The one dollar field a provider's change-limit action asks for, read from
 * the action's own form, so the dashboard never assumes one provider's field
 * name or bounds.
 */
export interface LimitField {
  /** The form property the amount is sent as. */
  name: string;
  /** The amount must be at least this, when the form says so. */
  minimum: number | null;
  /** The amount must be more than this, when the form says so. */
  exclusiveMinimum: number | null;
  /** The amount must be at most this, when the form says so. */
  maximum: number | null;
  /** True when the form takes whole dollars only. */
  wholeDollars: boolean;
}

const NUMBER_TYPES = new Set(['number', 'integer']);

function finite(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

/**
 * Find the amount field in a change-limit action's form, or null when the
 * dashboard cannot fill the form with one amount.
 *
 * The field is the required number property marked with the `usd` unit; with
 * no such mark, the only required number property. Any other required field
 * means the form needs more than one amount, so it is not shown here.
 */
export function limitField(schema: Record<string, unknown> | null | undefined): LimitField | null {
  const properties = record(schema?.properties);
  if (!properties) return null;
  const declared: unknown = schema?.required;
  const required = Array.isArray(declared) ? declared.filter((name): name is string => typeof name === 'string') : [];
  const numeric = required.filter((name) => {
    const field = record(properties[name]);
    return field !== null && typeof field.type === 'string' && NUMBER_TYPES.has(field.type);
  });
  if (numeric.length !== required.length) return null;
  const marked = numeric.filter((name) => record(properties[name])?.['x-skulk-unit'] === 'usd');
  const name = marked.length === 1 ? marked[0] : numeric.length === 1 ? numeric[0] : null;
  if (name === null || required.length !== 1) return null;
  const field = record(properties[name]);
  if (!field) return null;
  return {
    name,
    minimum: finite(field.minimum),
    exclusiveMinimum: finite(field.exclusiveMinimum),
    maximum: finite(field.maximum),
    wholeDollars: field.type === 'integer',
  };
}

/**
 * Whether a setup action's form takes no values, so it can run from a single
 * button: no required field, or no form at all.
 */
export function takesNoValues(schema: Record<string, unknown> | null | undefined): boolean {
  if (!schema) return false;
  const declared: unknown = schema.required;
  return !Array.isArray(declared) || declared.length === 0;
}

/** Whether an amount in dollars fits the field's own type and bounds (and is above zero). */
export function amountFits(field: LimitField, amount: number): boolean {
  if (!(amount > 0)) return false;
  if (field.wholeDollars && !Number.isInteger(amount)) return false;
  if (field.minimum !== null && amount < field.minimum) return false;
  if (field.exclusiveMinimum !== null && amount <= field.exclusiveMinimum) return false;
  if (field.maximum !== null && amount > field.maximum) return false;
  return true;
}
