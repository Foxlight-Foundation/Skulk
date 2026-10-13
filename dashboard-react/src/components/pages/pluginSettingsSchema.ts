/**
 * Reading a plugin's declared settings schema for the dashboard's settings
 * form: which contracts the form can render faithfully, how an optional
 * (nullable) scalar is unwrapped, and how a field is labelled.
 *
 * Schema content never causes a network request: only local `#/` references
 * are resolved, and anything the form cannot render without dropping a
 * constraint is refused rather than approximated.
 */

/** A JSON object, or an empty one for anything else. */
export function schemaRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

/** Resolve only local schema references; schema content never causes a network request. */
export function resolveSchema(schema: Record<string, unknown>, root: Record<string, unknown>): Record<string, unknown> {
  if (typeof schema.$ref !== 'string') return schema;
  if (!schema.$ref.startsWith('#/')) return {};
  let current: unknown = root;
  for (const key of schema.$ref.slice(2).split('/')) current = schemaRecord(current)[key.replaceAll('~1', '/').replaceAll('~0', '~')];
  return { ...schemaRecord(current), ...Object.fromEntries(Object.entries(schema).filter(([key]) => key !== '$ref')) };
}

/** The declared type, ignoring a `null` member of a type list such as `["string", "null"]`. */
export function schemaKind(schema: Record<string, unknown>): unknown {
  return Array.isArray(schema.type) ? schema.type.find((value) => value !== 'null') : schema.type;
}

const SCALAR_KINDS = ['string', 'boolean', 'number', 'integer'];
const UNSUPPORTED_KEYWORDS = ['oneOf', 'anyOf', 'allOf', 'if', 'patternProperties'];

function isNullSchema(schema: Record<string, unknown>): boolean {
  return schema.type === 'null' && Object.keys(schema).every((key) => key === 'type' || key === 'title' || key === 'description');
}

/** One field as the form renders it: the scalar or object schema plus whether it may be `null`. */
export interface SettingsField {
  /** The schema to render, with the property's own title, description and default kept. */
  schema: Record<string, unknown>;
  /** Whether an empty input means `null`: an optional value such as pydantic's `str | None`. */
  nullable: boolean;
}

/**
 * Unwrap a property into the field the form renders.
 *
 * Pydantic writes every optional field as
 * `anyOf: [{type: string, maxLength: N}, {type: null}]`, with the title,
 * description and default on the property itself. Exactly one scalar branch
 * plus one `null` branch is a nullable scalar: the scalar's constraints apply
 * and an empty input submits `null`. Any other union is returned unchanged so
 * the support check refuses it.
 */
export function settingsField(raw: Record<string, unknown>, root: Record<string, unknown>): SettingsField {
  const property = resolveSchema(raw, root);
  const unions = (['anyOf', 'oneOf'] as const).filter((key) => key in property);
  // Both keywords at once is a wider contract than one optional value.
  if (unions.length === 1 && Array.isArray(property[unions[0]])) {
    const union = unions[0];
    const branches = (property[union] as unknown[]).map((branch) => resolveSchema(schemaRecord(branch), root));
    const nulls = branches.filter(isNullSchema);
    const values = branches.filter((branch) => !isNullSchema(branch));
    if (branches.length === 2 && nulls.length === 1 && values.length === 1 && SCALAR_KINDS.includes(String(schemaKind(values[0])))) {
      const outer = Object.fromEntries(Object.entries(property).filter(([key]) => key !== union));
      return { schema: { ...values[0], ...outer }, nullable: true };
    }
    return { schema: property, nullable: false };
  }
  return { schema: property, nullable: Array.isArray(property.type) && property.type.includes('null') };
}

/**
 * Whether the form can render this settings contract without dropping a constraint.
 *
 * Supported: objects of strings, booleans, numbers and integers (with enums
 * and their bounds), nested objects of the same, and nullable scalars.
 * Refused: secrets (`writeOnly`, `format: password`), arrays, nested or wider
 * unions, conditionals and pattern properties. A refused contract still lets
 * the owner turn the node on or off; only editing needs the plugin's own tool.
 */
export function supportedConfigurationSchema(schema: Record<string, unknown>, root = schema, depth = 0): boolean {
  if (depth > 8) return false;
  const { schema: resolved } = settingsField(schema, root);
  if (resolved.writeOnly === true || resolved.format === 'password') return false;
  if (UNSUPPORTED_KEYWORDS.some((key) => key in resolved)) return false;
  const type = schemaKind(resolved);
  if (type === 'object') return Object.values(schemaRecord(resolved.properties)).every((field) => supportedConfigurationSchema(schemaRecord(field), root, depth + 1));
  return SCALAR_KINDS.includes(String(type));
}

/**
 * The units a plugin may declare for a number setting with `x-skulk-unit`.
 *
 * The stored value is already in that unit; the form only shows it ("$"
 * before a dollar amount, "minutes" or "hours" after a duration) and never
 * converts. A person never sees millionths of a dollar or seconds this way.
 */
export type SettingsUnit = 'usd' | 'minutes' | 'hours';

const SETTINGS_UNITS: readonly SettingsUnit[] = ['usd', 'minutes', 'hours'];

/** The unit a number field declares with `x-skulk-unit`, or null for none or one this dashboard does not know. */
export function settingsUnit(schema: Record<string, unknown>): SettingsUnit | null {
  const kind = schemaKind(schema);
  if (kind !== 'number' && kind !== 'integer') return null;
  return SETTINGS_UNITS.find((unit) => unit === schema['x-skulk-unit']) ?? null;
}

/**
 * The credentials a node's configuration schema says its setup fills in, with
 * `"x-skulk-setup-credentials"` at the schema's top level. Manage credentials
 * folds them away so the owner sees the keys they enter first. The hint is
 * advisory: anything malformed reads as none, which lists every credential.
 */
export function setupFilledCredentials(schema: Record<string, unknown> | undefined): ReadonlySet<string> {
  const raw = schema?.['x-skulk-setup-credentials'];
  if (!Array.isArray(raw)) return new Set();
  return new Set(raw.filter((item): item is string => typeof item === 'string' && item.length > 0 && item.length <= 128).slice(0, 32));
}

/** Whether a field asks, with `"x-skulk-advanced": true`, to sit in the collapsed Advanced section. */
export function isAdvancedField(schema: Record<string, unknown>): boolean {
  return schema['x-skulk-advanced'] === true;
}

/**
 * Whether a dollar amount is a whole number of cents. Amounts are compared at
 * millionths of a dollar, the finest unit plugins hold exactly, so binary
 * floating-point noise such as 0.29 never reads as finer than a cent.
 */
export function isWholeCents(value: number): boolean {
  return Number.isInteger(Math.round(value * 1e6) / 1e4);
}

/**
 * A dollar amount as a person reads it: two decimals ("10.00", "0.50"), or
 * every decimal it has when it is finer than a cent, so nothing is hidden.
 */
export function formatUsdAmount(value: number): string {
  return isWholeCents(value) ? value.toFixed(2) : String(value);
}

// Short words a mechanical title would write as "Api" or "Url".
const ACRONYMS = new Set(['api', 'url', 'urls', 'id', 'ids', 'uri', 'http', 'https', 'ui', 'gpu', 'cpu', 'ip', 'tls', 'json', 'llm', 'tts', 'stt', 'vram', 'ram']);
// A trailing unit reads as a qualifier: "Render timeout (seconds)".
const UNITS = new Set(['seconds', 'minutes', 'hours', 'days', 'bytes', 'megabytes', 'gigabytes', 'percent']);
// The key words a declared unit already says beside the control, so the label leaves them out.
const UNIT_KEY_WORDS: Record<SettingsUnit, readonly string[]> = { usd: ['usd', 'dollars'], minutes: ['minutes'], hours: ['hours'] };

/** What Python's `str.title()` makes of a key, as pydantic does when it names a field: `skulk_api_url` becomes `Skulk Api Url`. */
function mechanicalTitle(key: string): string {
  return key.replace(/[A-Za-z]+/g, (word) => word[0].toUpperCase() + word.slice(1).toLowerCase()).replaceAll('_', ' ');
}

/**
 * A key in plain words: `skulk_api_url` becomes `Skulk API URL`, `render_timeout_seconds` becomes `Render timeout (seconds)`.
 *
 * With a declared `unit`, a trailing word naming that same unit is left
 * out, since the control shows the unit itself: `spending_limit_usd` with
 * unit `usd` becomes `Spending limit` beside its "$". Without the hint a key
 * reads exactly as it always has.
 */
export function humanizeKey(key: string, unit: SettingsUnit | null = null): string {
  const words = key.split(/[_\s-]+/).filter(Boolean).map((word) => word.toLowerCase());
  if (words.length === 0) return key;
  if (unit && words.length > 1 && UNIT_KEY_WORDS[unit].includes(words[words.length - 1])) words.pop();
  const qualifier = words.length > 1 && UNITS.has(words[words.length - 1]) ? words.pop() : undefined;
  const text = words.map((word, index) => ACRONYMS.has(word) ? word.toUpperCase() : index === 0 ? word[0].toUpperCase() + word.slice(1) : word).join(' ');
  return qualifier ? `${text} (${qualifier})` : text;
}

/**
 * The label for one settings field: the author's title when it wrote one,
 * otherwise the key in plain words. A title that is only the generated
 * title-casing of the key counts as no title. `unit` is the field's
 * declared unit, which a label made from the key does not repeat.
 */
export function fieldLabel(key: string, title: unknown, unit: SettingsUnit | null = null): string {
  if (typeof title === 'string' && title.trim() && title !== mechanicalTitle(key)) return title.trim();
  return humanizeKey(key, unit);
}
