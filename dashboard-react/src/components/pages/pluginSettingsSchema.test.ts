import { describe, expect, it } from 'vitest';
import { exampleStudioSettingsSchema } from './exampleStudioSettings.fixture';
import {
  fieldLabel, formatUsdAmount, humanizeKey, isAdvancedField, isWholeCents, settingsField, settingsUnit, supportedConfigurationSchema,
} from './pluginSettingsSchema';

describe('supportedConfigurationSchema', () => {
  it('accepts the optional fields pydantic writes as a scalar or null', () => {
    expect(supportedConfigurationSchema(exampleStudioSettingsSchema)).toBe(true);
    expect(supportedConfigurationSchema({ type: 'object', properties: { name: { type: ['string', 'null'] } } })).toBe(true);
    expect(supportedConfigurationSchema({ type: 'object', properties: { size: { oneOf: [{ type: 'integer', minimum: 1 }, { type: 'null' }] } } })).toBe(true);
    // An optional enum lives behind a local reference.
    expect(supportedConfigurationSchema({
      type: 'object', $defs: { Reach: { enum: ['host', 'fabric'], type: 'string' } },
      properties: { reach: { anyOf: [{ $ref: '#/$defs/Reach' }, { type: 'null' }], default: null } },
    })).toBe(true);
  });

  it('still refuses what the form cannot render faithfully', () => {
    const refused = [
      { type: 'object', properties: { tags: { type: 'array', items: { type: 'string' } } } },
      { type: 'object', properties: { token: { type: 'string', writeOnly: true } } },
      { type: 'object', properties: { token: { anyOf: [{ type: 'string', format: 'password' }, { type: 'null' }] } } },
      { type: 'object', properties: { either: { anyOf: [{ type: 'string' }, { type: 'integer' }] } } },
      { type: 'object', properties: { wider: { anyOf: [{ type: 'string' }, { type: 'integer' }, { type: 'null' }] } } },
      { type: 'object', properties: { nested: { anyOf: [{ anyOf: [{ type: 'string' }, { type: 'integer' }] }, { type: 'null' }] } } },
      { type: 'object', properties: { list: { anyOf: [{ type: 'array', items: { type: 'string' } }, { type: 'null' }] } } },
      { type: 'object', properties: { both: { anyOf: [{ type: 'string' }, { type: 'null' }], oneOf: [{ type: 'string' }, { type: 'null' }] } } },
    ];
    for (const schema of refused) expect(supportedConfigurationSchema(schema)).toBe(false);
  });
});

describe('settingsField', () => {
  it('unwraps a nullable scalar, keeping its constraints and the property\'s own title, description and default', () => {
    const field = settingsField((exampleStudioSettingsSchema.properties as Record<string, Record<string, unknown>>).default_model, exampleStudioSettingsSchema);
    expect(field.nullable).toBe(true);
    expect(field.schema).toMatchObject({ type: 'string', maxLength: 256, title: 'Default Model', default: null });
    expect(field.schema).not.toHaveProperty('anyOf');
    expect(settingsField({ type: 'integer', default: 32 }, {}).nullable).toBe(false);
  });
});

describe('field labels', () => {
  it('turns a mechanical title back into plain words, keeping acronyms and units readable', () => {
    expect(fieldLabel('skulk_api_url', 'Skulk Api Url')).toBe('Skulk API URL');
    expect(fieldLabel('render_timeout_seconds', 'Render Timeout Seconds')).toBe('Render timeout (seconds)');
    expect(fieldLabel('comfy_url', 'Comfy Url')).toBe('Comfy URL');
    expect(fieldLabel('keep_renders', undefined)).toBe('Keep renders');
    expect(fieldLabel('node_id', 'Node Id')).toBe('Node ID');
  });

  it('keeps a title the author wrote', () => {
    expect(fieldLabel('comfy_url', 'ComfyUI address')).toBe('ComfyUI address');
    expect(fieldLabel('page_reach', 'Who can open the page')).toBe('Who can open the page');
  });

  it('humanizes check codes the same way', () => {
    expect(humanizeKey('placed_video_model')).toBe('Placed video model');
    expect(humanizeKey('engine')).toBe('Engine');
  });
});

describe('field hints', () => {
  it('reads a known unit only on a number field', () => {
    expect(settingsUnit({ type: 'number', 'x-skulk-unit': 'usd' })).toBe('usd');
    expect(settingsUnit({ type: 'integer', 'x-skulk-unit': 'minutes' })).toBe('minutes');
    expect(settingsUnit({ type: ['integer', 'null'], 'x-skulk-unit': 'hours' })).toBe('hours');
    expect(settingsUnit({ type: 'number', 'x-skulk-unit': 'seconds' })).toBeNull();
    expect(settingsUnit({ type: 'string', 'x-skulk-unit': 'usd' })).toBeNull();
    expect(settingsUnit({ type: 'number' })).toBeNull();
  });

  it('marks a field advanced only when it says so exactly', () => {
    expect(isAdvancedField({ 'x-skulk-advanced': true })).toBe(true);
    expect(isAdvancedField({ 'x-skulk-advanced': 'true' })).toBe(false);
    expect(isAdvancedField({})).toBe(false);
  });

  it('keeps both hints through a nullable value and a local reference', () => {
    const nullable = settingsField({ anyOf: [{ type: 'number' }, { type: 'null' }], 'x-skulk-unit': 'usd', 'x-skulk-advanced': true }, {});
    expect(settingsUnit(nullable.schema)).toBe('usd');
    expect(isAdvancedField(nullable.schema)).toBe(true);
    const root = { $defs: { Limit: { type: 'number', 'x-skulk-unit': 'usd' } } };
    const referenced = settingsField({ $ref: '#/$defs/Limit', 'x-skulk-advanced': true }, root);
    expect(settingsUnit(referenced.schema)).toBe('usd');
    expect(isAdvancedField(referenced.schema)).toBe(true);
  });

  it('tells whole cents from finer amounts without floating-point noise', () => {
    expect(isWholeCents(0.29)).toBe(true);
    expect(isWholeCents(10)).toBe(true);
    expect(isWholeCents(0.005)).toBe(false);
  });

  it('shows dollars as cents, and finer amounts in full', () => {
    expect(formatUsdAmount(10)).toBe('10.00');
    expect(formatUsdAmount(0.5)).toBe('0.50');
    expect(formatUsdAmount(0.29)).toBe('0.29');
    expect(formatUsdAmount(0.123456)).toBe('0.123456');
  });

  it('does not repeat a declared unit in a label made from the key', () => {
    expect(fieldLabel('spending_limit_usd', undefined, 'usd')).toBe('Spending limit');
    expect(fieldLabel('idle_minutes', 'Idle Minutes', 'minutes')).toBe('Idle');
    expect(fieldLabel('rental_lifetime_hours', undefined, 'hours')).toBe('Rental lifetime');
    // Without the hint the unit word still reads as a qualifier.
    expect(humanizeKey('spending_limit_usd')).toBe('Spending limit (US dollars)');
    expect(humanizeKey('idle_minutes')).toBe('Idle (minutes)');
    // An author's own title is kept as written.
    expect(fieldLabel('spending_limit_usd', 'Spend no more than', 'usd')).toBe('Spend no more than');
  });
});
