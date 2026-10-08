import { describe, expect, it } from 'vitest';
import { exampleStudioSettingsSchema } from './exampleStudioSettings.fixture';
import { fieldLabel, humanizeKey, settingsField, supportedConfigurationSchema } from './pluginSettingsSchema';

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
