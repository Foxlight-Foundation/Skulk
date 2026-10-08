/**
 * Test fixture: the node settings a studio-style capability declares, in the
 * exact JSON Schema shape pydantic writes for its settings model. Optional
 * fields are `anyOf` a bounded string and `null`; titles are pydantic's
 * title-casing of each key. A freshly installed node reports `{}` values at
 * revision 0, turned off.
 */
export const exampleStudioSettingsSchema: Record<string, unknown> = {
  additionalProperties: false,
  properties: {
    skulk_api_url: {
      default: 'http://127.0.0.1:52415', description: 'Where the studio reaches the Skulk API that renders its shots.',
      maxLength: 2048, title: 'Skulk Api Url', type: 'string',
    },
    default_model: {
      anyOf: [{ maxLength: 256, type: 'string' }, { type: 'null' }], default: null,
      description: 'The video model a new shot uses. Leave empty to pick a placed video model each time.', title: 'Default Model',
    },
    render_timeout_seconds: {
      default: 86400, description: 'How long one render may run before it is stopped.', maximum: 86400, minimum: 60,
      title: 'Render Timeout Seconds', type: 'integer',
    },
    keep_renders: {
      default: 32, description: 'How many finished renders are kept before the oldest are removed.', maximum: 1024, minimum: 1,
      title: 'Keep Renders', type: 'integer',
    },
    refine_model: {
      anyOf: [{ maxLength: 256, type: 'string' }, { type: 'null' }], default: null,
      description: 'The chat model that refines a prompt with guides. Leave empty to use any ready chat model.', title: 'Refine Model',
    },
    refine_timeout_seconds: {
      default: 600, description: 'How long refining a prompt may take.', maximum: 3600, minimum: 30,
      title: 'Refine Timeout Seconds', type: 'integer',
    },
    page_reach: {
      default: 'fabric', description: 'Which browsers may open the studio page.', title: 'Page Reach', type: 'string',
    },
    comfy_url: {
      anyOf: [{ maxLength: 2048, type: 'string' }, { type: 'null' }], default: null,
      description: 'An engine address to use instead of the one Skulk manages.', title: 'Comfy Url',
    },
  },
  title: 'StudioSettings',
  type: 'object',
};
