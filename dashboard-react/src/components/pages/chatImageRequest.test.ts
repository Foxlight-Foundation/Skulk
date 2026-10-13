import { afterEach, describe, expect, it, vi } from 'vitest';
import { requestDashboardImages } from './chatImageRequest';
import { DEFAULT_IMAGE_PARAMS } from '../display/ImageParamsPanel';
import type { ImageCapabilityInfo } from '../../types/models';

const profile: ImageCapabilityInfo = {
  family: 'qwen_image_21_turbo', defaultWidth: 2048, defaultHeight: 2048,
  dimensionMultiple: 32, maxPixels: 4400000, maxReferenceImages: 2,
  supportsAlpha: true, defaultSteps: 8, defaultGuidance: 1, fixedSigmas: [1, 0],
};
afterEach(() => vi.unstubAllGlobals());

describe('Native image request ownership', () => {
  it('preserves reference order and removes incompatible distilled overrides', async () => {
    let form: FormData | null = null;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.startsWith('data:')) return new Response(url.endsWith('MQ==') ? 'first' : 'second');
      expect(url).toBe('/v1/images/edits');
      expect(init?.body).toBeInstanceOf(FormData);
      form = init?.body as FormData;
      return new Response(JSON.stringify({ data: [{ b64_json: 'aW1hZ2U=' }] }), { status: 200 });
    }));
    const output = await requestDashboardImages('model', 'literal', {
      ...DEFAULT_IMAGE_PARAMS, seed: 42, numInferenceSteps: 20, guidance: 6, negativePrompt: 'bad', outputFormat: 'webp',
    }, profile, ['data:image/png;base64,MQ==', 'data:image/png;base64,Mg=='], new AbortController().signal);
    const captured = form as FormData | null;
    expect(captured?.get('advanced_params')).toBe('{"seed":42}');
    const files = captured?.getAll('image[]') as File[];
    expect(await files[0].text()).toBe('first');
    expect(await files[1].text()).toBe('second');
    expect(output).toEqual(['data:image/webp;base64,aW1hZ2U=']);
  });
  it('rejects excess references before any request', async () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    await expect(requestDashboardImages('model', 'literal', DEFAULT_IMAGE_PARAMS, profile, ['a', 'b', 'c'], new AbortController().signal)).rejects.toThrow('at most 2');
    expect(fetch).not.toHaveBeenCalled();
  });
});
