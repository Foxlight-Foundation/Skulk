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
  it.each(['flux_3_klein', 'flux_3_klein_base'] as const)('sends literal %s captions and ten references in order', async family => {
    const caption = 'Studio photograph: an orange fox sits beside two ceramic mugs, cream wall and soft window light.';
    const references = Array.from({ length: 10 }, (_, index) => `data:image/png;base64,${btoa(String(index))}`);
    let form: FormData | null = null;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.startsWith('data:')) return new Response(atob(url.split(',')[1]));
      expect(url).toBe('/v1/images/edits');
      form = init?.body as FormData;
      return new Response('data: {"type":"final","image_index":0,"data":{"b64_json":"aW1hZ2U="}}\n\ndata: [DONE]\n\n');
    }));
    const klein: ImageCapabilityInfo = { ...profile, family, maxReferenceImages: 10, fixedSigmas: [], defaultSteps: family === 'flux_3_klein' ? 16 : 50 };
    await requestDashboardImages('model', caption, DEFAULT_IMAGE_PARAMS, klein, references, new AbortController().signal);
    const captured = form as FormData | null;
    expect(captured?.get('prompt')).toBe(caption);
    expect(captured?.get('advanced_params')).toBe('{}');
    const files = captured?.getAll('image[]') as File[];
    expect(await Promise.all(files.map(file => file.text()))).toEqual(Array.from({ length: 10 }, (_, index) => String(index)));
    await expect(requestDashboardImages('model', caption, DEFAULT_IMAGE_PARAMS, klein, [...references, references[0]], new AbortController().signal)).rejects.toThrow('at most 10');
  });
  it('preserves reference order and removes incompatible distilled overrides', async () => {
    let form: FormData | null = null;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.startsWith('data:')) return new Response(url.endsWith('MQ==') ? 'first' : 'second');
      expect(url).toBe('/v1/images/edits');
      expect(init?.body).toBeInstanceOf(FormData);
      form = init?.body as FormData;
      return new Response('data: {"type":"final","image_index":0,"data":{"b64_json":"aW1hZ2U="}}\n\ndata: [DONE]\n\n', { headers: { 'Content-Type': 'text/event-stream' } });
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
  it('collects fragmented final events in output order and surfaces terminal errors', async () => {
    const wire = 'data: {"type":"final","image_index":1,"data":{"b64_json":"second"}}\r\n\r\ndata: {"type":"final","image_index":0,"data":{"b64_json":"first"}}\r\n\r\ndata: [DONE]\r\n\r\n';
    vi.stubGlobal('fetch', vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(JSON.parse(String(init?.body)).stream).toBe(true);
      const encoded = new TextEncoder().encode(wire);
      return new Response(new ReadableStream({ start(controller) {
        for (let index = 0; index < encoded.length; index += 7) controller.enqueue(encoded.slice(index, index + 7));
        controller.close();
      } }));
    }));
    expect(await requestDashboardImages('model', 'literal', { ...DEFAULT_IMAGE_PARAMS, numImages: 2 }, profile, [], new AbortController().signal)).toEqual(['data:image/png;base64,first', 'data:image/png;base64,second']);
    vi.stubGlobal('fetch', vi.fn(async () => new Response('data: {"error":{"message":"native producer failed"}}\n\ndata: [DONE]\n\n')));
    await expect(requestDashboardImages('model', 'literal', DEFAULT_IMAGE_PARAMS, profile, [], new AbortController().signal)).rejects.toThrow('native producer failed');
  });
  it('cancels the owned streaming response when Stop aborts an active render', async () => {
    const controller = new AbortController();
    const cancelled = vi.fn();
    let started: (() => void) | undefined;
    const ready = new Promise<void>(resolve => { started = resolve; });
    vi.stubGlobal('fetch', vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(init?.signal).toBe(controller.signal);
      expect(JSON.parse(String(init?.body)).stream).toBe(true);
      return new Response(new ReadableStream({ start() { started?.(); }, cancel: cancelled }));
    }));
    const request = requestDashboardImages('model', 'literal', DEFAULT_IMAGE_PARAMS, profile, [], controller.signal);
    const rejected = expect(request).rejects.toThrow();
    await ready;
    // Let fetch deliver its response so the body owner is waiting for a frame.
    await new Promise(resolve => setTimeout(resolve, 0));
    controller.abort();
    await rejected;
    expect(cancelled).toHaveBeenCalledOnce();
  });

});
