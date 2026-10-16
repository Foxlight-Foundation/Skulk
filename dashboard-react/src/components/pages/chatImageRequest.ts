import { operatorSession } from '../../auth/operatorSession';
import type { ImageGenerationParams } from '../display/ImageParamsPanel';
import type { ImageCapabilityInfo } from '../../types/models';

/** Generate final images or edit ordered references on a mounted native image model. */
export async function requestDashboardImages(
  model: string, prompt: string, params: ImageGenerationParams,
  profile: ImageCapabilityInfo, references: readonly string[], signal: AbortSignal,
): Promise<string[]> {
  if (references.length > profile.maxReferenceImages) {
    throw new Error(`This model accepts at most ${profile.maxReferenceImages} reference images.`);
  }
  const fixed = profile.fixedSigmas.length > 0;
  const advanced = {
    ...(params.seed === null ? {} : { seed: params.seed }),
    ...(!fixed && params.numInferenceSteps !== null ? { num_inference_steps: params.numInferenceSteps } : {}),
    ...(!fixed && params.guidance !== null ? { guidance: params.guidance } : {}),
    ...(!fixed && params.negativePrompt ? { negative_prompt: params.negativePrompt } : {}),
  };
  const payload = { model, prompt, size: params.size, n: params.numImages,
    output_format: params.outputFormat, response_format: 'b64_json', quality: params.quality,
    stream: true, partial_images: 0, advanced_params: advanced };
  let body: BodyInit;
  let headers: HeadersInit | undefined;
  if (references.length) {
    const form = new FormData();
    for (const [key, value] of Object.entries(payload)) {
      form.append(key, typeof value === 'object' ? JSON.stringify(value) : String(value));
    }
    form.append('input_fidelity', params.inputFidelity);
    for (const [index, source] of references.entries()) {
      const response = await fetch(source, { signal });
      if (!response.ok) throw new Error('Reference image could not be read.');
      form.append('image[]', await response.blob(), `reference-${index}.png`);
    }
    body = form;
  } else {
    body = JSON.stringify(payload); headers = { 'Content-Type': 'application/json' };
  }
  const response = await operatorSession.fetch(references.length ? '/v1/images/edits' : '/v1/images/generations', {
    method: 'POST', headers, body, signal,
  });
  if (!response.ok) {
    const result: unknown = await response.json();
    const detail = result && typeof result === 'object' && 'detail' in result ? result.detail : null;
    throw new Error(typeof detail === 'string' ? detail : `Image request failed (${response.status}).`);
  }
  if (!response.body) throw new Error('Image response did not contain a stream.');
  // Stop aborts this owned streaming response, which cancels the API command;
  // a non-streaming fetch disconnect would leave the native render running.
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const outputs = new Map<number, string>();
  let pending = '';
  let complete = false;
  const abort = () => { void reader.cancel().catch(() => undefined); };
  signal.addEventListener('abort', abort, { once: true });
  try {
    while (!complete) {
      signal.throwIfAborted();
      const chunk = await reader.read();
      signal.throwIfAborted();
      pending = (pending + decoder.decode(chunk.value, { stream: !chunk.done })).replace(/\r\n/g, '\n');
      let boundary: number;
      while ((boundary = pending.indexOf('\n\n')) >= 0) {
        const frame = pending.slice(0, boundary);
        pending = pending.slice(boundary + 2);
        const data = frame.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
        if (!data) continue;
        if (data === '[DONE]') { complete = true; break; }
        const event: unknown = JSON.parse(data);
        if (!event || typeof event !== 'object') throw new Error('Invalid image stream event.');
        if ('error' in event) {
          const error = event.error;
          throw new Error(error && typeof error === 'object' && 'message' in error && typeof error.message === 'string' ? error.message : 'Image generation failed.');
        }
        if (!('type' in event) || event.type !== 'final') continue;
        if (!('image_index' in event) || typeof event.image_index !== 'number' || !Number.isInteger(event.image_index) || event.image_index < 0 || event.image_index >= params.numImages || outputs.has(event.image_index)
          || !('data' in event) || !event.data || typeof event.data !== 'object' || !('b64_json' in event.data) || typeof event.data.b64_json !== 'string' || !event.data.b64_json) {
          throw new Error('Image response contained an invalid output.');
        }
        outputs.set(event.image_index, `data:image/${params.outputFormat};base64,${event.data.b64_json}`);
      }
      if (chunk.done) break;
    }
    if (!complete || outputs.size !== params.numImages) throw new Error('Image response did not contain the requested outputs.');
    return Array.from({ length: params.numImages }, (_, index) => {
      const output = outputs.get(index);
      if (!output) throw new Error('Image response did not contain the requested outputs.');
      return output;
    });
  } finally {
    signal.removeEventListener('abort', abort);
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
