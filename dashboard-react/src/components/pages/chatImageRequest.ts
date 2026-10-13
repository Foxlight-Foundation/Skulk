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
    stream: false, partial_images: 0, advanced_params: advanced };
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
  const result: unknown = await response.json();
  if (!response.ok) {
    const detail = result && typeof result === 'object' && 'detail' in result ? result.detail : null;
    throw new Error(typeof detail === 'string' ? detail : `Image request failed (${response.status}).`);
  }
  if (!result || typeof result !== 'object' || !('data' in result) || !Array.isArray(result.data) || result.data.length !== params.numImages) {
    throw new Error('Image response did not contain the requested outputs.');
  }
  return result.data.map((item: unknown) => {
    if (!item || typeof item !== 'object' || !('b64_json' in item) || typeof item.b64_json !== 'string' || !item.b64_json) {
      throw new Error('Image response contained an invalid output.');
    }
    return `data:image/${params.outputFormat};base64,${item.b64_json}`;
  });
}
