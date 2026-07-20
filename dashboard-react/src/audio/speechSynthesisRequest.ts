import type { AudioResponseFormat } from '../types/chat';

/** Dashboard-side mirror of the API's bounded reference-audio upload limit. */
export const MAX_REFERENCE_AUDIO_BYTES = 25 * 1024 * 1024;

/** Inputs required to construct one dashboard speech-synthesis request. */
export interface SpeechSynthesisRequestOptions {
  model: string;
  input: string;
  responseFormat: AudioResponseFormat;
  stream: boolean;
  voice: string | null;
  referenceAudio: File | null;
  referenceText: string;
  signal: AbortSignal;
}

/**
 * Build the JSON or multipart request used by dashboard TTS playback.
 *
 * Multipart is selected only when the user has supplied request-scoped
 * reference audio. Keeping the original File in the caller lets a sentence
 * queue reuse the exact same clip for every segment in one response.
 */
export function buildSpeechSynthesisRequest(
  options: SpeechSynthesisRequestOptions,
): RequestInit {
  const {
    model,
    input,
    responseFormat,
    stream,
    voice,
    referenceAudio,
    referenceText,
    signal,
  } = options;
  if (referenceAudio) {
    const formData = new FormData();
    formData.set('model', model);
    formData.set('input', input);
    formData.set('response_format', responseFormat);
    if (stream) formData.set('stream', 'true');
    if (voice) formData.set('voice', voice);
    formData.set('reference_audio', referenceAudio, referenceAudio.name);
    const normalizedReferenceText = referenceText.trim();
    if (normalizedReferenceText) formData.set('reference_text', normalizedReferenceText);
    return {
      method: 'POST',
      signal,
      body: formData,
    };
  }

  return {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify({
      model,
      input,
      response_format: responseFormat,
      ...(stream ? { stream: true } : {}),
      ...(voice ? { voice } : {}),
    }),
  };
}
