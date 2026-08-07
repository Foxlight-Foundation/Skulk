import type { AudioResponseFormat } from '../types/chat';

/** Dashboard-side mirror of the API's bounded reference-audio upload limit. */
export const MAX_REFERENCE_AUDIO_BYTES = 25 * 1024 * 1024;

/** Stable seed used for every dashboard sentence and replay generation. */
export const DASHBOARD_SPEECH_SEED = 42;

/** Default omitted server budget and bounded ceiling for dashboard batch playback. */
const DEFAULT_BATCH_SPEECH_MAX_TOKENS = 4096;
const MAX_BATCH_SPEECH_MAX_TOKENS = 131_072;
const BATCH_SPEECH_TOKENS_PER_CHARACTER = 32;

/**
 * Size a batch-only synthesis budget for the complete input.
 *
 * Fish-family audio generation consumes substantially more acoustic tokens than
 * text tokens. The server's safe omitted default covers only about twenty
 * seconds, so completed-turn playback supplies a conservative length-scaled
 * ceiling. The model may stop naturally before this limit; the upper bound
 * prevents an unexpectedly non-terminating generator from running unbounded.
 */
export function batchSpeechMaxTokens(input: string): number {
  return Math.min(
    MAX_BATCH_SPEECH_MAX_TOKENS,
    Math.max(
      DEFAULT_BATCH_SPEECH_MAX_TOKENS,
      Math.ceil(input.length * BATCH_SPEECH_TOKENS_PER_CHARACTER),
    ),
  );
}

/** Inputs required to construct one dashboard speech-synthesis request. */
export interface SpeechSynthesisRequestOptions {
  model: string;
  input: string;
  language: string;
  responseFormat: AudioResponseFormat;
  stream: boolean;
  maxTokens: number | null;
  seed: number;
  voice: string | null;
  referenceAudio: File | null;
  referenceText: string;
  signal: AbortSignal;
}

const MODEL_LANGUAGE_BY_DASHBOARD_LANGUAGE: Readonly<Record<string, string>> = {
  de: 'German',
  en: 'English',
  es: 'Spanish',
  fr: 'French',
  it: 'Italian',
  ja: 'Japanese',
  ko: 'Korean',
  pt: 'Portuguese',
  ru: 'Russian',
  zh: 'Chinese',
};

/** Map the active dashboard locale to the language vocabulary used by TTS models. */
export function speechLanguageForDashboardLocale(locale: string | undefined): string {
  const primaryLanguage = (locale ?? 'en').trim().toLowerCase().replace('_', '-').split('-')[0];
  return MODEL_LANGUAGE_BY_DASHBOARD_LANGUAGE[primaryLanguage] ?? 'English';
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
    language,
    responseFormat,
    stream,
    maxTokens,
    seed,
    voice,
    referenceAudio,
    referenceText,
    signal,
  } = options;
  if (referenceAudio) {
    const formData = new FormData();
    formData.set('model', model);
    formData.set('input', input);
    formData.set('lang_code', language);
    formData.set('response_format', responseFormat);
    formData.set('seed', String(seed));
    if (maxTokens !== null) formData.set('max_tokens', String(maxTokens));
    if (stream) formData.set('stream', 'true');
    // The uploaded clip is the voice condition for this request. Sending the
    // catalog selection as well would make the conditioning source ambiguous.
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
      lang_code: language,
      response_format: responseFormat,
      seed,
      ...(maxTokens !== null ? { max_tokens: maxTokens } : {}),
      ...(stream ? { stream: true } : {}),
      ...(voice ? { voice } : {}),
    }),
  };
}
