import {
  buildSpeechSynthesisRequest,
  DASHBOARD_SPEECH_SEED,
} from './speechSynthesisRequest';

/** Product voice id reserved for Skulk's operator-facing fabric cognition. */
export const SKULK_VOICE_ID = 'skulk';

/** Build one sentence request with the product voice pinned explicitly. */
export function buildSkulkSpeechSynthesisRequest(
  modelId: string,
  input: string,
  language: string,
  signal: AbortSignal,
): RequestInit {
  return buildSpeechSynthesisRequest({
    model: modelId,
    input,
    language,
    responseFormat: 'pcm',
    stream: true,
    maxTokens: null,
    seed: DASHBOARD_SPEECH_SEED,
    voice: SKULK_VOICE_ID,
    referenceAudio: null,
    referenceText: '',
    signal,
  });
}
