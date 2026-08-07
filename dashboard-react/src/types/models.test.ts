import { describe, expect, it } from 'vitest';

import { modelSupportsTextChat } from './models';

describe('modelSupportsTextChat', () => {
  it('allows text-generation models, including models with additional speech capability', () => {
    expect(modelSupportsTextChat({ tasks: ['TextGeneration'] })).toBe(true);
    expect(modelSupportsTextChat({ tasks: ['TextGeneration', 'TextToSpeech'] })).toBe(true);
  });

  it('rejects dedicated speech models as direct text-chat targets', () => {
    expect(modelSupportsTextChat({ tasks: ['TextToSpeech'] })).toBe(false);
    expect(modelSupportsTextChat({ tasks: ['SpeechToText'] })).toBe(false);
    expect(modelSupportsTextChat({ tasks: ['SpeechTranslation'] })).toBe(false);
    expect(modelSupportsTextChat({ tags: ['tts'] })).toBe(false);
    expect(modelSupportsTextChat({
      resolved_capabilities: { supports_speech_synthesis: true },
    })).toBe(false);
  });

  it('rejects embeddings while preserving legacy models without capability metadata', () => {
    expect(modelSupportsTextChat({ tasks: ['TextEmbedding'] })).toBe(false);
    expect(modelSupportsTextChat({ capabilities: ['embedding'] })).toBe(false);
    expect(modelSupportsTextChat(undefined)).toBe(true);
    expect(modelSupportsTextChat({})).toBe(true);
  });
});
