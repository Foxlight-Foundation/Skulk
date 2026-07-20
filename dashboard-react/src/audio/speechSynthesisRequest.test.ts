import { describe, expect, it } from 'vitest';

import {
  buildSpeechSynthesisRequest,
  MAX_REFERENCE_AUDIO_BYTES,
} from './speechSynthesisRequest';

describe('buildSpeechSynthesisRequest', () => {
  it('keeps ordinary speech requests on the JSON contract', () => {
    const controller = new AbortController();
    const request = buildSpeechSynthesisRequest({
      model: 'org/tts',
      input: 'Hello',
      responseFormat: 'mp3',
      stream: false,
      voice: 'ryan',
      referenceAudio: null,
      referenceText: '',
      signal: controller.signal,
    });

    expect(request.headers).toEqual({ 'Content-Type': 'application/json' });
    expect(JSON.parse(request.body as string)).toEqual({
      model: 'org/tts',
      input: 'Hello',
      response_format: 'mp3',
      voice: 'ryan',
    });
  });

  it('uses multipart and preserves the selected clip for streaming reference audio', () => {
    const controller = new AbortController();
    const referenceAudio = new File(['RIFF-reference'], 'reference.wav', {
      type: 'audio/wav',
    });
    const request = buildSpeechSynthesisRequest({
      model: 'org/voice-clone',
      input: 'First sentence.',
      responseFormat: 'pcm',
      stream: true,
      voice: null,
      referenceAudio,
      referenceText: '  Reference transcript.  ',
      signal: controller.signal,
    });

    expect(request.headers).toBeUndefined();
    expect(request.body).toBeInstanceOf(FormData);
    const body = request.body as FormData;
    expect(body.get('model')).toBe('org/voice-clone');
    expect(body.get('input')).toBe('First sentence.');
    expect(body.get('response_format')).toBe('pcm');
    expect(body.get('stream')).toBe('true');
    const uploadedReference = body.get('reference_audio');
    expect(uploadedReference).toBeInstanceOf(File);
    expect((uploadedReference as File).name).toBe(referenceAudio.name);
    expect((uploadedReference as File).type).toBe(referenceAudio.type);
    expect((uploadedReference as File).size).toBe(referenceAudio.size);
    expect(body.get('reference_text')).toBe('Reference transcript.');
  });

  it('mirrors the server upload bound', () => {
    expect(MAX_REFERENCE_AUDIO_BYTES).toBe(26_214_400);
  });
});
