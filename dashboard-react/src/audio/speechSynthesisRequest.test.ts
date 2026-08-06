import { describe, expect, it } from 'vitest';

import {
  batchSpeechMaxTokens,
  buildSpeechSynthesisRequest,
  DASHBOARD_SPEECH_SEED,
  MAX_REFERENCE_AUDIO_BYTES,
  speechLanguageForDashboardLocale,
} from './speechSynthesisRequest';

describe('buildSpeechSynthesisRequest', () => {
  it('keeps ordinary speech requests on the JSON contract', () => {
    const controller = new AbortController();
    const request = buildSpeechSynthesisRequest({
      model: 'org/tts',
      input: 'Hello',
      language: 'English',
      responseFormat: 'mp3',
      stream: false,
      maxTokens: 8192,
      seed: DASHBOARD_SPEECH_SEED,
      voice: 'ryan',
      referenceAudio: null,
      referenceText: '',
      signal: controller.signal,
    });

    expect(request.headers).toEqual({ 'Content-Type': 'application/json' });
    expect(JSON.parse(request.body as string)).toEqual({
      model: 'org/tts',
      input: 'Hello',
      lang_code: 'English',
      response_format: 'mp3',
      seed: DASHBOARD_SPEECH_SEED,
      max_tokens: 8192,
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
      language: 'English',
      responseFormat: 'pcm',
      stream: true,
      maxTokens: null,
      seed: DASHBOARD_SPEECH_SEED,
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
    expect(body.get('lang_code')).toBe('English');
    expect(body.get('response_format')).toBe('pcm');
    expect(body.get('seed')).toBe(String(DASHBOARD_SPEECH_SEED));
    expect(body.get('stream')).toBe('true');
    expect(body.has('voice')).toBe(false);
    const uploadedReference = body.get('reference_audio');
    expect(uploadedReference).toBeInstanceOf(File);
    expect((uploadedReference as File).name).toBe(referenceAudio.name);
    expect((uploadedReference as File).type).toBe(referenceAudio.type);
    expect((uploadedReference as File).size).toBe(referenceAudio.size);
    expect(body.get('reference_text')).toBe('Reference transcript.');
  });

  it('uses uploaded reference audio instead of a catalog voice when both UI states exist', () => {
    const controller = new AbortController();
    const referenceAudio = new File(['RIFF-reference'], 'reference.wav', {
      type: 'audio/wav',
    });
    const request = buildSpeechSynthesisRequest({
      model: 'org/voice-clone',
      input: 'First sentence.',
      language: 'English',
      responseFormat: 'wav',
      stream: false,
      maxTokens: 8192,
      seed: DASHBOARD_SPEECH_SEED,
      voice: 'angus',
      referenceAudio,
      referenceText: 'Reference transcript.',
      signal: controller.signal,
    });

    const body = request.body as FormData;
    expect(body.has('voice')).toBe(false);
    expect(body.get('max_tokens')).toBe('8192');
    expect(body.get('reference_audio')).toBeInstanceOf(File);
  });

  it('mirrors the server upload bound', () => {
    expect(MAX_REFERENCE_AUDIO_BYTES).toBe(26_214_400);
  });

  it('scales batch synthesis budgets beyond the short server default', () => {
    expect(batchSpeechMaxTokens('brief')).toBe(4096);
    expect(batchSpeechMaxTokens('x'.repeat(500))).toBe(16_000);
    expect(batchSpeechMaxTokens('x'.repeat(10_000))).toBe(131_072);
  });

  it('maps the dashboard locale to the TTS model language vocabulary', () => {
    expect(speechLanguageForDashboardLocale('en-US')).toBe('English');
    expect(speechLanguageForDashboardLocale('pt_BR')).toBe('Portuguese');
    expect(speechLanguageForDashboardLocale(undefined)).toBe('English');
  });
});
