import { describe, expect, it, vi } from 'vitest';

import type { ChatVoiceOption } from '../types/chat';
import {
  createPinnedSpeechVoiceSelector,
  fetchSpeechVoiceCatalog,
  inferSpeechLanguage,
  selectSpeechVoice,
} from './speechVoiceSelection';

const VOICES: ChatVoiceOption[] = [
  { id: 'serena', name: 'Serena', preferredLanguages: ['zh'] },
  { id: 'ryan', name: 'Ryan', preferredLanguages: ['en-US'] },
  { id: 'aiden', name: 'Aiden', preferredLanguages: ['en'] },
  { id: 'ono_anna', name: 'Ono Anna', preferredLanguages: ['ja'] },
];

describe('speech voice discovery', () => {
  it('loads normalized language metadata for one encoded model id', async () => {
    const fetcher = vi.fn(async () => new Response(JSON.stringify({
      object: 'list',
      data: [{
        id: 'ryan',
        name: 'Ryan',
        preferred_languages: ['en'],
      }],
    }), { status: 200 })) as unknown as typeof fetch;

    await expect(fetchSpeechVoiceCatalog('org/model name', undefined, fetcher)).resolves.toEqual([
      { id: 'ryan', name: 'Ryan', preferredLanguages: ['en'] },
    ]);
    expect(fetcher).toHaveBeenCalledWith(
      '/v1/audio/voices?model=org%2Fmodel%20name',
      { signal: undefined },
    );
  });
});

describe('speech language and voice selection', () => {
  it('recognizes the scripts represented by the bundled voice catalog', () => {
    expect(inferSpeechLanguage('An English response.')).toBeNull();
    expect(inferSpeechLanguage('这是中文。')).toBe('zh');
    expect(inferSpeechLanguage('これは日本語です。')).toBe('ja');
    expect(inferSpeechLanguage('한국어 응답입니다.')).toBe('ko');
  });

  it('selects the first discovered voice matching the response language', () => {
    expect(selectSpeechVoice('This is English.', VOICES, null, 'serena')).toBe('ryan');
    expect(selectSpeechVoice('これは日本語です。', VOICES, null, 'serena')).toBe('ono_anna');
  });

  it('prefers the card default among voices matching the response language', () => {
    expect(selectSpeechVoice('This is English.', VOICES, null, 'aiden')).toBe('aiden');
  });

  it('does not misclassify Latin text when the catalog has multiple Latin languages', () => {
    const multilingualVoices: ChatVoiceOption[] = [
      ...VOICES,
      { id: 'lucia', name: 'Lucia', preferredLanguages: ['es'] },
    ];

    expect(selectSpeechVoice('Una respuesta en español.', multilingualVoices, null, 'serena'))
      .toBe('serena');
  });

  it('pins one voice for every sentence in the playback session', () => {
    const selectVoice = createPinnedSpeechVoiceSelector(VOICES, null, 'serena');

    expect(selectVoice('The first sentence is English.')).toBe('ryan');
    expect(selectVoice('这是后来的中文句子。')).toBe('ryan');
    expect(selectVoice('A third sentence.')).toBe('ryan');
  });

  it('keeps an explicit user selection regardless of response language', () => {
    const selectVoice = createPinnedSpeechVoiceSelector(VOICES, 'aiden', 'serena');

    expect(selectVoice('这是中文。')).toBe('aiden');
    expect(selectVoice('This is English.')).toBe('aiden');
  });
});
