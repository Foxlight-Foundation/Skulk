import type { ChatVoiceOption } from '../types/chat';

interface VoiceCatalogPayload {
  data?: unknown;
}

interface VoiceCatalogEntryPayload {
  id?: unknown;
  name?: unknown;
  preferred_languages?: unknown;
}

/** Load and validate one mounted model's stable voice catalog. */
export async function fetchSpeechVoiceCatalog(
  modelId: string,
  signal?: AbortSignal,
  fetcher: typeof fetch = fetch,
): Promise<ChatVoiceOption[]> {
  const response = await fetcher(
    `/v1/audio/voices?model=${encodeURIComponent(modelId)}`,
    { signal },
  );
  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(detail || `Voice discovery failed with HTTP ${response.status}.`);
  }
  const payload = await response.json() as VoiceCatalogPayload;
  if (!Array.isArray(payload.data)) {
    throw new Error('Voice discovery returned an invalid catalog.');
  }
  return payload.data.map((value) => {
    const entry = value as VoiceCatalogEntryPayload;
    if (typeof entry.id !== 'string' || entry.id.trim().length === 0) {
      throw new Error('Voice discovery returned an entry without an id.');
    }
    const preferredLanguages = Array.isArray(entry.preferred_languages)
      ? entry.preferred_languages.filter(
          (language): language is string => typeof language === 'string' && language.length > 0,
        )
      : [];
    return {
      id: entry.id,
      name: typeof entry.name === 'string' && entry.name.trim().length > 0
        ? entry.name
        : entry.id,
      preferredLanguages,
    };
  });
}

/** Infer the primary speech language conservatively from the response script. */
export function inferSpeechLanguage(text: string): string | null {
  if (/[\uac00-\ud7af]/u.test(text)) return 'ko';
  if (/[\u3040-\u30ff]/u.test(text)) return 'ja';
  if (/[\u3400-\u4dbf\u4e00-\u9fff]/u.test(text)) return 'zh';
  if (/[\u0400-\u052f]/u.test(text)) return 'ru';
  return null;
}

function primaryLanguage(languageTag: string): string {
  return languageTag.trim().toLowerCase().replace('_', '-').split('-')[0];
}

function containsLatinScript(text: string): boolean {
  return /[A-Za-z\u00c0-\u024f]/u.test(text);
}

function usesLatinScript(languageTag: string): boolean {
  try {
    return new Intl.Locale(languageTag).maximize().script === 'Latn';
  } catch {
    return false;
  }
}

function soleCatalogLatinLanguage(voices: readonly ChatVoiceOption[]): string | null {
  const languages = new Set<string>();
  for (const voice of voices) {
    for (const language of voice.preferredLanguages) {
      if (usesLatinScript(language)) languages.add(primaryLanguage(language));
    }
  }
  return languages.size === 1 ? languages.values().next().value ?? null : null;
}

/** Resolve an explicit or automatic voice for one complete textual response. */
export function selectSpeechVoice(
  text: string,
  voices: readonly ChatVoiceOption[],
  explicitVoice: string | null,
  defaultVoice: string | null,
): string | null {
  if (explicitVoice) return explicitVoice;
  const inferredLanguage = inferSpeechLanguage(text);
  const language = inferredLanguage ?? (
    containsLatinScript(text) ? soleCatalogLatinLanguage(voices) : null
  );
  if (language) {
    const match = voices.find((voice) => voice.preferredLanguages.some(
      (candidate) => primaryLanguage(candidate) === language,
    ));
    if (match) return match.id;
  }
  if (defaultVoice && voices.some((voice) => voice.id === defaultVoice)) {
    return defaultVoice;
  }
  return voices[0]?.id ?? defaultVoice;
}

/**
 * Create a response-scoped selector that pins its first result.
 *
 * Sentence streaming invokes TTS once per sentence, so resolving on every call
 * could change speakers mid-response. This closure guarantees every request in
 * one playback session carries the same voice string.
 */
export function createPinnedSpeechVoiceSelector(
  voices: readonly ChatVoiceOption[],
  explicitVoice: string | null,
  defaultVoice: string | null,
): (text: string) => string | null {
  let pinnedVoice: string | null | undefined;
  return (text) => {
    if (pinnedVoice === undefined) {
      pinnedVoice = selectSpeechVoice(text, voices, explicitVoice, defaultVoice);
    }
    return pinnedVoice;
  };
}
