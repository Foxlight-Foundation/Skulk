import type { ChatSpeechModelOption, ChatVoiceOption } from '../types/chat';
import type { ModelInfo } from '../types/models';
import { SKULK_VOICE_ID } from './fabricSpeechRequest';
import { speechModelOption } from './speechModelOption';

/** A ready speech model paired with Skulk's product voice. */
export interface SkulkSpeechSelection {
  model: ChatSpeechModelOption;
  voice: ChatVoiceOption;
}

/** Find a ready model whose live voice catalog contains Skulk's product voice. */
export async function discoverSkulkSpeechSelection(
  models: readonly ModelInfo[],
  readyModelIds: ReadonlySet<string>,
  loadVoices: (modelId: string) => Promise<ChatVoiceOption[]>,
  playbackAvailable: boolean,
): Promise<SkulkSpeechSelection | null> {
  if (!playbackAvailable) return null;
  for (const model of models) {
    if (
      !readyModelIds.has(model.id)
      || model.resolved_capabilities?.supports_speech_synthesis !== true
    ) continue;
    const option = speechModelOption(model.id, model);
    if (
      !option.supportsStreaming
      || !option.supportsVoiceListing
      || !option.responseFormats.includes('pcm')
    ) continue;
    let voices: ChatVoiceOption[];
    try {
      voices = await loadVoices(model.id);
    } catch {
      continue;
    }
    const voice = voices.find((candidate) => candidate.id === SKULK_VOICE_ID);
    if (voice) return { model: option, voice };
  }
  return null;
}
