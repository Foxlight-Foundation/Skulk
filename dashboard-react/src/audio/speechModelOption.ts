import type { AudioResponseFormat, ChatSpeechModelOption } from '../types/chat';
import type { ModelInfo } from '../types/models';

const AUDIO_RESPONSE_FORMATS: readonly AudioResponseFormat[] = [
  'mp3',
  'wav',
  'flac',
  'ogg',
  'opus',
  'pcm',
];

function isAudioResponseFormat(value: string | null | undefined): value is AudioResponseFormat {
  return AUDIO_RESPONSE_FORMATS.includes(value as AudioResponseFormat);
}

/** Project canonical model capability truth into one dashboard speech option. */
export function speechModelOption(
  modelId: string,
  model: ModelInfo | undefined,
): ChatSpeechModelOption {
  const resolved = model?.resolved_capabilities;
  const responseFormats = (
    resolved?.audio_response_formats
      ?? model?.audio?.response_formats
      ?? []
  ).filter(isAudioResponseFormat);
  const resolvedDefault = resolved?.default_audio_response_format ?? null;
  const cardDefault = model?.audio?.default_response_format ?? null;
  const defaultResponseFormat: AudioResponseFormat = isAudioResponseFormat(resolvedDefault)
    ? resolvedDefault
    : isAudioResponseFormat(cardDefault)
      ? cardDefault
      : responseFormats[0] ?? 'mp3';
  const formats = responseFormats.length > 0 ? responseFormats : [defaultResponseFormat];
  const parts = modelId.split('/');
  return {
    modelId,
    label: parts[parts.length - 1] || modelId,
    defaultResponseFormat,
    responseFormats: formats,
    supportsVoiceListing: model?.audio?.supports_voice_listing ?? false,
    defaultVoice: model?.audio?.default_voice ?? null,
    supportsStreaming: model?.audio?.supports_streaming ?? false,
    supportsReferenceAudio: model?.audio?.supports_reference_audio ?? false,
    supportsRealtime: Boolean(
      resolved?.supports_realtime_audio
        && model?.audio?.supports_streaming
        && model.audio.supports_realtime,
    ),
  };
}
