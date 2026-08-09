/** Dashboard-friendly model metadata derived from the Skulk model catalog. */
export interface ReasoningCapabilityInfo {
  supports_toggle?: boolean;
  supports_budget?: boolean;
  format?: string;
  default_effort?: string;
  disabled_effort?: string;
}

/** Optional declarative modality metadata copied from a model card. */
export interface ModalitiesCapabilityInfo {
  supports_audio_input?: boolean;
  supports_native_multimodal?: boolean;
}

/** Optional declarative speech metadata copied from a model card. */
export interface AudioCapabilityInfo {
  kind?: string;
  default_response_format?: string;
  response_formats?: string[];
  supports_streaming?: boolean;
  supports_realtime?: boolean;
  supports_voice_listing?: boolean;
  default_voice?: string | null;
  voices?: string[];
  supports_reference_audio?: boolean;
  supports_translation?: boolean;
  sample_rates?: number[];
}

/** Optional declarative tool-calling metadata copied from a model card. */
export interface ToolingCapabilityInfo {
  supports_tool_calling?: boolean;
  builtin_tools?: string[];
  tool_call_format?: string;
}

/** Optional declarative runtime hints copied from a model card. */
export interface RuntimeCapabilityInfo {
  prompt_renderer?: string;
  output_parser?: string;
}

/** Normalized runtime capability contract returned by `/v1/models`. */
export interface ResolvedModelCapabilities {
  family: string;
  supports_thinking: boolean;
  supports_thinking_toggle: boolean;
  supports_thinking_budget: boolean;
  default_reasoning_effort: string;
  disabled_reasoning_effort: string;
  thinking_format: string;
  supports_image_input: boolean;
  supports_audio_input: boolean;
  supports_speech_synthesis: boolean;
  supports_transcription: boolean;
  supports_speech_translation: boolean;
  supports_audio_output: boolean;
  supports_realtime_audio: boolean;
  default_audio_response_format?: string | null;
  audio_response_formats: string[];
  supports_tool_calling: boolean;
  builtin_tools: string[];
  tool_call_format: string;
  prompt_renderer: string;
  output_parser: string;
  supports_native_multimodal: boolean;
}

/** Complete dashboard-facing model metadata entry returned by the model catalog. */
export interface ModelInfo {
  id: string;
  name?: string;
  context_length?: number;
  tags?: string[];
  storage_size_megabytes?: number;
  base_model?: string;
  quantization?: string;
  supports_tensor?: boolean;
  capabilities?: string[];
  family?: string;
  is_custom?: boolean;
  tasks?: string[];
  hugging_face_id?: string;
  reasoning?: ReasoningCapabilityInfo;
  modalities?: ModalitiesCapabilityInfo;
  audio?: AudioCapabilityInfo;
  tooling?: ToolingCapabilityInfo;
  runtime?: RuntimeCapabilityInfo;
  resolved_capabilities?: ResolvedModelCapabilities;
}

/** Model-card fields needed to decide whether the dashboard may open text chat. */
export interface ModelTextChatMetadata {
  tasks?: readonly string[];
  capabilities?: readonly string[];
  tags?: readonly string[];
  resolved_capabilities?: Partial<Pick<
    ResolvedModelCapabilities,
    'supports_speech_synthesis' | 'supports_transcription' | 'supports_speech_translation'
  >>;
}

/**
 * Return whether a model can be selected as the direct target of dashboard text chat.
 *
 * Text generation wins for multi-capability models. Speech-only and embedding-only
 * models remain available to their dedicated APIs and chat voice controls without
 * being offered as `/v1/chat/completions` targets.
 */
export function modelSupportsTextChat(model: ModelTextChatMetadata | undefined): boolean {
  if (!model) return true;

  const tasks = new Set(model.tasks ?? []);
  if (tasks.has('TextGeneration')) return true;
  if (tasks.has('TextEmbedding')) return false;
  if (
    tasks.has('TextToSpeech')
    || tasks.has('SpeechToText')
    || tasks.has('SpeechTranslation')
  ) {
    return false;
  }

  const capabilities = new Set([...(model.capabilities ?? []), ...(model.tags ?? [])]);
  if (capabilities.has('embedding')) return false;
  if (capabilities.has('text')) return true;

  const resolved = model.resolved_capabilities;
  return !(
    capabilities.has('tts')
    || capabilities.has('stt')
    || resolved?.supports_speech_synthesis
    || resolved?.supports_transcription
    || resolved?.supports_speech_translation
  );
}

/** Group of related model variants shown as one family in the picker UI. */
export interface ModelGroup {
  id: string;
  name: string;
  capabilities: string[];
  family: string;
  variants: ModelInfo[];
  smallestVariant: ModelInfo;
  hasMultipleVariants: boolean;
}

/** Filter state for the dashboard model picker. */
export interface FilterState {
  capabilities: string[];
  sizeRange: { min: number; max: number } | null;
  downloadedOnly: boolean;
  readyOnly: boolean;
}

export const EMPTY_FILTERS: FilterState = {
  capabilities: [],
  sizeRange: null,
  downloadedOnly: false,
  readyOnly: false,
};

export type ModelFitStatus = 'fits_now' | 'fits_cluster_capacity' | 'too_large';

/** Availability of a model across nodes or store-backed downloads. */
export interface DownloadAvailability {
  available: boolean;
  nodeNames: string[];
  nodeIds: string[];
}

/** UI-friendly summary of whether a model is already launched and ready. */
export interface InstanceStatus {
  status: string;
  statusClass: string;
}

/** Lightweight search result returned by the Hugging Face search API. */
export interface HuggingFaceModel {
  id: string;
  author: string;
  downloads: number;
  likes: number;
  last_modified: string;
  tags: string[];
  /** Exact repo-relative GGUF path matched by a filename search. */
  matched_file?: string | null;
  /** Hugging Face task tag (text-generation, image-text-to-text, ...). */
  pipeline_tag?: string | null;
  /** Framework the repository targets (transformers, diffusers, mlx, gguf). */
  library_name?: string | null;
  /** True when the license must be accepted and a token presented to download. */
  gated?: boolean;
  /** License identifier from the model card. */
  license?: string | null;
  /** Total parameter count from safetensors/GGUF metadata. */
  param_count?: number | null;
  /** Exact total artifact bytes from GGUF metadata. */
  total_file_size?: number | null;
  /** Context window from GGUF metadata. */
  context_length?: number | null;
  /** Parent repository this model derives from, when tagged. */
  base_model_repo?: string | null;
  /** Derivation kind: finetune, quantized, merge, or adapter. */
  base_model_relation?: string | null;
  /** arXiv paper identifiers tagged on the repository. */
  arxiv_ids?: string[];
  /** ISO 639-1 language tags declared on the repository. */
  languages?: string[];
  /** Model architecture from repository config or GGUF metadata. */
  architecture?: string | null;
}

/** Progress snapshot for a download shown in the dashboard. */
export interface DownloadProgress {
  totalBytes: number;
  downloadedBytes: number;
  speed: number;
  etaMs: number;
  percentage: number;
  completedFiles: number;
  totalFiles: number;
  files: Array<{
    name: string;
    totalBytes: number;
    downloadedBytes: number;
  }>;
}

/** Placement preview returned by the Skulk placement preview endpoint. */
export interface PlacementPreview {
  model_id: string;
  sharding: 'Pipeline' | 'Tensor';
  instance_meta: 'MlxRing' | 'MlxJaccl' | 'LlamaRpc';
  instance: unknown | null;
  memory_delta_by_node: Record<string, number> | null;
  error: string | null;
  /** Per-host alternative to the ranked pick: a single-node placement on a
   * host that passes admission but lost the planner ranking (#557). */
  alternative?: boolean;
}

/** All known capability tags. */
export const CAPABILITIES = [
  'text',
  'thinking',
  'code',
  'vision',
  'image_gen',
  'image_edit',
  'embedding',
  'tts',
  'stt',
] as const;

export type Capability = (typeof CAPABILITIES)[number];

/** Size range presets for the filter popover. */
export const SIZE_RANGES = [
  { min: 0, max: 10 * 1024 },
  { min: 10 * 1024, max: 50 * 1024 },
  { min: 50 * 1024, max: 200 * 1024 },
  { min: 200 * 1024, max: Infinity },
] as const;

export type PickerMode = 'launch' | 'store-download';

function isKnownCapability(value: string): value is Capability {
  return (CAPABILITIES as readonly string[]).includes(value);
}

function modelFilterCapabilities(model: ModelInfo): string[] {
  const capabilities = new Set<string>();
  for (const value of model.capabilities ?? []) {
    if (isKnownCapability(value)) capabilities.add(value);
  }
  for (const value of model.tags ?? []) {
    if (isKnownCapability(value)) capabilities.add(value);
  }
  const resolved = model.resolved_capabilities;
  if (resolved?.supports_speech_synthesis) capabilities.add('tts');
  if (resolved?.supports_transcription || resolved?.supports_speech_translation) {
    capabilities.add('stt');
  }
  return Array.from(capabilities);
}

/**
 * Group model variants by base model (or model id if no base model is present).
 * Variants are sorted by size ascending so the UI can show the smallest representative first.
 */
export function groupModels(models: ModelInfo[]): ModelGroup[] {
  const map = new Map<string, ModelInfo[]>();
  for (const m of models) {
    const key = m.base_model || m.id;
    const existing = map.get(key);
    if (existing) existing.push(m);
    else map.set(key, [m]);
  }

  return Array.from(map.entries()).map(([key, variants]) => {
    const sorted = [...variants].sort(
      (a, b) => (a.storage_size_megabytes ?? 0) - (b.storage_size_megabytes ?? 0),
    );
    const first = sorted[0];
    return {
      id: key,
      name: first.name ?? first.id,
      capabilities: Array.from(
        new Set(sorted.flatMap((variant) => modelFilterCapabilities(variant))),
      ),
      family: first.family ?? '',
      variants: sorted,
      smallestVariant: first,
      hasMultipleVariants: sorted.length > 1,
    };
  });
}
