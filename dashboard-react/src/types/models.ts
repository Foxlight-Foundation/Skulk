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

/** Optional declarative tool-calling metadata copied from a model card. */
export interface ToolingCapabilityInfo {
  supports_tool_calling?: boolean;
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
  supports_tool_calling: boolean;
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
  tooling?: ToolingCapabilityInfo;
  runtime?: RuntimeCapabilityInfo;
  resolved_capabilities?: ResolvedModelCapabilities;
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
  instance_meta: 'MlxRing' | 'MlxJaccl';
  instance: unknown | null;
  memory_delta_by_node: Record<string, number> | null;
  error: string | null;
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
] as const;

export type Capability = (typeof CAPABILITIES)[number];

/** Size range presets for the filter popover. */
export const SIZE_RANGES = [
  { label: '< 10 GB', min: 0, max: 10 * 1024 },
  { label: '10–50 GB', min: 10 * 1024, max: 50 * 1024 },
  { label: '50–200 GB', min: 50 * 1024, max: 200 * 1024 },
  { label: '> 200 GB', min: 200 * 1024, max: Infinity },
] as const;

export type PickerMode = 'launch' | 'store-download';

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
      capabilities: first.capabilities ?? [],
      family: first.family ?? '',
      variants: sorted,
      smallestVariant: first,
      hasMultipleVariants: sorted.length > 1,
    };
  });
}
