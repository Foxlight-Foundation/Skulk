/**
 * Models Store
 *
 * Manages available model metadata fetched from /models.
 * Used by the model picker, placement previews, and chat form.
 */
import { create } from 'zustand';

export interface ModelTask {
  name: string;
}

export interface ModelEntry {
  id: string;
  name?: string;
  family?: string;
  base_model?: string;
  num_params?: number;
  context_length?: number;
  quantization?: string;
  tasks?: string[];
  capabilities?: string[];
  /** Size in bytes */
  sizeBytes?: number;
  /** Whether the model is fully downloaded on the cluster */
  isDownloaded?: boolean;
  /** Whether a download is in progress */
  isDownloading?: boolean;
  /** Download progress 0-100 */
  downloadProgress?: number;
}

interface ModelsState {
  models: ModelEntry[];
  isLoading: boolean;
  lastFetchedAt: number | null;

  fetchModels: () => Promise<void>;
  getModelById: (id: string) => ModelEntry | undefined;
}

export const useModelsStore = create<ModelsState>((set, get) => ({
  models: [],
  isLoading: false,
  lastFetchedAt: null,

  fetchModels: async () => {
    set({ isLoading: true });
    try {
      const res = await fetch('/models');
      if (!res.ok) throw new Error(`/models ${res.status}`);
      const data = (await res.json()) as ModelEntry[] | { models?: ModelEntry[] };
      const models: ModelEntry[] = Array.isArray(data)
        ? data
        : (data.models ?? []);
      set({ models, isLoading: false, lastFetchedAt: Date.now() });
    } catch {
      set({ isLoading: false });
    }
  },

  getModelById: (id) => get().models.find((m) => m.id === id),
}));
