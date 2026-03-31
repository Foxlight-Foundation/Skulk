import { useCallback, useMemo, useState } from 'react';
import {
  groupModels,
  EMPTY_FILTERS,
  type ModelInfo,
  type ModelGroup,
  type FilterState,
  type ModelFitStatus,
  type DownloadAvailability,
  type InstanceStatus,
} from '../types/models';

/** Inputs used to build the dashboard's model-picker state machine. */
export interface UseModelPickerOptions {
  models: ModelInfo[];
  favorites: Set<string>;
  recentModelIds?: string[];
  canModelFit: (id: string) => boolean;
  getModelFitStatus: (id: string) => ModelFitStatus;
  downloadStatusMap?: Map<string, DownloadAvailability>;
  instanceStatuses?: Record<string, InstanceStatus>;
}

/** State, setters, and derived groups exposed by {@link useModelPicker}. */
export interface UseModelPickerReturn {
  /* state */
  searchQuery: string;
  selectedFamily: string | null;
  expandedGroups: Set<string>;
  filters: FilterState;
  showFilters: boolean;

  /* setters */
  setSearchQuery: (q: string) => void;
  setSelectedFamily: (f: string | null) => void;
  toggleExpanded: (groupId: string) => void;
  setFilters: (f: FilterState) => void;
  clearFilters: () => void;
  setShowFilters: (v: boolean) => void;

  /* derived */
  uniqueFamilies: string[];
  hasFavorites: boolean;
  hasRecents: boolean;
  /** Groups that fit on cluster. */
  recommendedGroups: ModelGroup[];
  /** Groups that don't fit. */
  otherGroups: ModelGroup[];
  /** All filtered groups (recommended + other). */
  filteredGroups: ModelGroup[];
  /** Recent model groups. */
  recentGroups: ModelGroup[];
}

/** Manage model-picker filtering, grouping, favorites, recents, and fit-based recommendations. */
export function useModelPicker({
  models,
  favorites,
  recentModelIds = [],
  canModelFit,
  getModelFitStatus,
  downloadStatusMap,
  instanceStatuses,
}: UseModelPickerOptions): UseModelPickerReturn {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFamily, setSelectedFamily] = useState<string | null>(null);
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const [filters, setFilters] = useState<FilterState>({ ...EMPTY_FILTERS });
  const [showFilters, setShowFilters] = useState(false);

  const toggleExpanded = useCallback((groupId: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupId)) next.delete(groupId);
      else next.add(groupId);
      return next;
    });
  }, []);

  const clearFilters = useCallback(() => setFilters({ ...EMPTY_FILTERS }), []);

  // Grouped models
  const allGroups = useMemo(() => groupModels(models), [models]);

  // Unique families
  const uniqueFamilies = useMemo(() => {
    const set = new Set<string>();
    for (const m of models) {
      if (m.family) set.add(m.family);
    }
    return Array.from(set).sort();
  }, [models]);

  const hasFavorites = useMemo(() => {
    return allGroups.some((g) => favorites.has(g.id));
  }, [allGroups, favorites]);

  const hasRecents = recentModelIds.length > 0;

  // Recent groups
  const recentGroups = useMemo(() => {
    if (!recentModelIds.length) return [];
    const recentSet = new Set(recentModelIds);
    return allGroups.filter((g) => g.variants.some((v) => recentSet.has(v.id)));
  }, [allGroups, recentModelIds]);

  // Filtering pipeline
  const filteredGroups = useMemo(() => {
    let groups = allGroups;
    const q = searchQuery.toLowerCase().trim();

    // Family filter
    if (selectedFamily === 'favorites') {
      groups = groups.filter((g) => favorites.has(g.id));
    } else if (selectedFamily === 'recents') {
      return recentGroups;
    } else if (selectedFamily === 'huggingface') {
      return []; // HF results are handled separately
    } else if (selectedFamily) {
      groups = groups.filter((g) => g.family === selectedFamily);
    }

    // Text search
    if (q) {
      groups = groups.filter(
        (g) =>
          g.name.toLowerCase().includes(q) ||
          g.variants.some((v) => v.id.toLowerCase().includes(q)),
      );
    }

    // Capability filter (AND)
    if (filters.capabilities.length > 0) {
      groups = groups.filter((g) =>
        filters.capabilities.every((c) => g.capabilities.includes(c)),
      );
    }

    // Size range filter
    if (filters.sizeRange) {
      const { min, max } = filters.sizeRange;
      groups = groups.filter((g) =>
        g.variants.some((v) => {
          const mb = v.storage_size_megabytes ?? 0;
          return mb >= min && mb <= max;
        }),
      );
    }

    // Downloaded only
    if (filters.downloadedOnly && downloadStatusMap) {
      groups = groups.filter((g) =>
        g.variants.some((v) => downloadStatusMap.get(v.id)?.available),
      );
    }

    // Ready only
    if (filters.readyOnly && instanceStatuses) {
      groups = groups.filter((g) =>
        g.variants.some((v) => instanceStatuses[v.id]?.statusClass === 'ready'),
      );
    }

    return groups;
  }, [
    allGroups,
    searchQuery,
    selectedFamily,
    favorites,
    recentGroups,
    filters,
    downloadStatusMap,
    instanceStatuses,
  ]);

  // Split into recommended (fits) and other
  const { recommendedGroups, otherGroups } = useMemo(() => {
    const rec: ModelGroup[] = [];
    const other: ModelGroup[] = [];
    for (const g of filteredGroups) {
      const fits = g.variants.some((v) => getModelFitStatus(v.id) === 'fits_now');
      if (fits) rec.push(g);
      else other.push(g);
    }
    return { recommendedGroups: rec, otherGroups: other };
  }, [filteredGroups, getModelFitStatus]);

  return {
    searchQuery,
    selectedFamily,
    expandedGroups,
    filters,
    showFilters,
    setSearchQuery,
    setSelectedFamily,
    toggleExpanded,
    setFilters,
    clearFilters,
    setShowFilters,
    uniqueFamilies,
    hasFavorites,
    hasRecents,
    recommendedGroups,
    otherGroups,
    filteredGroups,
    recentGroups,
  };
}
