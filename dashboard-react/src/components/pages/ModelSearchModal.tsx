import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';
import { FiX } from 'react-icons/fi';
import { ModelBrowser } from '../models/ModelBrowser';
import { burstVerdict, hfWeightBytes, type BurstInfo, type FleetServingSummary } from '../models/burst';
import { deriveFormatLabel, deriveQuantLabel } from '../models/quantBadge';
import type { ModelInfo, HuggingFaceModel, DownloadAvailability } from '../../types/models';
import { addToast } from '../../hooks/useToast';
import { useSkulkTranslation } from '../../i18n/tolgee';

const FAVORITES_KEY = 'skulk-favorite-models';
const RECENTS_KEY = 'skulk-recent-models';
const MAX_RECENT_MODELS = 20;

/**
 * One-time migration of pre-rename (exo-*) localStorage keys so existing
 * users keep their favorites and recents across the 2026-06 exo -> skulk
 * rename. Safe to remove a few releases after launch.
 */
function migrateLegacyKey(newKey: string, legacyKey: string): void {
  if (typeof window === 'undefined') return;
  try {
    if (window.localStorage.getItem(newKey) === null) {
      const legacy = window.localStorage.getItem(legacyKey);
      if (legacy !== null) window.localStorage.setItem(newKey, legacy);
    }
    window.localStorage.removeItem(legacyKey);
  } catch {
    // localStorage unavailable — nothing to migrate.
  }
}
migrateLegacyKey(FAVORITES_KEY, 'exo-favorite-models');
migrateLegacyKey(RECENTS_KEY, 'exo-recent-models');

interface RecentEntry {
  modelId: string;
  launchedAt: number;
}

function loadFavorites(): Set<string> {
  if (typeof window === 'undefined') return new Set();
  try {
    const stored = window.localStorage.getItem(FAVORITES_KEY);
    if (!stored) return new Set();
    const parsed = JSON.parse(stored) as unknown;
    if (!Array.isArray(parsed)) return new Set();
    return new Set(parsed.filter((value): value is string => typeof value === 'string'));
  } catch {
    return new Set();
  }
}

function loadRecentIds(): string[] {
  if (typeof window === 'undefined') return [];
  try {
    const stored = window.localStorage.getItem(RECENTS_KEY);
    if (!stored) return [];
    const parsed = JSON.parse(stored) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed
      .flatMap((entry) => {
        if (typeof entry === 'string') return [entry];
        if (
          entry
          && typeof entry === 'object'
          && typeof (entry as RecentEntry).modelId === 'string'
        ) {
          return [(entry as RecentEntry).modelId];
        }
        return [];
      })
      .slice(0, MAX_RECENT_MODELS);
  } catch {
    return [];
  }
}

interface ModelSearchModalProps {
  open: boolean;
  onClose: () => void;
  existingModelIds: Set<string>;
  onDownloadStarted: () => void;
  /** What the local fleet can serve; enables burst partitioning when set. */
  fleet?: FleetServingSummary | null;
}

export function ModelSearchModal({
  open,
  onClose,
  existingModelIds,
  onDownloadStarted,
  fleet = null,
}: ModelSearchModalProps) {
  const { t } = useSkulkTranslation();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [favorites, setFavorites] = useState<Set<string>>(() => loadFavorites());
  const [recentIds, setRecentIds] = useState<string[]>(() => loadRecentIds());
  const [hfResults, setHfResults] = useState<HuggingFaceModel[]>([]);
  const [hfTrending, setHfTrending] = useState<HuggingFaceModel[]>([]);
  const [hfSearching, setHfSearching] = useState(false);
  const [mlxOnly, setMlxOnly] = useState(false);
  // "Show more" paging: how many results to request, and whether the last
  // response filled the request (more may exist upstream).
  const [hfLimit, setHfLimit] = useState(50);
  const [hfHasMore, setHfHasMore] = useState(false);
  const lastQueryRef = useRef('');
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      window.localStorage.setItem(FAVORITES_KEY, JSON.stringify([...favorites]));
    } catch {
      /* ignore */
    }
  }, [favorites]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const recents: RecentEntry[] = recentIds.slice(0, MAX_RECENT_MODELS).map((modelId, index) => ({
        modelId,
        launchedAt: Date.now() - index,
      }));
      window.localStorage.setItem(RECENTS_KEY, JSON.stringify(recents));
    } catch {
      /* ignore */
    }
  }, [recentIds]);

  // Fetch models on open
  useEffect(() => {
    if (!open) return;
    (async () => {
      try {
        const res = await fetch('/models');
        if (!res.ok) return;
        const data = await res.json();
        setModels(data.data ?? []);
      } catch { /* ignore */ }
    })();
  }, [open]);

  // Fetch trending whenever the modal opens, mlxOnly changes, or More is
  // requested while browsing trending.
  useEffect(() => {
    if (!open) return;
    (async () => {
      try {
        const params = new URLSearchParams({ query: '', limit: String(hfLimit), mlx_only: String(mlxOnly) });
        const res = await fetch(`/models/search?${params}`);
        if (!res.ok) return;
        const results = (await res.json()) as HuggingFaceModel[];
        setHfTrending(results);
        setHfHasMore(results.length >= hfLimit);
      } catch { /* ignore */ }
    })();
  }, [open, mlxOnly, hfLimit]);

  const handleHfSearch = useCallback((query: string, mlxOnlyOverride?: boolean, limitOverride?: number) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    // A changed query restarts paging at one page.
    let limit = limitOverride ?? hfLimit;
    if (query !== lastQueryRef.current && limitOverride === undefined) {
      limit = 50;
      setHfLimit(50);
    }
    lastQueryRef.current = query;
    if (!query.trim()) {
      setHfResults([]);
      setHfSearching(false);
      return;
    }
    setHfSearching(true);
    const useMlxOnly = mlxOnlyOverride ?? mlxOnly;
    debounceRef.current = setTimeout(async () => {
      try {
        const params = new URLSearchParams({ query, limit: String(limit), mlx_only: String(useMlxOnly) });
        const res = await fetch(`/models/search?${params}`);
        if (res.ok) {
          const results = (await res.json()) as HuggingFaceModel[];
          setHfResults(results);
          setHfHasMore(results.length >= limit);
        }
      } catch { /* ignore */ }
      finally { setHfSearching(false); }
    }, 500);
  }, [mlxOnly, hfLimit]);

  const handleHfLoadMore = useCallback(() => {
    const next = Math.min(hfLimit + 50, 200);
    if (next === hfLimit) { setHfHasMore(false); return; }
    setHfLimit(next);
    // Trending refetches via its effect; an active search refetches here.
    if (lastQueryRef.current.trim()) {
      handleHfSearch(lastQueryRef.current, undefined, next);
    }
  }, [hfLimit, handleHfSearch]);

  const handleSelect = useCallback(async (modelId: string, ggufFile?: string | null) => {
    try {
      const res = await fetch(`/store/models/${encodeURIComponent(modelId)}/download`, {
        method: 'POST',
        ...(ggufFile ? {
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ gguf_file: ggufFile }),
        } : {}),
      });
      if (res.ok) {
        addToast({
          type: 'success',
          message: t('modelSearch.toasts.downloadingToStore', 'Downloading {modelId} to store', { modelId }),
        });
        setRecentIds((prev) => [modelId, ...prev.filter((id) => id !== modelId)].slice(0, MAX_RECENT_MODELS));
        onDownloadStarted();
      } else {
        addToast({
          type: 'error',
          message: t('modelSearch.toasts.downloadStartFailedForModel', 'Failed to start download for {modelId}', { modelId }),
        });
      }
    } catch {
      addToast({ type: 'error', message: t('modelSearch.toasts.downloadStartFailed', 'Failed to start download') });
    }
  }, [onDownloadStarted, t]);

  const handleAddModel = useCallback(async (modelId: string, ggufFile?: string | null): Promise<boolean> => {
    try {
      const res = await fetch('/models/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId, gguf_file: ggufFile ?? null }),
      });
      if (res.ok) {
        addToast({ type: 'success', message: t('modelSearch.toasts.addedModel', 'Added {modelId}', { modelId }) });
        // Refresh model list
        const listRes = await fetch('/models');
        if (listRes.ok) {
          const data = await listRes.json();
          setModels(data.data ?? []);
        }
        return true;
      }
      addToast({ type: 'error', message: t('modelSearch.toasts.addModelFailed', 'Failed to add {modelId}', { modelId }) });
    } catch {
      addToast({ type: 'error', message: t('modelSearch.toasts.addModelFailed', 'Failed to add {modelId}', { modelId }) });
    }
    return false;
  }, [t]);

  const handleToggleMlxOnly = useCallback(() => {
    setMlxOnly(prev => !prev);
    setHfResults([]);
  }, []);

  const toggleFavorite = useCallback((groupId: string) => {
    setFavorites(prev => {
      const next = new Set(prev);
      if (next.has(groupId)) next.delete(groupId);
      else next.add(groupId);
      return next;
    });
  }, []);

  // Burst verdicts: catalog variants judge by card-truth size and format;
  // Hugging Face results fall back to a name-derived size estimate.
  const modelById = useMemo(() => {
    const map = new Map<string, ModelInfo>();
    for (const m of models) map.set(m.id, m);
    return map;
  }, [models]);

  const getBurstInfo = useCallback((variantId: string): BurstInfo | null => {
    if (!fleet) return null;
    const info = modelById.get(variantId);
    const weightBytes = info?.storage_size_megabytes
      ? info.storage_size_megabytes * 1024 * 1024
      : null;
    return burstVerdict(fleet, weightBytes, false, deriveFormatLabel(variantId));
  }, [fleet, modelById]);

  const getHfBurstInfo = useCallback((model: HuggingFaceModel): BurstInfo | null => {
    if (!fleet) return null;
    const quant = deriveQuantLabel(model.id, model.matched_file, model.tags);
    const format = deriveFormatLabel(model.id, model.matched_file, model.tags);
    const weight = hfWeightBytes(model, quant);
    return burstVerdict(fleet, weight?.bytes ?? null, weight?.estimated ?? true, format);
  }, [fleet]);

  // Build a download status map so models already in store show a checkmark
  const storeDownloadMap = useMemo(() => {
    const map = new Map<string, DownloadAvailability>();
    for (const id of existingModelIds) {
      map.set(id, { available: true, nodeNames: ['store'], nodeIds: [] });
    }
    return map;
  }, [existingModelIds]);

  if (!open) return null;

  return (
    <>
      <Backdrop onClick={onClose} />
      <ModalContainer role="dialog" aria-modal="true" aria-labelledby="model-search-title">
        <ModalHeader>
          <ModalTitle id="model-search-title">{t('modelSearch.title', 'Find Models')}</ModalTitle>
          <CloseButton onClick={onClose} aria-label={t('common.close', 'Close')}>
            <FiX size={20} />
          </CloseButton>
        </ModalHeader>
        <ModalBody>
          <ModelBrowser
            models={models}
            selectedModelId={null}
            favorites={favorites}
            recentModelIds={recentIds}
            existingModelIds={existingModelIds}
            downloadStatusMap={storeDownloadMap}
            canModelFit={() => true}
            getModelFitStatus={() => 'fits_now'}
            onSelect={handleSelect}
            onToggleFavorite={toggleFavorite}
            onAddModel={handleAddModel}
            hfSearchResults={hfResults}
            hfTrendingModels={hfTrending}
            hfIsSearching={hfSearching}
            onHfSearch={handleHfSearch}
            mlxOnly={mlxOnly}
            onToggleMlxOnly={handleToggleMlxOnly}
            mode="store-download"
            getBurstInfo={getBurstInfo}
            getHfBurstInfo={getHfBurstInfo}
            fleetMemoryBytes={fleet?.totalMemoryBytes}
            onHfLoadMore={hfHasMore && hfLimit < 200 ? handleHfLoadMore : undefined}
          />
        </ModalBody>
      </ModalContainer>
    </>
  );
}

const Backdrop = styled.div`
  position: fixed;
  inset: 0;
  z-index: 50;
  background: ${({ theme }) => theme.colors.overlay};
`;

const ModalContainer = styled.div`
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 51;
  display: flex;
  flex-direction: column;
  width: min(94vw, 900px);
  height: min(86vh, 760px);
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg};
  overflow: hidden;
  box-shadow: 0 0 43px ${({ theme }) => theme.colors.shadowStrong}, 0 0 88px ${({ theme }) => theme.colors.shadow};

  @media (max-width: 640px) {
    width: calc(100vw - 16px);
    height: calc(100vh - 16px);
  }
`;

const ModalHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
`;

const ModalTitle = styled.h2`
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.lg};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.gold};
  margin: 0;
`;

const CloseButton = styled.button`
  all: unset;
  cursor: pointer;
  color: ${({ theme }) => theme.colors.textMuted};
  transition: color 0.15s;
  display: flex;
  &:hover { color: ${({ theme }) => theme.colors.text}; }
`;

const ModalBody = styled.div`
  flex: 1;
  overflow: hidden;
`;
