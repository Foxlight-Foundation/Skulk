import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';
import type { TopologyData } from '../../types/topology';
import { detectDeviceModel } from '../../types/topology';
import type { RawDownloads, RawInstances, RawRunners } from '../../hooks/useClusterState';
import type { RawNodeResources } from '../../store/endpoints/cluster';
import type { FleetServingSummary } from '../models/burst';
import { StoreRegistryTable, type StoreRegistryEntry, type StoreDownloadProgress, type ModelCardInfo, type CompanionInfo } from '../layout/StoreRegistryTable';
import type { ClusterCardProps, ClusterCardNode } from '../cluster/ClusterCard';
import { ModelSearchModal } from './ModelSearchModal';
import { FiTrash2, FiSearch } from 'react-icons/fi';
import { Button } from '../common/Button';
import { addToast } from '../../hooks/useToast';
import { PlacementManager } from '../cluster/PlacementManager';
import { useSkulkTranslation } from '../../i18n/tolgee';

interface ModelStorePageProps {
  topology: TopologyData | null;
  nodeResources?: Record<string, RawNodeResources>;
  downloads: RawDownloads;
  instances: RawInstances;
  runners: RawRunners;
  onChat?: (modelId: string) => void;
}

/* ── Data extraction helpers ──────────────────────────── */

function getTag(entry: unknown): [string, Record<string, unknown>] | null {
  if (!entry || typeof entry !== 'object') return null;
  const obj = entry as Record<string, unknown>;
  for (const key of ['DownloadCompleted', 'DownloadOngoing', 'DownloadPending', 'DownloadFailed']) {
    if (key in obj && obj[key] && typeof obj[key] === 'object') {
      return [key, obj[key] as Record<string, unknown>];
    }
  }
  return null;
}

const TrashIcon = () => <FiTrash2 size={14} />;
const SearchIcon = () => <FiSearch size={14} />;
// formatBytes / formatSpeed deleted along with dead CellContent.

/* ── Component ────────────────────────────────────────── */

export function ModelStorePage({ topology, nodeResources = {}, downloads, instances, runners, onChat }: ModelStorePageProps) {
  const { t } = useSkulkTranslation();
  const [storeEntries, setStoreEntries] = useState<StoreRegistryEntry[]>([]);
  const [storeDownloads, setStoreDownloads] = useState<StoreDownloadProgress[]>([]);
  const [storeLoading, setStoreLoading] = useState(false);
  const [placementModelId, setPlacementModelId] = useState<string | null>(null);
  const [apiModelCards, setApiModelCards] = useState<Record<string, ModelCardInfo>>({});
  // Companion (drafter / MTP-head sidecar) repos, keyed by the companion's own
  // model_id. Derived from each parent card's runtime references so the store
  // can mark them as non-placeable instead of offering launch/placement/optiq.
  const [companionRoles, setCompanionRoles] = useState<Record<string, CompanionInfo>>({});
  const pollRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const storePollingDisposedRef = useRef(false);
  const optimizePollRef = useRef<ReturnType<typeof setInterval>>(undefined);

  // Fetch authoritative model card info from /models API
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch('/models');
        if (!res.ok) return;
        const data = await res.json();
        const cards: Record<string, ModelCardInfo> = {};
        // A model's runtime section names the companion repos it pulls in: an
        // MTP sidecar (prediction heads) and/or a separate draft model (MLX
        // assistant or served draft GGUF). Those repos land in the store but are
        // not independently placeable, so map them to a role for the store UI.
        // Skip self-references (same-repo co-fetch) so the parent isn't hidden.
        const companions: Record<string, CompanionInfo> = {};
        for (const m of data.data ?? []) {
          if (!m.id) continue;
          const rt = m.runtime ?? {};
          const sidecar = rt.mtp_sidecar_repo as string | undefined;
          const assistant = rt.assistant_model_repo as string | undefined;
          const draft = rt.served_spec_draft_repo as string | undefined;
          if (sidecar && sidecar !== m.id) {
            companions[sidecar] = { role: 'sidecar', parent: m.id };
          }
          for (const d of [assistant, draft]) {
            if (d && d !== m.id && !companions[d]) {
              companions[d] = { role: 'drafter', parent: m.id };
            }
          }
          cards[m.id] = {
            family: m.family ?? undefined,
            quantization: m.quantization ?? undefined,
            baseModel: m.base_model ?? undefined,
            supportsTensor: m.supports_tensor ?? undefined,
            capabilities: m.capabilities ?? undefined,
            tags: m.tags ?? [],
            tasks: m.tasks ?? [],
            resolvedCapabilities: m.resolved_capabilities ? {
              supportsThinking: m.resolved_capabilities.supports_thinking ?? false,
              supportsThinkingToggle: m.resolved_capabilities.supports_thinking_toggle ?? false,
              supportsThinkingBudget: m.resolved_capabilities.supports_thinking_budget ?? false,
              supportsImageInput: m.resolved_capabilities.supports_image_input ?? false,
              supportsAudioInput: m.resolved_capabilities.supports_audio_input ?? false,
              supportsSpeechSynthesis: m.resolved_capabilities.supports_speech_synthesis ?? false,
              supportsTranscription: m.resolved_capabilities.supports_transcription ?? false,
              supportsSpeechTranslation: m.resolved_capabilities.supports_speech_translation ?? false,
              supportsAudioOutput: m.resolved_capabilities.supports_audio_output ?? false,
              supportsRealtimeAudio: m.resolved_capabilities.supports_realtime_audio ?? false,
              defaultAudioResponseFormat: m.resolved_capabilities.default_audio_response_format ?? undefined,
              audioResponseFormats: m.resolved_capabilities.audio_response_formats ?? [],
              supportsToolCalling: m.resolved_capabilities.supports_tool_calling ?? false,
              builtinTools: m.resolved_capabilities.builtin_tools ?? [],
              thinkingFormat: m.resolved_capabilities.thinking_format ?? 'none',
              promptRenderer: m.resolved_capabilities.prompt_renderer ?? 'tokenizer',
              outputParser: m.resolved_capabilities.output_parser ?? 'generic',
              supportsNativeMultimodal: m.resolved_capabilities.supports_native_multimodal ?? false,
            } : undefined,
          };
        }
        setApiModelCards(cards);
        setCompanionRoles(companions);
      } catch { /* ignore */ }
    })();
  }, []);

  // Model card metadata: /models API is the source of truth (reflects
  // current TOML cards).  Download-state card data is only used as a
  // fallback for models not yet known to the API (e.g. mid-download
  // custom models).
  const modelCards = useMemo<Record<string, ModelCardInfo>>(() => {
    // Start with authoritative API data
    const cards: Record<string, ModelCardInfo> = { ...apiModelCards };

    // Fill in any models only visible in download state (custom/in-progress)
    for (const entries of Object.values(downloads)) {
      const list = Array.isArray(entries) ? entries : Object.values(entries);
      for (const entry of list) {
        const tagged = getTag(entry);
        if (!tagged) continue;
        const [, payload] = tagged;
        const shard = (payload.shardMetadata ?? payload.shard_metadata) as Record<string, unknown> | undefined;
        if (!shard) continue;
        for (const val of Object.values(shard)) {
          if (!val || typeof val !== 'object') continue;
          const inner = val as Record<string, unknown>;
          const raw = (inner.modelCard ?? inner.model_card) as Record<string, unknown> | undefined;
          if (!raw) continue;
          const mid = (raw.modelId ?? raw.model_id) as string | undefined;
          if (!mid || cards[mid]) continue; // API data takes precedence
          // Derive tags from download metadata (mirrors server-side _model_tags logic)
          const fallbackTags: string[] = [];
          if (mid.toLowerCase().includes('optiq') || (raw.quantization as string ?? '').toLowerCase().includes('optiq')) fallbackTags.push('optiq');
          if ((raw.capabilities as string[] ?? []).includes('thinking')) fallbackTags.push('thinking');
          if ((raw.capabilities as string[] ?? []).includes('vision')) fallbackTags.push('vision');
          if ((raw.supportsTensor ?? raw.supports_tensor) as boolean) fallbackTags.push('tensor');
          if ((raw.capabilities as string[] ?? []).includes('embedding')) fallbackTags.push('embedding');
          if ((raw.capabilities as string[] ?? []).includes('tts')) fallbackTags.push('tts');
          if ((raw.capabilities as string[] ?? []).includes('stt')) fallbackTags.push('stt');
          cards[mid] = {
            family: raw.family as string | undefined,
            quantization: raw.quantization as string | undefined,
            baseModel: (raw.baseModel ?? raw.base_model) as string | undefined,
            supportsTensor: (raw.supportsTensor ?? raw.supports_tensor) as boolean | undefined,
            capabilities: raw.capabilities as string[] | undefined,
            tags: fallbackTags,
            tasks: raw.tasks as string[] | undefined,
          };
        }
      }
    }
    return cards;
  }, [downloads, apiModelCards]);

  const fetchDownloads = useCallback(async (): Promise<StoreDownloadProgress[] | null> => {
    try {
      const res = await fetch('/store/downloads');
      if (!res.ok) return null;
      const data = await res.json();
      return (data.downloads ?? []) as StoreDownloadProgress[];
    } catch { return null; }
  }, []);

  const fetchRegistry = useCallback(async (): Promise<StoreRegistryEntry[] | null> => {
    try {
      const response = await fetch('/store/registry');
      if (!response.ok) return null;
      const data = await response.json();
      return (data.entries ?? []) as StoreRegistryEntry[];
    } catch { return null; }
  }, []);

  const refreshStore = useCallback(async (): Promise<boolean> => {
    const [registryEntries, downloadEntries] = await Promise.all([
      fetchRegistry(),
      fetchDownloads(),
    ]);
    if (storePollingDisposedRef.current) return true;
    if (registryEntries !== null) setStoreEntries(registryEntries);
    if (downloadEntries !== null) setStoreDownloads(downloadEntries);

    // A transient failure must keep the convergence loop alive. In particular,
    // the store can finish a download just as the dashboard reloads; stopping on
    // an empty downloads response while the registry request failed leaves a new
    // user permanently looking at "0 models in store" until a manual refresh.
    return registryEntries !== null
      && downloadEntries !== null
      && downloadEntries.length === 0;
  }, [fetchDownloads, fetchRegistry]);

  const scheduleStoreRefresh = useCallback(() => {
    if (storePollingDisposedRef.current || pollRef.current) return;

    const poll = async () => {
      const converged = await refreshStore();
      if (storePollingDisposedRef.current || converged) {
        pollRef.current = undefined;
        return;
      }
      pollRef.current = setTimeout(poll, 2000);
    };

    pollRef.current = setTimeout(poll, 2000);
  }, [refreshStore]);

  const loadRegistry = useCallback(async () => {
    setStoreLoading(true);
    try {
      if (!(await refreshStore())) scheduleStoreRefresh();
    } finally {
      if (!storePollingDisposedRef.current) setStoreLoading(false);
    }
  }, [refreshStore, scheduleStoreRefresh]);

  // Load store registry on mount
  useEffect(() => {
    loadRegistry();
  }, [loadRegistry]);

  // Cleanup polling on unmount
  useEffect(() => {
    storePollingDisposedRef.current = false;
    return () => {
      storePollingDisposedRef.current = true;
      if (pollRef.current) clearTimeout(pollRef.current);
      if (optimizePollRef.current) clearInterval(optimizePollRef.current);
    };
  }, []);

  const [purgeConfirm, setPurgeConfirm] = useState(false);
  const [purging, setPurging] = useState(false);

  const handlePurge = useCallback(async () => {
    setPurging(true);
    try {
      const res = await fetch('/store/purge-staging', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modelId: null }),
      });
      if (res.ok) {
        addToast({ type: 'success', message: t('downloads.toasts.purgeSent', 'Purge command sent to all nodes') });
      } else {
        addToast({ type: 'error', message: t('downloads.toasts.purgeFailed', 'Failed to send purge command') });
      }
    } catch {
      addToast({ type: 'error', message: t('downloads.toasts.purgeFailed', 'Failed to send purge command') });
    } finally {
      setPurging(false);
      setPurgeConfirm(false);
    }
  }, [t]);

  const [searchOpen, setSearchOpen] = useState(false);

  const storeModelIds = useMemo(
    () => new Set(storeEntries.map((e) => e.model_id)),
    [storeEntries],
  );

  // What the local fleet can serve: total memory (browse-time acquisition
  // decisions judge against capacity, not currently-free RAM) plus which
  // artifact formats some node's advertised engines can run. Drives the
  // Find Models burst partition.
  const fleet = useMemo<FleetServingSummary | null>(() => {
    if (!topology?.nodes || Object.keys(topology.nodes).length === 0) return null;
    const totalMemoryBytes = Object.values(topology.nodes).reduce(
      (sum, node) => sum + (node.mactop_info?.memory?.ram_total ?? 0),
      0,
    );
    if (totalMemoryBytes <= 0) return null;
    const backends = Object.values(nodeResources).flatMap((r) => r.backends ?? []);
    return {
      totalMemoryBytes,
      servesMlx: backends.includes('mlx'),
      servesGguf: backends.some(
        (b) => b.startsWith('llama_cpp') || b.startsWith('llama_server') || b.startsWith('vllm'),
      ),
    };
  }, [topology, nodeResources]);

  // Total cluster available (free) RAM
  const totalClusterMemoryBytes = useMemo(() => {
    if (!topology?.nodes) return 0;
    return Object.values(topology.nodes).reduce((sum, node) => {
      const mem = node.mactop_info?.memory;
      if (!mem) return sum;
      const available = mem.ram_total - mem.ram_usage;
      return sum + Math.max(available, 0);
    }, 0);
  }, [topology]);

  // Derive active model IDs and model→instanceId mapping from running instances
  const { activeModelIds, modelToInstanceId } = useMemo(() => {
    const ids: string[] = [];
    const mapping: Record<string, string> = {};
    for (const [iid, inst] of Object.entries(instances)) {
      const inner =
        inst.MlxRingInstance ?? inst.MlxJacclInstance ?? inst.LlamaRpcInstance;
      const modelId = inner?.shardAssignments?.modelId;
      if (modelId) {
        ids.push(modelId);
        mapping[modelId] = (inner as Record<string, unknown>).instanceId as string ?? iid;
      }
    }
    return { activeModelIds: ids, modelToInstanceId: mapping };
  }, [instances]);

  // Build ClusterCard data for active models
  const clusterCards = useMemo<Record<string, Omit<ClusterCardProps, 'onLaunch'>>>(() => {
    const cards: Record<string, Omit<ClusterCardProps, 'onLaunch'>> = {};
    for (const [, inst] of Object.entries(instances)) {
      const isRing = !!inst.MlxRingInstance;
      const inner =
        inst.MlxRingInstance ?? inst.MlxJacclInstance ?? inst.LlamaRpcInstance;
      if (!inner) continue;
      const sa = inner.shardAssignments;
      const modelId = sa?.modelId;
      if (!modelId) continue;

      const nodeToRunner = sa?.nodeToRunner;
      const nodeIds = nodeToRunner ? Object.keys(nodeToRunner) : [];
      const runnerIds = nodeToRunner ? Object.values(nodeToRunner) : [];

      // Derive sharding from shard metadata (tagged union: { "PipelineShardMetadata": {...} } etc.)
      const runnerToShard = sa?.runnerToShard;
      let sharding: 'Pipeline' | 'Tensor' = 'Pipeline';
      if (runnerToShard) {
        const firstShard = Object.values(runnerToShard)[0];
        if (firstShard && 'TensorShardMetadata' in firstShard) {
          sharding = 'Tensor';
        }
      }

      // Gate isReady on all runners being in RunnerReady or RunnerRunning state
      const isReady = runnerIds.length > 0 && runnerIds.every((rid) => {
        const status = runners[rid];
        if (!status) return false;
        return 'RunnerReady' in status || 'RunnerRunning' in status;
      });

      const cardNodes: ClusterCardNode[] = nodeIds.map((nid) => {
        const nodeInfo = topology?.nodes[nid];
        const mem = nodeInfo?.mactop_info?.memory;
        const pct = mem && mem.ram_total > 0
          ? Math.round((mem.ram_usage / mem.ram_total) * 100)
          : 0;
        return {
          nodeId: nid,
          name: nodeInfo?.friendly_name ?? nid.slice(0, 8),
          memoryUsedPercent: pct,
          deviceModel: detectDeviceModel(nodeInfo?.system_info?.model_id, nodeInfo?.system_info?.chip, nodeInfo?.system_info?.accelerator_vendor),
        };
      });

      const storeEntry = storeEntries.find((e) => e.model_id === modelId);

      cards[modelId] = {
        modelId,
        sizeBytes: storeEntry?.total_bytes,
        sharding,
        instanceType: isRing
          ? 'MlxRing'
          : inst.MlxJacclInstance
            ? 'MlxJaccl'
            : 'LlamaRpc',
        nodes: cardNodes,
        isReady,
      };
    }
    return cards;
  }, [instances, runners, topology, storeEntries]);

  const handleLaunchWithParams = useCallback(async (params: { modelId: string; sharding: string; instanceMeta: string; minNodes: number; excludedNodes?: string[] }) => {
    try {
      const res = await fetch('/place_instance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: params.modelId,
          sharding: params.sharding,
          instance_meta: params.instanceMeta,
          min_nodes: params.minNodes,
          excluded_nodes: params.excludedNodes ?? [],
        }),
      });
      if (res.ok) {
        addToast({
          type: 'success',
          message: t('downloads.toasts.launchingModel', 'Launching {modelId}', { modelId: params.modelId }),
        });
      } else {
        const err = await res.json().catch(() => ({}));
        addToast({
          type: 'error',
          message: (err as Record<string, string>).detail
            ?? t('downloads.toasts.launchFailedForModel', 'Failed to launch {modelId}', { modelId: params.modelId }),
        });
      }
    } catch {
      addToast({ type: 'error', message: t('downloads.toasts.launchFailed', 'Failed to launch model') });
    }
  }, [t]);

  const handleLaunch = useCallback((modelId: string) => {
    handleLaunchWithParams({ modelId, sharding: 'Pipeline', instanceMeta: 'MlxRing', minNodes: 1, excludedNodes: [] });
  }, [handleLaunchWithParams]);

  const handleStop = useCallback(async (modelId: string) => {
    const instanceId = modelToInstanceId[modelId];
    if (!instanceId) {
      addToast({
        type: 'error',
        message: t('downloads.toasts.noActiveInstance', 'No active instance found for {modelId}', { modelId }),
      });
      return;
    }
    try {
      const res = await fetch(`/instance/${encodeURIComponent(instanceId)}`, { method: 'DELETE' });
      if (res.ok) {
        addToast({ type: 'success', message: t('downloads.toasts.stoppingModel', 'Stopping {modelId}', { modelId }) });
      } else {
        addToast({ type: 'error', message: t('downloads.toasts.stopFailedForModel', 'Failed to stop {modelId}', { modelId }) });
      }
    } catch {
      addToast({ type: 'error', message: t('downloads.toasts.stopFailed', 'Failed to stop model') });
    }
  }, [modelToInstanceId, t]);

  const handleDelete = useCallback(async (entry: { model_id: string }, isActive: boolean) => {
    if (isActive) {
      addToast({ type: 'error', message: t('downloads.toasts.stopBeforeDelete', 'Stop the model before deleting') });
      return;
    }
    try {
      const res = await fetch(`/store/models/${encodeURIComponent(entry.model_id)}`, { method: 'DELETE' });
      if (res.ok) {
        addToast({
          type: 'success',
          message: t('downloads.toasts.deletedModel', 'Deleted {modelId}', { modelId: entry.model_id }),
        });
        loadRegistry();
      } else {
        addToast({
          type: 'error',
          message: t('downloads.toasts.deleteFailedForModel', 'Failed to delete {modelId}', { modelId: entry.model_id }),
        });
      }
    } catch {
      addToast({ type: 'error', message: t('downloads.toasts.deleteFailed', 'Failed to delete model') });
    }
  }, [loadRegistry, t]);

  const handleOptimize = useCallback(async (modelId: string) => {
    try {
      const res = await fetch(`/store/models/${encodeURIComponent(modelId)}/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_bpw: 4.5, candidate_bits: [4, 8] }),
      });
      if (res.ok) {
        addToast({
          type: 'success',
          message: t('downloads.toasts.optiqStarted', 'OptiQ optimization started for {modelId}', { modelId }),
        });
        // Poll for completion/failure (tracked for cleanup on unmount)
        if (optimizePollRef.current) clearInterval(optimizePollRef.current);
        const pollInterval = setInterval(async () => {
          try {
            const statusRes = await fetch(`/store/models/${encodeURIComponent(modelId)}/optimize/status`);
            if (!statusRes.ok) { clearInterval(pollInterval); return; }
            const status = await statusRes.json();
            if (status.status === 'complete') {
              clearInterval(pollInterval);
              addToast({
                type: 'success',
                message: t(
                  'downloads.toasts.optiqComplete',
                  'OptiQ optimization complete: {achievedBpw} BPW',
                  { achievedBpw: status.achievedBpw?.toFixed(2) ?? '?' },
                ),
              });
              loadRegistry();
            } else if (status.status === 'failed') {
              clearInterval(pollInterval);
              addToast({
                type: 'error',
                message: status.error ?? t('downloads.toasts.optimizationFailed', 'Optimization failed'),
              });
            }
          } catch { clearInterval(pollInterval); optimizePollRef.current = undefined; }
        }, 5000);
        optimizePollRef.current = pollInterval;
      } else {
        const err = await res.json().catch(() => ({}));
        addToast({
          type: 'error',
          message: (err as Record<string, string>).detail
            ?? t('downloads.toasts.optimizationStartFailed', 'Failed to start optimization'),
        });
      }
    } catch {
      addToast({ type: 'error', message: t('downloads.toasts.optimizationStartFailed', 'Failed to start optimization') });
    }
  }, [loadRegistry, t]);

  return (
    <Container>
      {purgeConfirm && (
        <PurgeModal>
          <ModalBackdrop onClick={() => setPurgeConfirm(false)} />
          <ModalBox>
            <ModalTitle>{t('downloads.purge.title', 'Purge all node caches')}</ModalTitle>
            <ModalText>
              {t(
                'downloads.purge.description',
                'This will remove all staged model files and partial downloads from every node in the cluster. Models will need to be re-downloaded before they can run again.',
              )}
            </ModalText>
            <ModalNote>
              {t('downloads.purge.offlineNote', 'Nodes that are currently offline will not receive this command.')}
            </ModalNote>
            <ModalActions>
              <Button variant="outline" size="md" onClick={() => setPurgeConfirm(false)}>
                {t('common.cancel', 'Cancel')}
              </Button>
              <Button variant="danger" size="md" loading={purging} onClick={handlePurge}>
                {purging ? t('downloads.purge.purging', 'Purging...') : t('downloads.purge.confirm', 'Purge All')}
              </Button>
            </ModalActions>
          </ModalBox>
        </PurgeModal>
      )}

      <StoreRegistryTable
          entries={storeEntries}
          activeDownloads={storeDownloads}
          loading={storeLoading}
          activeModelIds={activeModelIds}
          modelCards={modelCards}
          companions={companionRoles}
          actions={
            <>
              <Button variant="danger" size="sm" onClick={() => setPurgeConfirm(true)}>
                <TrashIcon /> {t('downloads.actions.purgeNodeCaches', 'Purge Node Caches')}
              </Button>
              <Button
                variant="primary"
                size="sm"
                onClick={() => setSearchOpen(true)}
                aria-label={t('downloads.actions.findModels', 'Find Models')}
              >
                <SearchIcon /> {t('downloads.actions.findModels', 'Find Models')}
              </Button>
            </>
          }
          onRefresh={loadRegistry}
          onDelete={handleDelete}
          onLaunch={handleLaunch}
          onStop={handleStop}
          onChat={onChat}
          onPlacement={setPlacementModelId}
          clusterCards={clusterCards}
          totalClusterMemoryBytes={totalClusterMemoryBytes}
          onOptimize={handleOptimize}
        />
      <ModelSearchModal
        open={searchOpen}
        onClose={() => setSearchOpen(false)}
        existingModelIds={storeModelIds}
        onDownloadStarted={loadRegistry}
        fleet={fleet}
      />
      {placementModelId && topology && (
        <PlacementManager
          modelId={placementModelId}
          modelSizeMb={storeEntries.find((e) => e.model_id === placementModelId)?.total_bytes
            ? Math.round(storeEntries.find((e) => e.model_id === placementModelId)!.total_bytes / 1e6)
            : undefined}
          topology={topology}
          open={!!placementModelId}
          onClose={() => setPlacementModelId(null)}
          onLaunch={handleLaunchWithParams}
          isEmbedding={modelCards[placementModelId]?.tags?.includes('embedding')}
        />
      )}
    </Container>
  );
}

// CellContent / CheckIcon were unused dead code — removed during light-theme migration.

/* ── Styles ───────────────────────────────────────────── */

const Container = styled.div`
  padding: 32px;
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;


const PurgeModal = styled.div`
  position: fixed;
  inset: 0;
  z-index: 60;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const ModalBackdrop = styled.div`
  position: absolute;
  inset: 0;
  background: ${({ theme }) => theme.colors.overlay};
`;

const ModalBox = styled.div`
  position: relative;
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: 24px;
  width: 420px;
  max-width: 90vw;
`;

const ModalTitle = styled.h3`
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.error};
  margin: 0 0 12px;
`;

const ModalText = styled.p`
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.textSecondary};
  line-height: 1.5;
  margin: 0 0 8px;
`;

const ModalNote = styled.p`
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  margin: 0 0 16px;
`;

const ModalActions = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: 12px;
`;












// Cell-status display styled-components removed alongside dead CellContent.
