import { useCallback, useEffect, useMemo, useState } from 'react';
import styled from 'styled-components';
import { ThemeProvider } from 'styled-components';
import { darkTheme, lightTheme, GlobalStyle } from './theme';
import { useClusterState } from './hooks/useClusterState';
import { HeaderNav, type NavRoute } from './components/layout/HeaderNav';
import { TopologyGraph } from './components/topology/TopologyGraph';
// ClusterWarnings replaced by inline header warning indicator
import { ConnectionBanner } from './components/status/ConnectionBanner';
import { ToastContainer } from './components/status/ToastContainer';
import { NetworkMesh } from './components/common/NetworkMesh';
import { SettingsPanel } from './components/layout/SettingsPanel';
import { ModelStorePage } from './components/pages/DownloadsPage';
import { ChatView } from './components/pages/ChatView';
import { InstancePanel, type InstanceCardData } from './components/layout/InstancePanel';
import { ConversationPanel } from './components/layout/ConversationPanel';
import { addToast } from './hooks/useToast';
import type { InstanceStatus } from './components/cluster/RunningInstanceCard';
import { useChatStore } from './stores/chatStore';
import { useUIStore } from './stores/uiStore';

const Shell = styled.div`
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  height: 100%;
`;

const ContentRow = styled.div`
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: row;
`;

const Main = styled.main`
  flex: 1;
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow-x: hidden;
  overflow-y: auto;
`;

interface StoreDownload {
  modelId: string;
  progress: number;
  status: string;
}

/* ── Runner status → InstanceStatus mapping ───────────── */

function deriveInstanceStatus(
  runnerIds: string[],
  runners: Record<string, Record<string, unknown>>,
): { status: InstanceStatus; message?: string; progress?: number } {
  if (runnerIds.length === 0) return { status: 'loading', message: 'Waiting for runners...' };

  const statuses = runnerIds.map((rid) => runners[rid]);

  // If any runner has failed, the instance is failed
  const failed = statuses.find((s) => s && 'RunnerFailed' in s);
  if (failed) {
    const inner = failed.RunnerFailed as Record<string, unknown> | undefined;
    return { status: 'failed', message: inner?.errorMessage as string | undefined };
  }

  // If any runner is shutting down
  if (statuses.some((s) => s && ('RunnerShuttingDown' in s || 'RunnerShutdown' in s))) {
    return { status: 'shutting_down' };
  }

  // If all runners are ready or running
  const allReady = statuses.every((s) => s && ('RunnerReady' in s || 'RunnerRunning' in s));
  if (allReady) {
    const anyRunning = statuses.some((s) => s && 'RunnerRunning' in s);
    return { status: anyRunning ? 'running' : 'ready' };
  }

  // If any runner is warming up
  if (statuses.some((s) => s && 'RunnerWarmingUp' in s)) {
    return { status: 'warming_up' };
  }

  // Loading — try to extract progress from RunnerLoading
  const loading = statuses.find((s) => s && 'RunnerLoading' in s);
  if (loading) {
    const inner = loading.RunnerLoading as Record<string, unknown> | undefined;
    const loaded = inner?.layersLoaded as number | undefined;
    const total = inner?.totalLayers as number | undefined;
    const progress = loaded != null && total != null && total > 0
      ? Math.round((loaded / total) * 100)
      : undefined;
    const message = loaded != null && total != null
      ? `Downloading layers ${loaded}/${total}...`
      : 'Loading model...';
    return { status: 'loading', message, progress };
  }

  // Connecting or idle
  return { status: 'loading', message: 'Connecting...' };
}

export function App() {
  const { topology, connected, downloads, nodeDisk, instances, runners } = useClusterState();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [storeDownloads, setStoreDownloads] = useState<StoreDownload[]>([]);
  const activeRoute = useUIStore((s) => s.activeRoute);
  const panelOpen = useUIStore((s) => s.panelOpen);
  const historyPanelOpen = useUIStore((s) => s.historyPanelOpen);
  const themeName = useUIStore((s) => s.theme);
  const activeTheme = themeName === 'light' ? lightTheme : darkTheme;

  // Reflect theme on the html root so non-styled-components surfaces (highlight.js,
  // scrollbars, etc.) can react via `html[data-theme='light'] …` selectors.
  useEffect(() => {
    if (typeof document === 'undefined') return;
    document.documentElement.setAttribute('data-theme', themeName);
  }, [themeName]);
  const setActiveRoute = useUIStore((s) => s.setActiveRoute);
  const togglePanel = useUIStore((s) => s.togglePanel);
  const toggleHistoryPanel = useUIStore((s) => s.toggleHistoryPanel);
  const conversationsMap = useChatStore((s) => s.conversations);
  const allConversations = useMemo(
    () => Object.values(conversationsMap).sort((a, b) => b.updatedAt - a.updatedAt),
    [conversationsMap],
  );
  const activeConversationId = useChatStore((s) => s.activeConversationId);
  const selectedModelId = useChatStore((s) => s.selectedModelId);
  const selectConversation = useChatStore((s) => s.selectConversation);
  const deleteConversation = useChatStore((s) => s.deleteConversation);
  const newConversation = useChatStore((s) => s.newConversation);

  // Compute cluster warnings from topology
  const clusterWarnings = useMemo(() => {
    const nodes = topology?.nodes;
    if (!nodes) return null;
    const items: { level: 'error' | 'warning'; message: string }[] = [];

    // Version mismatch (error)
    const entries = Object.values(nodes).filter(
      (n) => n.exo_commit && n.exo_commit !== 'Unknown' && n.exo_commit !== 'unknown',
    );
    if (entries.length >= 2) {
      const commits = new Set(entries.map((n) => n.exo_commit));
      if (commits.size > 1) {
        const details = entries.map((n) => {
          const ver = n.exo_version ?? '?';
          const commit = n.exo_commit?.slice(0, 7);
          return `${n.friendly_name ?? 'Unknown'} (${ver}${commit ? `-${commit}` : ''})`;
        }).join(', ');
        items.push({ level: 'error', message: `Version mismatch: ${details}. Update all nodes with git pull && uv sync.` });
      }
    }

    // RDMA phantom (warning)
    const hasRdmaPhantom = Object.values(nodes).some(
      (n) => n.rdma_enabled && n.rdma_interfaces_present === false,
    );
    if (hasRdmaPhantom) {
      items.push({ level: 'warning', message: 'RDMA reported as enabled but no RDMA interfaces exist. Thunderbolt 5 (M4 Pro/Max+) is required for tensor parallel.' });
    }

    if (items.length === 0) return null;
    const worstLevel = items.some((i) => i.level === 'error') ? 'error' : 'warning';
    return { level: worstLevel as 'error' | 'warning', items };
  }, [topology]);

  // Poll store downloads for the header progress indicator
  const pollStoreDownloads = useCallback(async () => {
    try {
      const res = await fetch('/store/downloads');
      if (res.ok) {
        const data = await res.json();
        setStoreDownloads(data.downloads ?? []);
      }
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    pollStoreDownloads();
    const id = setInterval(pollStoreDownloads, 3000);
    return () => clearInterval(id);
  }, [pollStoreDownloads]);

  const downloadProgress = useMemo(() => {
    if (storeDownloads.length === 0) return null;
    const totalPct = storeDownloads.reduce((sum, d) => sum + (d.progress * 100), 0);
    const avgPct = Math.round(totalPct / storeDownloads.length);
    return { count: storeDownloads.length, percentage: avgPct };
  }, [storeDownloads]);

  // Derive instance card data for the right panel
  const instanceCards = useMemo<InstanceCardData[]>(() => {
    const cards: InstanceCardData[] = [];
    for (const [iid, inst] of Object.entries(instances)) {
      const isRing = !!inst.MlxRingInstance;
      const inner = inst.MlxRingInstance ?? inst.MlxJacclInstance;
      if (!inner) continue;
      const sa = inner.shardAssignments;
      const modelId = sa?.modelId;
      if (!modelId) continue;

      const instanceId = inner.instanceId ?? iid;
      const nodeToRunner = sa?.nodeToRunner;
      const runnerIds = nodeToRunner ? Object.values(nodeToRunner) : [];
      const nodeIds = nodeToRunner ? Object.keys(nodeToRunner) : [];

      // Derive sharding and model type from shard metadata
      const runnerToShard = sa?.runnerToShard;
      let sharding: 'Pipeline' | 'Tensor' = 'Pipeline';
      let isEmbedding = false;
      if (runnerToShard) {
        const firstShard = Object.values(runnerToShard)[0];
        if (firstShard && 'TensorShardMetadata' in firstShard) {
          sharding = 'Tensor';
        }
        // Check model card tasks for embedding
        const shardInner = firstShard ? Object.values(firstShard)[0] as Record<string, unknown> | undefined : undefined;
        const mc = (shardInner?.modelCard ?? shardInner?.model_card) as Record<string, unknown> | undefined;
        const tasks = mc?.tasks as string[] | undefined;
        if (tasks?.includes('TextEmbedding')) isEmbedding = true;
      }

      // Derive status from runners
      const derived = deriveInstanceStatus(runnerIds, runners);

      // Get first node's friendly name
      const firstNodeId = nodeIds[0];
      const nodeName = firstNodeId && topology?.nodes[firstNodeId]?.friendly_name
        ? topology.nodes[firstNodeId].friendly_name
        : firstNodeId?.slice(0, 8) ?? 'unknown';

      cards.push({
        instanceId,
        modelId,
        sharding,
        instanceType: isRing ? 'MlxRing' : 'MlxJaccl',
        nodeName,
        status: derived.status,
        statusMessage: derived.message,
        loadProgress: derived.progress,
        isEmbedding,
      });
    }
    return cards;
  }, [instances, runners, topology]);

  const hasInstances = instanceCards.length > 0;

  const handleDeleteInstance = useCallback(async (instanceId: string) => {
    try {
      const res = await fetch(`/instance/${instanceId}`, { method: 'DELETE' });
      if (res.ok) {
        addToast({ type: 'success', message: 'Instance deleted' });
      } else {
        addToast({ type: 'error', message: 'Failed to delete instance' });
      }
    } catch {
      addToast({ type: 'error', message: 'Failed to delete instance' });
    }
  }, []);

  return (
    <ThemeProvider theme={activeTheme}>
      <GlobalStyle />
      <NetworkMesh radius={2.5} count={80} linkDistance={220} />
      <Shell>
        <ConnectionBanner connected={connected} />
        <HeaderNav
          showHome
          activeRoute={activeRoute}
          onNavigate={setActiveRoute}
          onOpenSettings={() => setSettingsOpen(true)}
          downloadProgress={downloadProgress}
          warnings={clusterWarnings}
          showSidebarToggle={activeRoute === 'chat' && allConversations.length > 0}
          sidebarVisible={historyPanelOpen}
          onToggleSidebar={toggleHistoryPanel}
          instanceCount={instanceCards.length}
          instancesHealthy={instanceCards.every((c) => c.status !== 'failed')}
          mobileRightOpen={panelOpen}
          onToggleMobileRight={togglePanel}
        />
        <ContentRow>
          {activeRoute === 'chat' && allConversations.length > 0 && historyPanelOpen && (
            <ConversationPanel
              conversations={allConversations}
              activeConversationId={activeConversationId}
              onSelect={selectConversation}
              onDelete={deleteConversation}
              onNewChat={() => { if (selectedModelId) newConversation(selectedModelId); }}
            />
          )}
          <Main>
            {activeRoute === 'model-store' ? (
              <ModelStorePage
                topology={topology}
                downloads={downloads}
                nodeDisk={nodeDisk}
                instances={instances}
                runners={runners}
                onChat={(modelId) => { useChatStore.getState().selectModel(modelId); setActiveRoute('chat'); }}
              />
            ) : activeRoute === 'chat' ? (
              <ChatView readyInstances={instanceCards} />
            ) : topology ? (
              <TopologyGraph data={topology} />
            ) : (
              <EmptyState>
                {connected ? 'Loading cluster state…' : 'Connecting to backend…'}
              </EmptyState>
            )}
          </Main>
          {hasInstances && panelOpen && (
            <InstancePanel
              instances={instanceCards}
              onDelete={handleDeleteInstance}
              onChat={(modelId) => { useChatStore.getState().selectModel(modelId); setActiveRoute('chat'); }}
            />
          )}
        </ContentRow>
        <ToastContainer />
        <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      </Shell>
    </ThemeProvider>
  );
}

const EmptyState = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 13px;
  color: ${({ theme }) => theme.colors.textMuted};
  text-transform: uppercase;
  letter-spacing: 2px;
`;
