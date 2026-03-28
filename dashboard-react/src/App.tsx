import { useCallback, useEffect, useMemo, useState } from 'react';
import styled from 'styled-components';
import { ThemeProvider } from 'styled-components';
import { theme, GlobalStyle } from './theme';
import { useClusterState } from './hooks/useClusterState';
import { HeaderNav, type NavRoute } from './components/layout/HeaderNav';
import { TopologyGraph } from './components/topology/TopologyGraph';
import { ClusterWarnings } from './components/status/ClusterWarnings';
import { ConnectionBanner } from './components/status/ConnectionBanner';
import { ToastContainer } from './components/status/ToastContainer';
import { NetworkMesh } from './components/common/NetworkMesh';
import { SettingsPanel } from './components/layout/SettingsPanel';
import { ModelStorePage } from './components/pages/DownloadsPage';

const Shell = styled.div`
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  height: 100%;
`;

const Main = styled.main`
  flex: 1;
  min-height: 0;
`;

interface StoreDownload {
  modelId: string;
  progress: number;
  status: string;
}

export function App() {
  const { topology, connected, downloads, nodeDisk, instances, runners } = useClusterState();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [activeRoute, setActiveRoute] = useState<NavRoute>('cluster');
  const [storeDownloads, setStoreDownloads] = useState<StoreDownload[]>([]);

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

  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      <NetworkMesh radius={2.5} lineColor="rgba(255,215,0,0.35)" />
      <Shell>
        <ConnectionBanner connected={connected} />
        <HeaderNav
          showHome
          activeRoute={activeRoute}
          onNavigate={setActiveRoute}
          onOpenSettings={() => setSettingsOpen(true)}
          downloadProgress={downloadProgress}
        />
        <ClusterWarnings topology={topology} />
        <Main>
          {activeRoute === 'model-store' ? (
            <ModelStorePage
              topology={topology}
              downloads={downloads}
              nodeDisk={nodeDisk}
              instances={instances}
              runners={runners}
            />
          ) : topology ? (
            <TopologyGraph data={topology} />
          ) : (
            <EmptyState>
              {connected ? 'Loading cluster state…' : 'Connecting to backend…'}
            </EmptyState>
          )}
        </Main>
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
