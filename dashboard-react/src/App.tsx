import styled from 'styled-components';
import { ThemeProvider } from 'styled-components';
import { theme, GlobalStyle } from './theme';
import { useClusterState } from './hooks/useClusterState';
import { HeaderNav } from './components/layout/HeaderNav';
import { TopologyGraph } from './components/topology/TopologyGraph';
import { ConnectionBanner } from './components/status/ConnectionBanner';
import { ToastContainer } from './components/status/ToastContainer';
import { NetworkMesh } from './components/common/NetworkMesh';

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

export function App() {
  const { topology, connected } = useClusterState();

  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      <NetworkMesh />
      <Shell>
        <ConnectionBanner connected={connected} />
        <HeaderNav
          showHome
          showSidebarToggle
          sidebarVisible={false}
        />
        <Main>
          {topology ? (
            <TopologyGraph data={topology} />
          ) : (
            <EmptyState>
              {connected ? 'Loading cluster state…' : 'Connecting to backend…'}
            </EmptyState>
          )}
        </Main>
        <ToastContainer />
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
