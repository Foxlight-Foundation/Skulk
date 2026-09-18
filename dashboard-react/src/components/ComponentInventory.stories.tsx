import type { Meta, StoryObj } from '@storybook/react-vite';
import styled from 'styled-components';
import { useState } from 'react';
import { RightDrawer } from './common/RightDrawer';
import { Button } from './common/Button';
import { Spinner, CenteredSpinner } from './common/Spinner';
import { FamilyAvatar } from './models/FamilyAvatar';
import { HuggingFaceLink } from './models/HuggingFaceLink';
import { TraceWaterfall } from './observability/TraceWaterfall';
import { LiveTab } from './observability/LiveTab';
import { NodeTab } from './observability/NodeTab';
import { PerformanceTab } from './observability/PerformanceTab';
import { TracesTab } from './observability/TracesTab';
import { SettingsPanel } from './layout/SettingsPanel';
import { PairingSettings } from './layout/PairingSettings';
import { ConversationPanel } from './layout/ConversationPanel';
import { IntegrationsPage } from './pages/IntegrationsPage';
import { PluginsPage } from './pages/PluginsPage';
import { OperatorPage } from './pages/OperatorPage';
import { OperatorAccessPanel } from './pages/OperatorAccessPanel';
import { StewardChatView } from './pages/StewardChatView';
import { ChatView } from './pages/ChatView';
import { ModelStorePage } from './pages/DownloadsPage';
import { ManagedRuntimesPanel } from './pages/ManagedRuntimesPanel';
import { NodeCredentialsPanel } from './pages/NodeCredentialsPanel';
import { NodePreflightPanel } from './pages/NodePreflightPanel';
import { NodeProposalsPanel } from './pages/NodeProposalsPanel';
import { NodeSetupActionsPanel } from './pages/NodeSetupActionsPanel';
import { NodeSetupPanel } from './pages/NodeSetupPanel';
import { RuntimeSourceForm } from './pages/RuntimeSourceForm';
import { PluginConfigurationFields } from './pages/PluginConfigurationFields';
import { HardwareBadge } from './topology/HardwareBadge';

const Canvas = styled.div`
  min-height: 100vh; padding: 24px; display: flex; flex-direction: column; gap: 20px;
  max-width: 1440px; margin: auto;

`;
const meta = { title: 'Inventory/Connected components', parameters: { layout: 'fullscreen' }, decorators: [(Story) => <Canvas><Story /></Canvas>] } satisfies Meta;
export default meta;
type Story = StoryObj<typeof meta>;

/** Shared modal chrome exercises resizing, focus return and local dismissal. */
function DrawerDemo() {
  const [open, setOpen] = useState(true);
  const [width, setWidth] = useState(440);
  return <><Button onClick={() => setOpen(true)}>Open drawer</Button><RightDrawer open={open} onClose={() => setOpen(false)} width={width} minWidth={320} maxWidth={720} onWidthChange={setWidth} title="Drawer with a long translated title" ariaLabel="Example drawer" closeLabel="Close" resizeLabel="Resize"><Canvas style={{ minHeight: 0 }}><p>Keyboard focus stays within this surface.</p><Button>First action</Button><Button>Last action</Button></Canvas></RightDrawer></>;
}
export const Drawer: Story = { render: () => <DrawerDemo /> };
export const Settings: Story = { render: () => <SettingsPanel open onClose={() => {}} /> };
export const Pairing: Story = { render: () => <PairingSettings /> };
export const Integrations: Story = { render: () => <IntegrationsPage readyInstances={[]} /> };
export const Plugins: Story = { render: () => <PluginsPage /> };
export const ManagedRuntimes: Story = { render: () => <ManagedRuntimesPanel /> };
export const Operator: Story = { render: () => <OperatorPage /> };
export const OperatorAccess: Story = { render: () => <OperatorAccessPanel /> };
export const StewardDisabled: Story = { render: () => <StewardChatView readyInstances={[]} /> };
export const ChatEmpty: Story = { render: () => <ChatView readyInstances={[]} /> };
export const StoreEmpty: Story = { render: () => <ModelStorePage topology={null} downloads={{}} instances={{}} runners={{}} /> };
export const Live: Story = { render: () => <LiveTab /> };
export const Node: Story = { render: () => <NodeTab nodeId="example-node" /> };
export const Performance: Story = { render: () => <PerformanceTab /> };
export const Traces: Story = { render: () => <TracesTab /> };
export const WaterfallEmpty: Story = { render: () => <TraceWaterfall events={[]} /> };
export const PluginCredentialsUnavailable: Story = { render: () => <NodeCredentialsPanel pluginId="example-plugin" nodeId="example-node" /> };
export const PluginPreflightUnavailable: Story = { render: () => <NodePreflightPanel pluginId="example-plugin" nodeId="example-node" /> };
export const PluginProposalsUnavailable: Story = { render: () => <NodeProposalsPanel pluginId="example-plugin" nodeId="example-node" actionsAvailable={false} /> };
export const PluginSetupActionsUnavailable: Story = { render: () => <NodeSetupActionsPanel pluginId="example-plugin" nodeId="example-node" /> };
export const PluginSetupUnavailable: Story = { render: () => <NodeSetupPanel pluginId="example-plugin" nodeId="example-node" /> };
export const RuntimeSourceUnavailable: Story = { render: () => <RuntimeSourceForm pluginId="example-plugin" onSaved={() => {}} /> };
export const ConfigurationFields: Story = { render: () => <PluginConfigurationFields schema={{ type: 'object', properties: { label: { type: 'string', title: 'Display name' }, enabled: { type: 'boolean', title: 'Enabled' } } }} values={{ label: 'Example', enabled: true }} onChange={() => {}} disabled={false} /> };
export const HistoryEmpty: Story = { render: () => <ConversationPanel conversations={[]} activeConversationId={null} onSelect={() => {}} onDelete={() => {}} onNewChat={() => {}} /> };
export const SmallComponents: Story = { render: () => <><div style={{ display: 'flex', gap: 16 }}><FamilyAvatar name="Qwen" /><FamilyAvatar name="Example" /><HuggingFaceLink repoId="example/model" /><svg width="60" height="40"><HardwareBadge model="nvidia-gpu" /></svg><Spinner /></div><CenteredSpinner /></> };

export const IntegrationsReady: Story = { render: () => <IntegrationsPage readyInstances={[{ instanceId: 'fictional-instance', modelId: 'example/Chat-8B', status: 'ready', supportsTextChat: true, sharding: 'Pipeline', instanceType: 'MlxRing', engine: 'mlx', nodeStatuses: [] }]} /> };
