import { Monogram } from '../common/Surfaces';
import { Button } from '../common/Button';
import { IntegrationSetupStep } from '../integrations/IntegrationSetupStep';
import { useEffect, useMemo, useState } from 'react';
import styled from 'styled-components';
import { FiCopy, FiCheck } from 'react-icons/fi';

import { SegmentedControl } from '../common/SegmentedControl';
import { RightDrawer } from '../common/RightDrawer';
import { IntegrationToolCard } from '../integrations/IntegrationToolCard';
import { Field } from '../common/Field';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { useRemoteAccess } from '../../hooks/useRemoteAccess';
import { copyToClipboard } from '../../utils/clipboard';
import { addToast } from '../../hooks/useToast';
import type { InstanceCardData } from '../layout/InstancePanel';
import type { ModelInfo } from '../../types/models';
import {
  INTEGRATION_TOOLS,
  PLACEHOLDER_MODEL_ID,
  buildIntegrationSnippets,
  deriveDefaultTiers,
  deriveIntegrationModels,
  partitionServingInstances,
  type IntegrationSnippet,
  type IntegrationToolId,
} from '../../utils/integrationConfigs';

/** Props for {@link IntegrationsPage}. */
export interface IntegrationsPageProps {
  /**
   * Instances that are ready or running.
   *
   * Passed down from the app shell so this page shares the single `/state`
   * poll rather than opening its own.
   */
  readyInstances: InstanceCardData[];
}

const Page = styled.div`
  padding: 32px;
  width: 100%;
  max-width: 1044px;
  @media (max-width: 600px) { padding: 16px; }
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 28px;
`;

const PageTitle = styled.h1`
  margin: 0;
  font-size: 28px;
  letter-spacing: -.02em;
  color: ${({ theme }) => theme.colors.text};
`;

const PageIntro = styled.p`
  margin: 6px 0 0;
  max-width: 560px;
  font-size: 14px;
  color: ${({ theme }) => theme.colors.textSecondary};
  line-height: 1.5;
`;

const SectionTitle = styled.h2`
  margin: 0;
  font-size: 10px;
  font-weight: 600;
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: ${({ theme }) => theme.colors.subtleText};
`;

const SurfaceRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
`;

const SurfaceChip = styled.div`
  flex: 1 1 200px;
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: 12px;
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
`;

const SurfaceLabel = styled.span`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-weight: 600;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: ${({ theme }) => theme.colors.subtleText};
`;

const SurfaceValue = styled.span`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 13px;
  color: ${({ theme }) => theme.colors.text};
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
`;

/** Copy an endpoint without treating the copy as connection evidence. */
function EndpointCard({ label, value }: { label: string; value: string }) {
  const { t } = useSkulkTranslation();
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try { await copyToClipboard(value); setCopied(true); }
    catch { addToast({ message: t('integrations.copyFailed', 'Could not copy to the clipboard'), type: 'error' }); }
  };
  return <SurfaceChip style={{ flexDirection: 'row', alignItems: 'center', gap: 12 }}><div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 5 }}><SurfaceLabel>{label}</SurfaceLabel><SurfaceValue title={value}>{value}</SurfaceValue></div><Button variant="outline" size="sm" icon aria-label={`${t('integrations.copy', 'Copy')} ${label}`} onClick={() => void copy()}>{copied ? <FiCheck size={13} /> : <FiCopy size={13} />}</Button></SurfaceChip>;
}

const ToolGrid = styled.div`
  display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px;
  @media (max-width: 900px) { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  @media (max-width: 600px) { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  @media (max-width: 360px) { grid-template-columns: minmax(0, 1fr); }
`;
const DrawerBody = styled.div`padding: 22px; overflow-y: auto; display: flex; flex-direction: column; gap: 22px; min-height: 0;`;

const ControlsRow = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px;
`;

/**
 * Standalone labelled control.
 *
 * The page is a column flex container, so this must not carry a flex basis
 * (which would be read as a height) and must not stretch its child to the full
 * page width.
 */
const StandaloneControl = styled.div`
  flex-shrink: 0;
  max-width: 100%;
  min-width: 0;
  flex-wrap: wrap;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
`;

const ControlBlock = styled.label`
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1 1 200px;
  min-width: 0;
`;

const ControlLabel = styled.span`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-weight: 600;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: ${({ theme }) => theme.colors.subtleText};
`;

const Select = styled.select`
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.borderControl};
  border-radius: 8px;
  color: ${({ theme }) => theme.colors.text};
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  padding: 9px 10px;
  cursor: pointer;
  min-width: 0;

  &:focus-visible {
    outline: none;
    border-color: ${({ theme }) => theme.colors.accentText};
  }
`;

const Card = styled.div`
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: 12px;
  overflow: hidden;
`;

const CardHeader = styled.div`
  padding: 12px 14px 10px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.borderLight};
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
`;

const CardHeading = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
`;

const CardSubtitle = styled.span`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 11px;
  color: ${({ theme }) => theme.colors.goldTextDim};
  overflow-wrap: anywhere;
`;

const CardDescription = styled.p`
  margin: 0;
  padding: 0;
  font-size: 13px;
  color: ${({ theme }) => theme.colors.textSecondary};
  line-height: 1.5;
`;

const CodeBlock = styled.pre`
  margin: 0;
  padding: 14px;
  background: ${({ theme }) => theme.colors.bg};
  border: 0;
  border-radius: 0;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  line-height: 1.7;
  color: ${({ theme }) => theme.colors.text};
  overflow-x: auto;
  white-space: pre;
`;

const CopyButton = styled.button`
  all: unset;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 0 10px;
  min-height: 30px;
  flex-shrink: 0;
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
  border-radius: ${({ theme }) => theme.radii.sm};
  color: ${({ theme }) => theme.colors.goldTextDim};
  font-size: 11px;
  white-space: nowrap;
  transition: all 0.15s;

  &:hover {
    color: ${({ theme }) => theme.colors.text};
    border-color: ${({ theme }) => theme.colors.accentText};
  }

  &:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px ${({ theme }) => theme.colors.gold};
  }
`;

const ModelChips = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
`;

const ModelChip = styled.span`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  color: ${({ theme }) => theme.colors.body};
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: 6px;
  padding: 3px 8px;
  overflow-wrap: anywhere;
`;

const EmptyNotice = styled.div`
  border: 1px dashed ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: 12px 14px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.subtleText};
  line-height: 1.5;
`;

/** Shortens a long model id for a dropdown option without losing the tail. */
function shortModelLabel(modelId: string): string {
  const tail = modelId.split('/').pop() ?? modelId;
  return tail.length > 46 ? `${tail.slice(0, 43)}...` : tail;
}

/** One snippet card with its own copy affordance. */
function SnippetCard({ snippet }: { snippet: IntegrationSnippet }) {
  const { t } = useSkulkTranslation();
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(timer);
  }, [copied]);

  const handleCopy = () => {
    void copyToClipboard(snippet.body)
      .then(() => setCopied(true))
      .catch(() => {
        addToast({
          type: 'error',
          message: t('integrations.copyFailed', 'Could not copy to the clipboard'),
        });
      });
  };

  return (
    <Card>
      <CardHeader>
        <CardHeading>
          <CardDescription>{snippet.description}</CardDescription>
          <CardSubtitle>{snippet.subtitle}</CardSubtitle>
        </CardHeading>
        <CopyButton onClick={handleCopy} aria-label={t('integrations.copy', 'Copy')}>
          {copied ? <FiCheck size={12} /> : <FiCopy size={12} />}
          {copied ? t('integrations.copied', 'Copied') : t('integrations.copy', 'Copy')}
        </CopyButton>
      </CardHeader>
      <CodeBlock>{snippet.body}</CodeBlock>
    </Card>
  );
}

/**
 * Shows copy-paste recipes for pointing external coding agents and apps at
 * this cluster.
 *
 * Every snippet is generated from live cluster truth: the models that
 * currently have a ready instance, their real context windows, and their
 * capability flags. The base URL comes from the node's remote-access info so
 * the snippet works from another machine rather than embedding `localhost`.
 */
export function IntegrationsPage({ readyInstances }: IntegrationsPageProps) {
  const { t } = useSkulkTranslation();
  const remoteAccess = useRemoteAccess();
  const [catalog, setCatalog] = useState<ModelInfo[]>([]);
  const [snippetId, setSnippetId] = useState<string | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [drawerWidth, setDrawerWidth] = useState(640);
  const [toolId, setToolId] = useState<IntegrationToolId>('claude-code');
  const [addressChoice, setAddressChoice] = useState<'local' | 'tailscale'>('local');
  const [codexFilesystemPath, setCodexFilesystemPath] = useState('/Users/username');
  const [opusOverride, setOpusOverride] = useState<string | null>(null);
  const [sonnetOverride, setSonnetOverride] = useState<string | null>(null);
  const [haikuOverride, setHaikuOverride] = useState<string | null>(null);
  const [selectedOverride, setSelectedOverride] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/models')
      .then(response => (response.ok ? (response.json() as Promise<{ data?: ModelInfo[] }>) : null))
      .then(payload => {
        if (!cancelled && payload?.data) setCatalog(payload.data);
      })
      .catch(() => {
        /* Capability metadata is an enhancement; snippets still render without it. */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Serving status alone is not enough: these recipes configure chat clients,
  // so a ready embedding or speech instance must not become a model choice.
  const { models, embeddingModels } = useMemo(() => {
    const partitioned = partitionServingInstances(readyInstances, catalog);
    return {
      models: deriveIntegrationModels(partitioned.chat, catalog),
      embeddingModels: deriveIntegrationModels(partitioned.embedding, catalog),
    };
  }, [readyInstances, catalog]);

  const localUrl =
    remoteAccess.status === 'ok' ? remoteAccess.data.local.url : null;
  const tailscaleUrl =
    remoteAccess.status === 'ok' ? remoteAccess.data.tailscale.url : null;

  // A snippet is pasted into a tool that may run on another machine, so an
  // origin of localhost would be wrong there. Prefer a routable address and
  // fall back to this page's origin only when the node reports none.
  const apiUrl = useMemo(() => {
    const chosen = addressChoice === 'tailscale' ? tailscaleUrl : localUrl;
    const resolved = chosen ?? localUrl ?? tailscaleUrl ?? window.location.origin;
    return resolved.replace(/\/+$/, '');
  }, [addressChoice, localUrl, tailscaleUrl]);

  const defaultTiers = useMemo(() => deriveDefaultTiers(models), [models]);
  const primaryModelId = models.length > 0 ? models[0].id : PLACEHOLDER_MODEL_ID;

  // Overrides are cleared implicitly: a stale id that is no longer ready falls
  // back to the derived default rather than pinning a model the cluster
  // stopped serving.
  const knownId = (candidate: string | null, fallback: string) =>
    candidate && models.some(model => model.id === candidate) ? candidate : fallback;

  const options = useMemo(
    () => ({
      apiUrl,
      models,
      embeddingModels,
      opusModelId: knownId(opusOverride, defaultTiers.opusModelId),
      sonnetModelId: knownId(sonnetOverride, defaultTiers.sonnetModelId),
      haikuModelId: knownId(haikuOverride, defaultTiers.haikuModelId),
      selectedModelId: knownId(selectedOverride, primaryModelId),
      codexFilesystemPath,
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      apiUrl,
      models,
      embeddingModels,
      opusOverride,
      sonnetOverride,
      haikuOverride,
      selectedOverride,
      defaultTiers,
      primaryModelId,
      codexFilesystemPath,
    ],
  );

  const tool = INTEGRATION_TOOLS.find(entry => entry.id === toolId) ?? INTEGRATION_TOOLS[0];
  const snippets = useMemo(
    () => buildIntegrationSnippets(tool.id, options, t),
    [tool.id, options, t],
  );

  const activeSnippet = snippets.find(snippet => snippet.id === snippetId) ?? snippets[0];
  const hasModels = models.length > 0;
  const showAddressChooser = Boolean(tailscaleUrl && localUrl && tailscaleUrl !== localUrl);

  const toolDescription = tool.usesTierChooser
    ? t('integrations.tierMapping', 'Anthropic-compatible · maps Opus / Sonnet / Haiku onto your models')
    : t('integrations.setupHint', 'Configure your tool using the endpoint and model choices below. Copying a recipe does not verify a connection.');

  return (
    <Page>
      <PageHeading><div>
        <PageTitle>{t('integrations.title', 'Integrations')}</PageTitle>
        <PageIntro>
          {t(
            'integrations.intro',
            'Point coding agents and apps at this cluster. Snippets are filled in with the models you have running right now.',
          )}
        </PageIntro>
      </div>
      {showAddressChooser && (
        <StandaloneControl style={{ flexDirection: 'row', alignItems: 'center' }}>
          <ControlLabel>{t('integrations.address', 'Address to use')}</ControlLabel>
          <SegmentedControl
            size="md"
            value={addressChoice}
            onChange={setAddressChoice}
            options={[
              { value: 'local' as const, label: t('integrations.addressLocal', 'Local network') },
              { value: 'tailscale' as const, label: t('integrations.addressTailscale', 'Tailscale') },
            ]}
          />
        </StandaloneControl>
      )}
      </PageHeading>

      <SurfaceRow>
        <EndpointCard label={t('integrations.surface.openai', 'OpenAI-compatible')} value={`${apiUrl}/v1`} />
        <EndpointCard label={t('integrations.surface.anthropic', 'Anthropic-compatible')} value={apiUrl} />
        <EndpointCard label={t('integrations.surface.ollama', 'Ollama-compatible')} value={`${apiUrl}/ollama`} />
      </SurfaceRow>



      <ReadyModels>
        <SectionTitle>{t('integrations.readyModels', 'Ready models')}</SectionTitle>
        {hasModels ? (
          <ModelChips>
            {models.map(model => (
              <ModelChip key={model.id}>{model.id}</ModelChip>
            ))}
          </ModelChips>
        ) : (
          <EmptyNotice style={{ marginTop: 8 }}>
            {t(
              'integrations.noReadyModels',
              'No models are running yet. The snippets below still show the right shape, with a placeholder where the model id goes. Mount a model and they will fill themselves in.',
            )}
          </EmptyNotice>
        )}
      </ReadyModels>

      <div>
        <ToolHeading><h2>{t('integrations.connectTool', 'Connect a tool')}</h2><span>{t('integrations.toolCount', '{count} tools · pick one to get its setup', { count: INTEGRATION_TOOLS.length })}</span></ToolHeading>
        <ToolGrid>
          {INTEGRATION_TOOLS.map(entry => <IntegrationToolCard key={entry.id} name={entry.label} monogram={TOOL_MONOGRAMS[entry.id]}
            description={t(`integrations.tools.${entry.id}.description`, {
              'claude-code': 'Anthropic-compatible coding agent in your terminal.', opencode: 'Open-source terminal agent with provider config.', codex: 'OpenAI Codex CLI pointed at a local model.',
              hermes: 'Agent harness with tool use and memory.', openclaw: 'Autonomous agent runtime.', pi: 'Minimalist assistant CLI.',
              anythingllm: 'Desktop RAG workspace: pairs a chat and an embedding model.', 'open-webui': 'Self-hosted chat UI for the whole cluster.', n8n: 'Workflow automation with LLM nodes.', firefox: 'Sidebar AI chat via about:config.',
            }[entry.id])}
            method={entry.surface === 'dashboard' ? 'Dashboard' : `${entry.surface} API`}
            onOpen={() => { setToolId(entry.id); setSnippetId(null); setDetailsOpen(true); }} />)}
        </ToolGrid>
      </div>
      <RightDrawer open={detailsOpen} onClose={() => setDetailsOpen(false)} title={<ToolIdentity><Monogram as="span" $size={40}>{TOOL_MONOGRAMS[tool.id]}</Monogram><span>{tool.label}<ToolSubtitle>{toolDescription}</ToolSubtitle></span></ToolIdentity>} ariaLabel={tool.label}
        width={drawerWidth} minWidth={360} maxWidth={800} onWidthChange={setDrawerWidth}
        closeLabel={t('common.close', 'Close')} resizeLabel={t('integrations.resize', 'Resize integration details')}>
      <DrawerBody>
      {(tool.usesTierChooser || tool.usesSingleModelChooser || tool.usesFilesystemPath) && (
        <IntegrationSetupStep number={1} title={tool.usesTierChooser ? t('integrations.chooseTiers', 'Choose which model answers each tier') : t('integrations.chooseOptions', 'Choose options')}>
          {!hasModels && (tool.usesTierChooser || tool.usesSingleModelChooser) && <EmptyNotice>{t('integrations.noReadyModels', 'No models are running yet. The snippets below still show the right shape, with a placeholder where the model id goes. Mount a model and they will fill themselves in.')}</EmptyNotice>}
          <ControlsRow>
          {tool.usesTierChooser && hasModels && (
            <>
              <ControlBlock>
                <ControlLabel>{t('integrations.tier.opus', 'Opus')}</ControlLabel>
                <Select
                  value={options.opusModelId}
                  onChange={event => setOpusOverride(event.target.value)}
                >
                  {models.map(model => (
                    <option key={model.id} value={model.id}>
                      {shortModelLabel(model.id)}
                    </option>
                  ))}
                </Select>
              </ControlBlock>
              <ControlBlock>
                <ControlLabel>{t('integrations.tier.sonnet', 'Sonnet')}</ControlLabel>
                <Select
                  value={options.sonnetModelId}
                  onChange={event => setSonnetOverride(event.target.value)}
                >
                  {models.map(model => (
                    <option key={model.id} value={model.id}>
                      {shortModelLabel(model.id)}
                    </option>
                  ))}
                </Select>
              </ControlBlock>
              <ControlBlock>
                <ControlLabel>{t('integrations.tier.haiku', 'Haiku')}</ControlLabel>
                <Select
                  value={options.haikuModelId}
                  onChange={event => setHaikuOverride(event.target.value)}
                >
                  {models.map(model => (
                    <option key={model.id} value={model.id}>
                      {shortModelLabel(model.id)}
                    </option>
                  ))}
                </Select>
              </ControlBlock>
            </>
          )}

          {tool.usesSingleModelChooser && hasModels && (
            <ControlBlock>
              <ControlLabel>{t('integrations.model', 'Model')}</ControlLabel>
              <Select
                value={options.selectedModelId}
                onChange={event => setSelectedOverride(event.target.value)}
              >
                {models.map(model => (
                  <option key={model.id} value={model.id}>
                    {shortModelLabel(model.id)}
                  </option>
                ))}
              </Select>
            </ControlBlock>
          )}

          {tool.usesFilesystemPath && (
            <ControlBlock>
              <ControlLabel>
                {t('integrations.filesystemPath', 'Filesystem path for MCP')}
              </ControlLabel>
              <Field
                value={codexFilesystemPath}
                onChange={event => setCodexFilesystemPath(event.target.value)}
                spellCheck={false}
              />
            </ControlBlock>
          )}
        </ControlsRow></IntegrationSetupStep>
      )}

      <IntegrationSetupStep number={tool.usesTierChooser || tool.usesSingleModelChooser || tool.usesFilesystemPath ? 2 : 1} title={t('integrations.apply', 'Apply it')} actions={snippets.length > 1 ? <SegmentedControl value={activeSnippet.id} onChange={setSnippetId} options={snippets.map(snippet => ({ value: snippet.id, label: snippet.title }))} /> : undefined}>
      {activeSnippet && <SnippetCard key={`${tool.id}-${activeSnippet.id}`} snippet={activeSnippet} />}
      </IntegrationSetupStep>
      <IntegrationSetupStep number={tool.usesTierChooser || tool.usesSingleModelChooser || tool.usesFilesystemPath ? 3 : 2} title={t('integrations.checkRequest', 'Check it reached the cluster')}>
        <EvidenceNotice>{t('integrations.evidenceUnavailable', 'Connection evidence is unavailable in this dashboard. Send a request from the configured tool and check its response. Copying these instructions does not establish a connection.')}</EvidenceNotice>
      </IntegrationSetupStep>
      </DrawerBody>
      </RightDrawer>
    </Page>
  );
}

export default IntegrationsPage;

const PageHeading = styled.div`
  display: flex; align-items: flex-end; justify-content: space-between; gap: 24px;
  @media (max-width: 1100px) { flex-wrap: wrap; gap: 12px; }
`;
const ReadyModels = styled.div`
  display: flex; flex-wrap: wrap; align-items: center; gap: 12px;
`;

const TOOL_MONOGRAMS: Record<IntegrationToolId, string> = { 'claude-code': 'CC', opencode: 'OC', codex: 'CX', hermes: 'HM', openclaw: 'OW', pi: 'PI', anythingllm: 'AL', 'open-webui': 'WU', n8n: 'N8', firefox: 'FF' };
const ToolHeading = styled.div`
  display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 12px;
  h2 { font-size: 16px; font-weight: 600; }
  > span { font-size: 12px; color: ${({ theme }) => theme.colors.subtleText}; }
`;
const ToolIdentity = styled.span`
  display: flex; align-items: center; gap: 12px; white-space: normal; min-width: 0;
  > span:last-child { min-width: 0; overflow-wrap: anywhere; }
`;
const ToolSubtitle = styled.span`
  display: block; margin-top: 2px; font-size: 12.5px; font-weight: 400; color: ${({ theme }) => theme.colors.textSecondary}; line-height: 1.5;
`;
const EvidenceNotice = styled.p`
  margin: 0; padding: 12px 14px; background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 12px;
  font-size: 13px; line-height: 1.5; color: ${({ theme }) => theme.colors.textSecondary};
`;
