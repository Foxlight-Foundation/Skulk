import { useEffect, useMemo, useState } from 'react';
import styled from 'styled-components';
import { FiCopy, FiCheck } from 'react-icons/fi';

import { SegmentedControl } from '../common/SegmentedControl';
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
  padding: 16px;
  max-width: 760px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
`;

const PageTitle = styled.h1`
  margin: 0;
  font-size: ${({ theme }) => theme.fontSizes.xl};
  color: ${({ theme }) => theme.colors.text};
`;

const PageIntro = styled.p`
  margin: 0;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  line-height: 1.5;
`;

const SectionTitle = styled.h2`
  margin: 0;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: ${({ theme }) => theme.colors.textMuted};
`;

const SurfaceRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
`;

const SurfaceChip = styled.div`
  flex: 1 1 200px;
  background: ${({ theme }) => theme.colors.surfaceSunken};
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: 8px 10px;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
`;

const SurfaceLabel = styled.span`
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: ${({ theme }) => theme.colors.textMuted};
`;

const SurfaceValue = styled.span`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.text};
  overflow-wrap: anywhere;
`;

const ChooserScroll = styled.div`
  overflow-x: auto;
  max-width: 100%;
  padding-bottom: 4px;
`;

const ControlsRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
`;

/**
 * Standalone labelled control.
 *
 * The page is a column flex container, so this must not carry a flex basis
 * (which would be read as a height) and must not stretch its child to the full
 * page width.
 */
const StandaloneControl = styled.div`
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
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: ${({ theme }) => theme.colors.textMuted};
`;

const Select = styled.select`
  background: ${({ theme }) => theme.colors.surfaceSunken};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.sm};
  color: ${({ theme }) => theme.colors.text};
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  padding: 6px 8px;
  cursor: pointer;
  min-width: 0;

  &:focus-visible {
    outline: none;
    border-color: ${({ theme }) => theme.colors.gold};
  }
`;

const Card = styled.div`
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg};
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

const CardTitle = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text};
`;

const CardSubtitle = styled.span`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 11px;
  color: ${({ theme }) => theme.colors.goldTextDim};
  overflow-wrap: anywhere;
`;

const CardDescription = styled.p`
  margin: 0;
  padding: 10px 14px 0;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
  line-height: 1.5;
`;

const CodeBlock = styled.pre`
  margin: 10px 14px 14px;
  padding: 10px 12px;
  background: ${({ theme }) => theme.colors.chatCodeBg};
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.md};
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 11.5px;
  line-height: 1.55;
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
  padding: 4px 9px;
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
  border-radius: ${({ theme }) => theme.radii.sm};
  color: ${({ theme }) => theme.colors.goldTextDim};
  font-size: 11px;
  white-space: nowrap;
  transition: all 0.15s;

  &:hover {
    color: ${({ theme }) => theme.colors.text};
    border-color: ${({ theme }) => theme.colors.gold};
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
  font-size: 11px;
  color: ${({ theme }) => theme.colors.goldTextDim};
  background: ${({ theme }) => theme.colors.goldBg};
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 2px 7px;
  overflow-wrap: anywhere;
`;

const EmptyNotice = styled.div`
  border: 1px dashed ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: 12px 14px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
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
          <CardTitle>{snippet.title}</CardTitle>
          <CardSubtitle>{snippet.subtitle}</CardSubtitle>
        </CardHeading>
        <CopyButton onClick={handleCopy} aria-label={t('integrations.copy', 'Copy')}>
          {copied ? <FiCheck size={12} /> : <FiCopy size={12} />}
          {copied ? t('integrations.copied', 'Copied') : t('integrations.copy', 'Copy')}
        </CopyButton>
      </CardHeader>
      <CardDescription>{snippet.description}</CardDescription>
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

  const toolOptions = INTEGRATION_TOOLS.map(entry => ({ value: entry.id, label: entry.label }));
  const hasModels = models.length > 0;
  const showAddressChooser = Boolean(tailscaleUrl && localUrl && tailscaleUrl !== localUrl);

  return (
    <Page>
      <div>
        <PageTitle>{t('integrations.title', 'Integrations')}</PageTitle>
        <PageIntro>
          {t(
            'integrations.intro',
            'Point external coding agents and apps at this cluster. Every snippet below is filled in with the models you actually have running.',
          )}
        </PageIntro>
      </div>

      <SurfaceRow>
        <SurfaceChip>
          <SurfaceLabel>{t('integrations.surface.openai', 'OpenAI-compatible')}</SurfaceLabel>
          <SurfaceValue>{`${apiUrl}/v1`}</SurfaceValue>
        </SurfaceChip>
        <SurfaceChip>
          <SurfaceLabel>{t('integrations.surface.anthropic', 'Anthropic-compatible')}</SurfaceLabel>
          <SurfaceValue>{apiUrl}</SurfaceValue>
        </SurfaceChip>
        <SurfaceChip>
          <SurfaceLabel>{t('integrations.surface.ollama', 'Ollama-compatible')}</SurfaceLabel>
          <SurfaceValue>{`${apiUrl}/ollama`}</SurfaceValue>
        </SurfaceChip>
      </SurfaceRow>

      {showAddressChooser && (
        <StandaloneControl>
          <ControlLabel>{t('integrations.address', 'Address to use')}</ControlLabel>
          <SegmentedControl
            size="sm"
            value={addressChoice}
            onChange={setAddressChoice}
            options={[
              { value: 'local' as const, label: t('integrations.addressLocal', 'Local network') },
              { value: 'tailscale' as const, label: t('integrations.addressTailscale', 'Tailscale') },
            ]}
          />
        </StandaloneControl>
      )}

      <div>
        <SectionTitle>{t('integrations.readyModels', 'Ready models')}</SectionTitle>
        {hasModels ? (
          <ModelChips style={{ marginTop: 8 }}>
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
      </div>

      <div>
        <SectionTitle>{t('integrations.tool', 'Tool')}</SectionTitle>
        <ChooserScroll style={{ marginTop: 8 }}>
          <SegmentedControl
            size="md"
            value={tool.id}
            onChange={setToolId}
            options={toolOptions}
          />
        </ChooserScroll>
      </div>

      {(tool.usesTierChooser || tool.usesSingleModelChooser || tool.usesFilesystemPath) && (
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
        </ControlsRow>
      )}

      {snippets.map(snippet => (
        <SnippetCard key={`${tool.id}-${snippet.id}`} snippet={snippet} />
      ))}
    </Page>
  );
}

export default IntegrationsPage;
