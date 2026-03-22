/**
 * SettingsPage  —  "/settings"
 *
 * Model store configuration form.
 * Mirrors the Svelte settings/+page.svelte implementation.
 */
import React, { useEffect, useState, useCallback } from 'react';
import styled from 'styled-components';
import { fetchConfig, updateConfig, fetchStoreHealth, fetchNodeIdentity } from '../api/client';
import type { StoreHealthResponse, NodeIdentityResponse } from '../api/types';
import DirectoryBrowser from '../components/ui/DirectoryBrowser';

// ─── Styled components ────────────────────────────────────────────────────────

const Page = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 24px;
  max-width: 720px;
  overflow-y: auto;
`;

const PageTitle = styled.h1`
  margin: 0 0 4px;
  font-size: 13px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.yellow};
`;

const PageSubtitle = styled.p`
  margin: 0 0 24px;
  font-size: 12px;
  color: ${({ theme }) => theme.colors.lightGray};
  font-family: ${({ theme }) => theme.fonts.mono};
`;

const Section = styled.fieldset`
  border: 1px solid ${({ theme }) => `${theme.colors.mediumGray}80`};
  border-radius: ${({ theme }) => theme.radius.sm};
  background: ${({ theme }) => `${theme.colors.black}80`};
  padding: 16px;
  margin-bottom: 16px;
`;

const SectionLegend = styled.legend`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.lightGray};
  padding: 0 8px;
`;

const FieldRow = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 12px;

  &:last-child { margin-bottom: 0; }
`;

const FieldGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 12px;

  @media (max-width: 600px) { grid-template-columns: 1fr; }
`;

const FieldLabel = styled.label`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.lightGray};
`;

const TextInput = styled.input`
  background: ${({ theme }) => theme.colors.darkGray};
  border: 1px solid ${({ theme }) => `${theme.colors.mediumGray}80`};
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 7px 10px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  color: ${({ theme }) => theme.colors.foreground};
  outline: none;
  width: 100%;
  box-sizing: border-box;

  &::placeholder { color: ${({ theme }) => `${theme.colors.lightGray}50`}; }
  &:focus { border-color: ${({ theme }) => theme.colors.yellow}; }
`;

const NumberInput = styled(TextInput).attrs({ type: 'number' })``;

const CheckboxRow = styled.label`
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  margin-bottom: 10px;

  &:last-child { margin-bottom: 0; }

  input[type='checkbox'] {
    accent-color: ${({ theme }) => theme.colors.yellow};
    width: 14px;
    height: 14px;
    cursor: pointer;
  }
`;

const CheckboxLabel = styled.span`
  font-size: 13px;
  color: ${({ theme }) => theme.colors.foreground};
`;

const CheckboxHint = styled.span`
  font-size: 11px;
  color: ${({ theme }) => `${theme.colors.lightGray}80`};
  margin-left: 4px;
`;

const InfoBox = styled.div`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  background: ${({ theme }) => `${theme.colors.black}80`};
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 8px 12px;
  color: ${({ theme }) => theme.colors.lightGray};
`;

const StatusRow = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
`;

const StatusDot = styled.div<{ $ok?: boolean }>`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${({ $ok }) => $ok ? '#4ade80' : '#6b7280'};
  flex-shrink: 0;
`;

const DiskBar = styled.div`
  flex: 1;
  height: 6px;
  border-radius: 3px;
  background: ${({ theme }) => `${theme.colors.mediumGray}60`};
  overflow: hidden;
  max-width: 200px;
`;

const DiskFill = styled.div<{ $pct: number }>`
  height: 100%;
  border-radius: 3px;
  background: ${({ theme }) => theme.colors.yellow};
  width: ${({ $pct }) => `${$pct}%`};
`;

const AdvancedToggle = styled.button`
  display: flex;
  align-items: center;
  gap: 6px;
  background: none;
  border: none;
  padding: 0;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.lightGray};
  cursor: pointer;
  margin-bottom: 12px;
  transition: ${({ theme }) => theme.transitions.fast};

  &:hover { color: ${({ theme }) => theme.colors.yellow}; }

  svg { transition: transform 200ms ease; }
`;

const OverrideCard = styled.div`
  border: 1px solid ${({ theme }) => `${theme.colors.mediumGray}40`};
  border-radius: ${({ theme }) => theme.radius.sm};
  background: ${({ theme }) => theme.colors.darkGray};
  padding: 12px;
  margin-bottom: 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const OverrideHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const RemoveBtn = styled.button`
  background: none;
  border: none;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.destructive};
  cursor: pointer;
  padding: 0;
  opacity: 0.7;
  transition: ${({ theme }) => theme.transitions.fast};

  &:hover { opacity: 1; }
`;

const AddBtn = styled.button`
  background: transparent;
  border: 1px solid ${({ theme }) => `${theme.colors.yellow}50`};
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 5px 12px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.yellow};
  cursor: pointer;
  transition: ${({ theme }) => theme.transitions.fast};

  &:hover { border-color: ${({ theme }) => theme.colors.yellow}; }
`;

const SaveRow = styled.div`
  display: flex;
  align-items: center;
  gap: 16px;
  margin-top: 8px;
`;

const SaveButton = styled.button`
  background: ${({ theme }) => theme.colors.yellow};
  color: ${({ theme }) => theme.colors.black};
  border: none;
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 8px 24px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 600;
  cursor: pointer;
  transition: ${({ theme }) => theme.transitions.fast};

  &:hover:not(:disabled) { opacity: 0.9; }
  &:disabled { opacity: 0.5; cursor: default; }
`;

const SaveMsg = styled.span<{ $error?: boolean }>`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  color: ${({ theme, $error }) => $error ? theme.colors.destructive : theme.colors.yellow};
`;

const ValidationMsg = styled.span`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  color: ${({ theme }) => theme.colors.destructive};
`;

const LoadingBox = styled.div`
  border: 1px solid ${({ theme }) => `${theme.colors.mediumGray}50`};
  border-radius: ${({ theme }) => theme.radius.sm};
  background: ${({ theme }) => `${theme.colors.black}80`};
  padding: 32px;
  text-align: center;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  color: ${({ theme }) => theme.colors.lightGray};
`;

// ─── Types ────────────────────────────────────────────────────────────────────

interface NodeOverride {
  key: string;
  stagingEnabled: boolean;
  nodeCachePath: string;
  cleanupOnDeactivate: boolean;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

// ─── Component ────────────────────────────────────────────────────────────────

const SettingsPage: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [storeHealth, setStoreHealth] = useState<StoreHealthResponse | null>(null);
  const [configPath, setConfigPath] = useState('');
  const [nodeIdentity, setNodeIdentity] = useState<NodeIdentityResponse | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Form state
  const [enabled, setEnabled] = useState(true);
  const [isThisNodeStoreHost, setIsThisNodeStoreHost] = useState(false);
  const [storeHost, setStoreHost] = useState('');
  const [storeHttpHost, setStoreHttpHost] = useState('');
  const [storePort, setStorePort] = useState(58080);
  const [storePath, setStorePath] = useState('');
  const [allowHfFallback, setAllowHfFallback] = useState(true);
  const [stagingEnabled, setStagingEnabled] = useState(true);
  const [nodeCachePath, setNodeCachePath] = useState('~/.exo/staging');
  const [cleanupOnDeactivate, setCleanupOnDeactivate] = useState(true);
  const [overrides, setOverrides] = useState<NodeOverride[]>([]);

  const loadConfig = useCallback(
    (data: { config: Record<string, unknown>; configPath: string }, identity: NodeIdentityResponse | null) => {
      setConfigPath(data.configPath);
      const ms = data.config?.model_store as Record<string, unknown> | undefined;
      if (!ms) return;
      setEnabled((ms.enabled as boolean) ?? true);
      setStoreHost((ms.store_host as string) ?? '');
      setStoreHttpHost((ms.store_http_host as string) ?? '');
      setStorePort((ms.store_port as number) ?? 58080);
      setStorePath((ms.store_path as string) ?? '');
      if (identity && ms.store_host) {
        setIsThisNodeStoreHost(
          ms.store_host === identity.hostname || ms.store_host === identity.nodeId,
        );
      }
      const dl = ms.download as Record<string, unknown> | undefined;
      setAllowHfFallback((dl?.allow_hf_fallback as boolean) ?? true);
      const stg = ms.staging as Record<string, unknown> | undefined;
      setStagingEnabled((stg?.enabled as boolean) ?? true);
      setNodeCachePath((stg?.node_cache_path as string) ?? '~/.exo/staging');
      setCleanupOnDeactivate((stg?.cleanup_on_deactivate as boolean) ?? true);
      const no = ms.node_overrides as Record<string, Record<string, unknown>> | undefined;
      if (no) {
        const ovr = Object.entries(no).map(([key, val]) => {
          const s = val.staging as Record<string, unknown> | undefined;
          return {
            key,
            stagingEnabled: (s?.enabled as boolean) ?? true,
            nodeCachePath: (s?.node_cache_path as string) ?? '',
            cleanupOnDeactivate: (s?.cleanup_on_deactivate as boolean) ?? true,
          };
        });
        setOverrides(ovr);
        if (ovr.length > 0) setShowAdvanced(true);
      }
    },
    [],
  );

  useEffect(() => {
    const load = async () => {
      try {
        const [configData, health, identity] = await Promise.all([
          fetchConfig(),
          fetchStoreHealth(),
          fetchNodeIdentity().catch(() => null),
        ]);
        setNodeIdentity(identity);
        loadConfig(configData, identity);
        setStoreHealth(health);
      } catch (err) {
        console.error('Failed to load settings:', err);
      } finally {
        setLoading(false);
      }
    };
    void load();
  }, [loadConfig]);

  const handleStoreHostToggle = (isHost: boolean) => {
    setIsThisNodeStoreHost(isHost);
    if (isHost && nodeIdentity) {
      setStoreHost(nodeIdentity.hostname);
      setStoreHttpHost(nodeIdentity.ipAddress);
    } else {
      setStoreHost('');
      setStoreHttpHost('');
    }
  };

  const buildConfig = (): Record<string, unknown> => {
    const nodeOverrides: Record<string, unknown> = {};
    for (const o of overrides) {
      if (o.key.trim()) {
        nodeOverrides[o.key.trim()] = {
          staging: {
            enabled: o.stagingEnabled,
            node_cache_path: o.nodeCachePath,
            cleanup_on_deactivate: o.cleanupOnDeactivate,
          },
        };
      }
    }
    return {
      model_store: {
        enabled,
        store_host: storeHost,
        ...(storeHttpHost ? { store_http_host: storeHttpHost } : {}),
        store_port: storePort,
        store_path: storePath,
        download: { allow_hf_fallback: allowHfFallback },
        staging: {
          enabled: stagingEnabled,
          node_cache_path: nodeCachePath,
          cleanup_on_deactivate: cleanupOnDeactivate,
        },
        ...(Object.keys(nodeOverrides).length > 0
          ? { node_overrides: nodeOverrides }
          : {}),
      },
    };
  };

  const canSave = !enabled || (storeHost.trim() !== '' && storePath.trim() !== '');

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSave) return;
    setSaving(true);
    setSaveMessage(null);
    setSaveError(null);
    try {
      const result = await updateConfig(buildConfig());
      setSaveMessage(result.message);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Failed to save');
    } finally {
      setSaving(false);
    }
  };

  const addOverride = () => {
    setOverrides((prev) => [
      ...prev,
      { key: '', stagingEnabled: true, nodeCachePath: '~/.exo/staging', cleanupOnDeactivate: true },
    ]);
  };

  const removeOverride = (index: number) => {
    setOverrides((prev) => prev.filter((_, i) => i !== index));
  };

  const updateOverride = (index: number, patch: Partial<NodeOverride>) => {
    setOverrides((prev) => prev.map((o, i) => (i === index ? { ...o, ...patch } : o)));
  };

  if (loading) {
    return (
      <Page>
        <PageTitle>Settings</PageTitle>
        <LoadingBox>Loading…</LoadingBox>
      </Page>
    );
  }

  const diskPct = storeHealth
    ? ((storeHealth.usedBytes / storeHealth.totalBytes) * 100)
    : 0;

  return (
    <Page>
      <PageTitle>Settings</PageTitle>
      <PageSubtitle>
        Model store configuration
        {configPath && <> — {configPath}</>}
      </PageSubtitle>

      {/* Store Health */}
      <Section>
        <SectionLegend>Store Status</SectionLegend>
        <StatusRow>
          <StatusDot $ok={!!storeHealth} />
          <span style={{ fontSize: 13 }}>
            {storeHealth ? 'Connected' : enabled ? 'Store unreachable' : 'Store not enabled'}
          </span>
        </StatusRow>
        {storeHealth && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <InfoBox>
              Path: <strong style={{ color: 'white' }}>{storeHealth.storePath}</strong>
            </InfoBox>
            <StatusRow style={{ gap: 10, margin: 0 }}>
              <span style={{ fontSize: 12, color: 'inherit' }}>Disk:</span>
              <DiskBar>
                <DiskFill $pct={Math.min(diskPct, 100)} />
              </DiskBar>
              <span style={{ fontFamily: 'monospace', fontSize: 11, color: 'white' }}>
                {formatBytes(storeHealth.usedBytes)} / {formatBytes(storeHealth.totalBytes)}
              </span>
            </StatusRow>
          </div>
        )}
      </Section>

      {/* Config Form */}
      <form onSubmit={(e) => void handleSave(e)}>
        {/* Model Store */}
        <Section>
          <SectionLegend>Model Store</SectionLegend>
          <CheckboxRow>
            <input
              type="checkbox"
              checked={enabled}
              onChange={(e) => setEnabled(e.target.checked)}
            />
            <CheckboxLabel>Enabled</CheckboxLabel>
          </CheckboxRow>
          {nodeIdentity && (
            <CheckboxRow>
              <input
                type="checkbox"
                checked={isThisNodeStoreHost}
                onChange={(e) => handleStoreHostToggle(e.target.checked)}
              />
              <CheckboxLabel>This node is the store host</CheckboxLabel>
            </CheckboxRow>
          )}
          {isThisNodeStoreHost && nodeIdentity ? (
            <InfoBox>
              Serving as <strong style={{ color: 'white' }}>{nodeIdentity.hostname}</strong>
              <span style={{ opacity: 0.5 }}> ({nodeIdentity.ipAddress})</span>
            </InfoBox>
          ) : (
            <FieldGrid>
              <FieldRow>
                <FieldLabel htmlFor="store-host">Store Host</FieldLabel>
                <TextInput
                  id="store-host"
                  value={storeHost}
                  onChange={(e) => setStoreHost(e.target.value)}
                  placeholder="e.g. mac-studio-1"
                />
              </FieldRow>
              <FieldRow>
                <FieldLabel htmlFor="store-http-host">
                  HTTP Host <span style={{ opacity: 0.5 }}>(optional)</span>
                </FieldLabel>
                <TextInput
                  id="store-http-host"
                  value={storeHttpHost}
                  onChange={(e) => setStoreHttpHost(e.target.value)}
                  placeholder="defaults to store host"
                />
              </FieldRow>
            </FieldGrid>
          )}
          <FieldGrid>
            <FieldRow>
              <FieldLabel htmlFor="store-port">Store Port</FieldLabel>
              <NumberInput
                id="store-port"
                value={storePort}
                onChange={(e) => setStorePort(Number(e.target.value))}
              />
            </FieldRow>
          </FieldGrid>
          <DirectoryBrowser
            value={storePath}
            label="Store Path"
            onChange={setStorePath}
          />
        </Section>

        {/* Download Policy */}
        <Section>
          <SectionLegend>Download Policy</SectionLegend>
          <CheckboxRow>
            <input
              type="checkbox"
              checked={allowHfFallback}
              onChange={(e) => setAllowHfFallback(e.target.checked)}
            />
            <CheckboxLabel>
              Allow HuggingFace fallback
              <CheckboxHint>Fall back to HF when model is not in store</CheckboxHint>
            </CheckboxLabel>
          </CheckboxRow>
        </Section>

        {/* Staging */}
        <Section>
          <SectionLegend>Staging</SectionLegend>
          <CheckboxRow>
            <input
              type="checkbox"
              checked={stagingEnabled}
              onChange={(e) => setStagingEnabled(e.target.checked)}
            />
            <CheckboxLabel>Staging enabled</CheckboxLabel>
          </CheckboxRow>
          <FieldRow>
            <DirectoryBrowser
              value={nodeCachePath}
              label="Node Cache Path"
              onChange={setNodeCachePath}
            />
          </FieldRow>
          <CheckboxRow>
            <input
              type="checkbox"
              checked={cleanupOnDeactivate}
              onChange={(e) => setCleanupOnDeactivate(e.target.checked)}
            />
            <CheckboxLabel>
              Cleanup on deactivate
              <CheckboxHint>Delete staged files when model is shut down</CheckboxHint>
            </CheckboxLabel>
          </CheckboxRow>
        </Section>

        {/* Advanced — Node Overrides */}
        <AdvancedToggle type="button" onClick={() => setShowAdvanced((v) => !v)}>
          <svg
            width="10"
            height="10"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            style={{ transform: showAdvanced ? 'rotate(90deg)' : undefined }}
          >
            <path d="M9 18l6-6-6-6" />
          </svg>
          Advanced
        </AdvancedToggle>

        {showAdvanced && (
          <Section>
            <SectionLegend>Node Overrides</SectionLegend>
            <p style={{ margin: '0 0 12px', fontSize: 11, color: 'rgba(150,150,150,0.8)' }}>
              Per-node staging overrides. Most clusters don't need this.
            </p>
            {overrides.map((override, i) => (
              <OverrideCard key={i}>
                <OverrideHeader>
                  <TextInput
                    value={override.key}
                    onChange={(e) => updateOverride(i, { key: e.target.value })}
                    placeholder="hostname or node_id"
                    style={{ flex: 1 }}
                  />
                  <RemoveBtn type="button" onClick={() => removeOverride(i)}>
                    Remove
                  </RemoveBtn>
                </OverrideHeader>
                <CheckboxRow>
                  <input
                    type="checkbox"
                    checked={override.stagingEnabled}
                    onChange={(e) => updateOverride(i, { stagingEnabled: e.target.checked })}
                  />
                  <CheckboxLabel>Staging enabled</CheckboxLabel>
                </CheckboxRow>
                <DirectoryBrowser
                  value={override.nodeCachePath}
                  label="Cache Path"
                  onChange={(p) => updateOverride(i, { nodeCachePath: p })}
                />
                <CheckboxRow>
                  <input
                    type="checkbox"
                    checked={override.cleanupOnDeactivate}
                    onChange={(e) =>
                      updateOverride(i, { cleanupOnDeactivate: e.target.checked })
                    }
                  />
                  <CheckboxLabel>Cleanup on deactivate</CheckboxLabel>
                </CheckboxRow>
              </OverrideCard>
            ))}
            <AddBtn type="button" onClick={addOverride}>
              + Add Override
            </AddBtn>
          </Section>
        )}

        {/* Save */}
        <SaveRow>
          <SaveButton type="submit" disabled={saving || !canSave}>
            {saving ? 'Saving…' : 'Save'}
          </SaveButton>
          {enabled && !canSave && (
            <ValidationMsg>Store Host and Store Path are required when enabled</ValidationMsg>
          )}
          {saveMessage && <SaveMsg>{saveMessage}</SaveMsg>}
          {saveError && <SaveMsg $error>{saveError}</SaveMsg>}
        </SaveRow>
      </form>
    </Page>
  );
};

export default SettingsPage;
