import { useCallback, useEffect, useState } from 'react';
import styled, { css, keyframes } from 'styled-components';
import { useConfig, type StoreConfig, type FullConfig, type LoggingConfig } from '../../hooks/useConfig';
import { Button } from '../common/Button';
import { Field } from '../common/Field';
import { InfoTooltip } from '../common/InfoTooltip';
import { addToast } from '../../hooks/useToast';
import { useUIStore } from '../../stores/uiStore';

export interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
}

/* ---- animations ---- */

const fadeIn = keyframes`
  from { opacity: 0; }
  to   { opacity: 1; }
`;

const slideIn = keyframes`
  from { transform: translateX(100%); }
  to   { transform: translateX(0); }
`;

/* ---- styles ---- */

const Backdrop = styled.div`
  position: fixed;
  inset: 0;
  z-index: 40;
  background: ${({ theme }) => theme.colors.shadowStrong};
  backdrop-filter: blur(2px);
  animation: ${fadeIn} 0.2s ease-out;
`;

const Drawer = styled.aside`
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  z-index: 50;
  width: 380px;
  max-width: 100vw;
  background: ${({ theme }) => theme.colors.surface};
  border-left: 1px solid ${({ theme }) => theme.colors.border};
  display: flex;
  flex-direction: column;
  animation: ${slideIn} 0.25s cubic-bezier(0.33, 1, 0.68, 1);
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
`;

const Title = styled.h2`
  font-size: ${({ theme }) => theme.fontSizes.md};
  font-family: ${({ theme }) => theme.fonts.body};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.gold};
`;

const Body = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

const Footer = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 16px 20px;
  border-top: 1px solid ${({ theme }) => theme.colors.border};
`;

const Fieldset = styled.fieldset`
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const Legend = styled.legend`
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-family: ${({ theme }) => theme.fonts.body};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.textSecondary};
  padding: 0 6px;
`;

const Row = styled.label`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
`;

const FieldLabel = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textSecondary};
  white-space: nowrap;
`;

const Toggle = styled.button<{ $on: boolean }>`
  all: unset;
  cursor: pointer;
  width: 36px;
  height: 20px;
  border-radius: 10px;
  position: relative;
  flex-shrink: 0;
  transition: background 0.2s;

  ${({ $on }) =>
    $on
      ? css`background: ${({ theme }) => theme.colors.gold};`
      : css`background: rgba(80, 80, 80, 0.5);`}

  &::after {
    content: '';
    position: absolute;
    top: 2px;
    left: ${({ $on }) => ($on ? '18px' : '2px')};
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: ${({ $on }) => ($on ? '#000' : '#999')};
    transition: left 0.2s;
  }
`;

const StyledField = styled(Field)`
  flex: 1;
  min-width: 0;
`;

const Select = styled.select`
  width: 100%;
  box-sizing: border-box;
  background: ${({ theme }) => theme.colors.bg};
  color: ${({ theme }) => theme.colors.text};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 4px 8px;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  outline: none;
  cursor: pointer;

  &:focus {
    border-color: ${({ theme }) => theme.colors.goldDim};
  }

  option {
    background: ${({ theme }) => theme.colors.surface};
    color: ${({ theme }) => theme.colors.text};
  }
`;

const HintText = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
  font-style: italic;
`;

const ConfigPath = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const ErrorText = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.error};
`;

const LoadingText = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  height: 200px;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const Spacer = styled.span`
  flex: 1;
`;

/* ---- component ---- */

export function SettingsPanel({ open, onClose }: SettingsPanelProps) {
  const { fullConfig, effective, configPath, loading, saving, error, fetchConfig, saveFullConfig } = useConfig();
  const themeName = useUIStore((s) => s.theme);
  const setTheme = useUIStore((s) => s.setTheme);
  const [draft, setDraft] = useState<StoreConfig | null>(null);
  const [kvBackend, setKvBackend] = useState('default');
  const [hfToken, setHfToken] = useState('');
  const [loggingDraft, setLoggingDraft] = useState<LoggingConfig>({
    enabled: false, ingest_url: '',
  });

  // Fetch config when panel opens
  useEffect(() => {
    if (open) fetchConfig();
  }, [open, fetchConfig]);

  // Seed draft from fetched config — use effective value for KV backend
  const envOverride = effective != null
    && effective.kv_cache_backend !== 'default'
    && effective.kv_cache_backend !== (fullConfig?.inference?.kv_cache_backend ?? 'default');
  useEffect(() => {
    setDraft(fullConfig?.model_store ? { ...fullConfig.model_store } : null);
    setKvBackend(effective?.kv_cache_backend ?? fullConfig?.inference?.kv_cache_backend ?? 'default');
    setHfToken(fullConfig?.hf_token ?? '');
    setLoggingDraft({
      enabled: fullConfig?.logging?.enabled ?? false,
      ingest_url: fullConfig?.logging?.ingest_url ?? '',
    });
  }, [fullConfig, effective]);

  const update = useCallback((patch: Partial<StoreConfig>) => {
    setDraft((prev) => prev ? { ...prev, ...patch } : prev);
  }, []);

  const updateDownload = useCallback((patch: Partial<StoreConfig['download']>) => {
    setDraft((prev) => prev ? { ...prev, download: { ...prev.download, ...patch } } : prev);
  }, []);

  const updateStaging = useCallback((patch: Partial<StoreConfig['staging']>) => {
    setDraft((prev) => prev ? { ...prev, staging: { ...prev.staging, ...patch } } : prev);
  }, []);

  const handleSave = useCallback(async () => {
    // Base on the last fetched config to avoid dropping sections
    const updated: FullConfig = { ...(fullConfig ?? {}) };
    if (draft) updated.model_store = draft;
    updated.inference = { kv_cache_backend: kvBackend };
    // Include logging config
    updated.logging = { ...loggingDraft };
    // Only send hf_token when user entered a new one
    if (hfToken && hfToken !== '') updated.hf_token = hfToken;
    const ok = await saveFullConfig(updated);
    if (ok) {
      addToast({ type: 'success', message: 'Settings saved — KV cache change takes effect on next model launch' });
      onClose();
    } else {
      addToast({ type: 'error', message: 'Failed to save settings' });
    }
  }, [draft, kvBackend, saveFullConfig, onClose]);

  // ESC to close
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <>
      <Backdrop onClick={onClose} />
      <Drawer>
        <Header>
          <Title>Settings</Title>
          <Button variant="ghost" size="sm" icon onClick={onClose} aria-label="Close settings">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M6 18L18 6M6 6l12 12" />
            </svg>
          </Button>
        </Header>

        <Body>
          {loading && <LoadingText>Loading config…</LoadingText>}
          {error && <ErrorText>{error}</ErrorText>}

          {/* Appearance */}
          <Fieldset>
            <Legend>Appearance</Legend>
            <Row>
              <FieldLabel>Color theme</FieldLabel>
              <Toggle
                $on={themeName === 'light'}
                onClick={() => setTheme(themeName === 'dark' ? 'light' : 'dark')}
                aria-label="Toggle light theme"
              />
              <Spacer />
              <span style={{ fontSize: 13, opacity: 0.7 }}>{themeName === 'light' ? 'Light' : 'Dark'}</span>
            </Row>
          </Fieldset>

          {draft && (
            <>
              {/* Model Store */}
              <Fieldset>
                <Legend>Model Store</Legend>
                <Row>
                  <FieldLabel>
                    Enabled
                    <InfoTooltip
                      filled
                      content="When enabled, model store allows specification of a single cluster attached storage device where downloaded models will be saved."
                    />
                  </FieldLabel>
                  <Toggle $on={draft.enabled} onClick={() => update({ enabled: !draft.enabled })} />
                </Row>
                {draft.enabled && (
                  <>
                    <Row>
                      <FieldLabel>Store host</FieldLabel>
                      <StyledField
                        size="sm"
                        value={draft.store_host}
                        onChange={(e) => update({ store_host: (e.target as HTMLInputElement).value })}
                        placeholder="hostname or node_id"
                      />
                    </Row>
                    <Row>
                      <FieldLabel>HTTP host</FieldLabel>
                      <StyledField
                        size="sm"
                        value={draft.store_http_host}
                        onChange={(e) => update({ store_http_host: (e.target as HTMLInputElement).value })}
                        placeholder="defaults to store host"
                      />
                    </Row>
                    <Row>
                      <FieldLabel>Port</FieldLabel>
                      <StyledField
                        size="sm"
                        type="number"
                        value={String(draft.store_port)}
                        onChange={(e) => update({ store_port: parseInt((e.target as HTMLInputElement).value) || 58080 })}
                        style={{ maxWidth: 80 }}
                      />
                    </Row>
                    <Row>
                      <FieldLabel>Store path</FieldLabel>
                      <StyledField
                        size="sm"
                        value={draft.store_path}
                        onChange={(e) => update({ store_path: (e.target as HTMLInputElement).value })}
                        placeholder="/path/to/models"
                      />
                    </Row>
                  </>
                )}
              </Fieldset>

              {/* Download */}
              <Fieldset>
                <Legend>Download</Legend>
                <Row>
                  <FieldLabel>
                    Allow HuggingFace fallback
                    <InfoTooltip
                      filled
                      content="When enabled, nodes can download models directly from HuggingFace if the model is not in the store. Disable for air-gapped clusters where all models must be pre-loaded into the store."
                    />
                  </FieldLabel>
                  <Toggle
                    $on={draft.download.allow_hf_fallback}
                    onClick={() => updateDownload({ allow_hf_fallback: !draft.download.allow_hf_fallback })}
                  />
                </Row>
              </Fieldset>

              {/* Staging */}
              <Fieldset>
                <Legend>Staging</Legend>
                <Row>
                  <FieldLabel>
                    Enabled
                    <InfoTooltip
                      filled
                      content="When enabled, worker nodes copy model files from the store to a local cache directory before loading. This gives MLX a local filesystem path for fast access. Disable only on the store host to load directly from the store path."
                    />
                  </FieldLabel>
                  <Toggle
                    $on={draft.staging.enabled}
                    onClick={() => updateStaging({ enabled: !draft.staging.enabled })}
                  />
                </Row>
                {draft.staging.enabled && (
                  <>
                    <Row>
                      <FieldLabel>Cache path</FieldLabel>
                      <StyledField
                        size="sm"
                        value={draft.staging.node_cache_path}
                        onChange={(e) => updateStaging({ node_cache_path: (e.target as HTMLInputElement).value })}
                        placeholder="~/.skulk/staging"
                      />
                    </Row>
                    <Row>
                      <FieldLabel>
                        Cleanup on deactivate
                        <InfoTooltip
                          filled
                          content="When enabled, on deactivate staged models will be removed from cluster nodes to prevent storage bloat on cluster nodes."
                        />
                      </FieldLabel>
                      <Toggle
                        $on={draft.staging.cleanup_on_deactivate}
                        onClick={() => updateStaging({ cleanup_on_deactivate: !draft.staging.cleanup_on_deactivate })}
                      />
                    </Row>
                  </>
                )}
              </Fieldset>

              {configPath && <ConfigPath>Config: {configPath}</ConfigPath>}
            </>
          )}

          {/* Inference — always shown, not gated on model_store config */}
          <Fieldset>
            <Legend>Inference</Legend>
            <FieldLabel>
              KV Cache Backend
              <InfoTooltip
                filled
                content={
                  `• Default — No cache quantization. Best baseline quality, highest memory use.\n` +
                  `• OptiQ — Rotation-based quantization via mlx-optiq. Best long-context quality.\n` +
                  `• TurboQuant Adaptive — Quantizes middle KV layers, keeps edge layers in FP16. Proven stable.\n` +
                  `• TurboQuant — Quantizes all KV layers. Most aggressive compression, higher quality risk.\n` +
                  `• MLX Quantized — MLX's built-in cache quantization.\n\n` +
                  `Takes effect on next model launch. Incompatible models fall back to Default automatically.`
                }
              />
            </FieldLabel>
            <Select value={kvBackend} onChange={(e) => setKvBackend(e.target.value)} disabled={!!envOverride}>
              <option value="default">Default (no quantization)</option>
              <option value="optiq">OptiQ (rotation-based)</option>
              <option value="turboquant_adaptive">TurboQuant Adaptive</option>
              <option value="turboquant">TurboQuant</option>
              <option value="mlx_quantized">MLX Quantized (requires SKULK_KV_CACHE_BITS env)</option>
            </Select>
            {envOverride ? (
              <HintText>Overridden by SKULK_KV_CACHE_BACKEND environment variable. Remove the env var to configure here.</HintText>
            ) : (
              <HintText>Changes take effect on the next model launch. Models with incompatible architectures (GQA, non-power-of-two head_dim) will automatically fall back to default.</HintText>
            )}
          </Fieldset>

          {/* HuggingFace */}
          <Fieldset>
            <Legend>HuggingFace</Legend>
            <Row>
              <FieldLabel>
                API Token
                <InfoTooltip
                  filled
                  content="Your HuggingFace API token enables faster downloads, higher rate limits, and access to gated models. Get one at huggingface.co/settings/tokens"
                />
              </FieldLabel>
              <StyledField
                size="sm"
                type="password"
                value={hfToken}
                onChange={(e) => setHfToken((e.target as HTMLInputElement).value)}
                placeholder={effective?.has_hf_token ? "Token is set — enter new to replace" : "hf_..."}
              />
            </Row>
            <HintText>
              {effective?.has_hf_token ? 'Token is configured. ' : ''}
              Synced to all nodes. Env var HF_TOKEN takes precedence if set.
            </HintText>
          </Fieldset>

          {/* Logging */}
          <Fieldset>
            <Legend>Logging</Legend>
            <Row>
              <FieldLabel>
                Enabled
                <InfoTooltip
                  filled
                  content="When enabled, nodes emit structured JSON logs on stdout for collection by Vector. Requires an ingest URL to be set."
                />
              </FieldLabel>
              <Toggle $on={loggingDraft.enabled} onClick={() => setLoggingDraft(prev => ({ ...prev, enabled: !prev.enabled }))} />
            </Row>
            {loggingDraft.enabled && (
              <>
                <Row>
                  <FieldLabel>Ingest URL</FieldLabel>
                  <StyledField
                    size="sm"
                    value={loggingDraft.ingest_url}
                    onChange={(e) => setLoggingDraft(prev => ({ ...prev, ingest_url: (e.target as HTMLInputElement).value }))}
                    placeholder="http://host:9428/insert/jsonline?_stream_fields=..."
                  />
                </Row>
                <HintText>
                  Settings are synced to all nodes. Nodes will start shipping logs when saved.
                </HintText>
              </>
            )}
          </Fieldset>
        </Body>

        <Footer>
          <Spacer />
          <Button variant="outline" size="md" onClick={onClose}>Cancel</Button>
          <Button variant="primary" size="md" loading={saving} onClick={handleSave} disabled={loading}>
            Save
          </Button>
        </Footer>
      </Drawer>
    </>
  );
}
