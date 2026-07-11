import { useCallback, useEffect, useState } from 'react';
import styled, { keyframes } from 'styled-components';
import { generateInstallId } from './TelemetryConsentModal';
import { useConfig, type StoreConfig, type FullConfig, type LoggingConfig, type ExperimentsConfig, type TelemetryConfig } from '../../hooks/useConfig';
import { Button } from '../common/Button';
import { Field } from '../common/Field';
import { InfoTooltip } from '../common/InfoTooltip';
import { addToast } from '../../hooks/useToast';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import { uiActions } from '../../store/slices/uiSlice';
import type { ThemeName } from '../../theme';
import { useSkulkTranslation } from '../../i18n/tolgee';

export interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
}

const defaultStoreConfig = (): StoreConfig => ({
  enabled: false,
  store_host: '',
  store_http_host: '',
  store_port: 58080,
  store_path: '',
  download: {
    allow_hf_fallback: true,
  },
  staging: {
    enabled: true,
    node_cache_path: '~/.skulk/staging',
    cleanup_on_deactivate: true,
  },
});

const defaultExperimentsConfig = (): ExperimentsConfig => ({
  tts_streaming: false,
  stt_realtime: false,
});

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

const SecondaryButton = styled.button`
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 0.78rem;
  cursor: pointer;
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: transparent;
  color: inherit;

  &:hover {
    border-color: ${({ theme }) => theme.colors.borderStrong};
  }
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

  background: ${({ $on, theme }) =>
    $on ? theme.colors.gold : theme.colors.surfaceSunken};
  border: 1px solid ${({ theme }) => theme.colors.border};

  &:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px ${({ theme }) => theme.colors.goldDim};
  }

  &::after {
    content: '';
    position: absolute;
    top: 2px;
    left: ${({ $on }) => ($on ? '18px' : '2px')};
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: ${({ theme }) => theme.colors.surface};
    box-shadow: 0 1px 2px ${({ theme }) => theme.colors.shadow};
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
  const { t } = useSkulkTranslation();
  const { fullConfig, effective, configPath, loading, saving, error, fetchConfig, saveFullConfig } = useConfig(
    t('settings.errors.fetchConfigFailed', 'Failed to fetch config'),
  );
  const themeName = useAppSelector((s) => s.ui.theme);
  const dispatch = useAppDispatch();
  const setTheme = (name: ThemeName) => dispatch(uiActions.setTheme(name));
  const [draft, setDraft] = useState<StoreConfig | null>(null);
  const [kvBackend, setKvBackend] = useState('default');
  const [hfToken, setHfToken] = useState('');
  const [telemetryDraft, setTelemetryDraft] = useState<TelemetryConfig | null>(null);
  const [loggingDraft, setLoggingDraft] = useState<LoggingConfig>({
    enabled: false, ingest_url: '',
  });
  const [experimentsDraft, setExperimentsDraft] = useState<ExperimentsConfig>(defaultExperimentsConfig());

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
    setTelemetryDraft(
      fullConfig?.telemetry ?? {
        consent: 'unasked',
        diagnostics_consent: 'unasked',
        install_id: '',
        consented_at: '',
        consented_version: '',
        ingest_url: 'https://skulk-ledger-ingest.thomastupper92618.workers.dev',
      },
    );
    setExperimentsDraft(fullConfig?.experiments ? { ...fullConfig.experiments } : defaultExperimentsConfig());
  }, [fullConfig, effective]);

  const modelStoreDraft = draft ?? defaultStoreConfig();

  const update = useCallback((patch: Partial<StoreConfig>) => {
    setDraft((prev) => ({ ...(prev ?? defaultStoreConfig()), ...patch }));
  }, []);

  const updateDownload = useCallback((patch: Partial<StoreConfig['download']>) => {
    setDraft((prev) => {
      const base = prev ?? defaultStoreConfig();
      return { ...base, download: { ...base.download, ...patch } };
    });
  }, []);

  const updateStaging = useCallback((patch: Partial<StoreConfig['staging']>) => {
    setDraft((prev) => {
      const base = prev ?? defaultStoreConfig();
      return { ...base, staging: { ...base.staging, ...patch } };
    });
  }, []);

  const handleSave = useCallback(async () => {
    // Base on the last fetched config to avoid dropping sections
    const updated: FullConfig = { ...(fullConfig ?? {}) };
    if (draft) updated.model_store = draft;
    updated.inference = { kv_cache_backend: kvBackend };
    // Include logging config
    updated.logging = { ...loggingDraft };
    // Persist only once the operator has interacted (an untouched unasked
    // draft must not overwrite the "never asked" state that gates the modal).
    if (telemetryDraft && (telemetryDraft.consent !== 'unasked' || telemetryDraft.diagnostics_consent !== 'unasked')) {
      updated.telemetry = {
        ...telemetryDraft,
        consent: telemetryDraft.consent === 'unasked' ? 'disabled' : telemetryDraft.consent,
        diagnostics_consent:
          telemetryDraft.diagnostics_consent === 'unasked' ? 'disabled' : telemetryDraft.diagnostics_consent,
        consented_at: telemetryDraft.consented_at || new Date().toISOString(),
      };
    }
    if (effective?.experimental_mode_enabled) {
      updated.experiments = { ...experimentsDraft };
    }
    // Only send hf_token when user entered a new one
    if (hfToken && hfToken !== '') updated.hf_token = hfToken;
    const ok = await saveFullConfig(updated);
    if (ok) {
      addToast({
        type: 'success',
        message: t(
          'settings.toasts.saved',
          'Settings saved - KV cache change takes effect on next model launch',
        ),
      });
      onClose();
    } else {
      addToast({ type: 'error', message: t('settings.toasts.saveFailed', 'Failed to save settings') });
    }
  }, [draft, effective?.experimental_mode_enabled, experimentsDraft, fullConfig, hfToken, kvBackend, loggingDraft, telemetryDraft, onClose, saveFullConfig, t]);

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
          <Title>{t('settings.title', 'Settings')}</Title>
          <Button
            variant="ghost"
            size="sm"
            icon
            onClick={onClose}
            aria-label={t('settings.close', 'Close settings')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M6 18L18 6M6 6l12 12" />
            </svg>
          </Button>
        </Header>

        <Body>
          {loading && <LoadingText>{t('settings.loadingConfig', 'Loading config...')}</LoadingText>}
          {error && <ErrorText>{error}</ErrorText>}

          {/* Appearance */}
          <Fieldset>
            <Legend>{t('settings.appearance.legend', 'Appearance')}</Legend>
            <Row>
              <FieldLabel>{t('settings.appearance.colorTheme', 'Color theme')}</FieldLabel>
              <Toggle
                $on={themeName === 'light'}
                onClick={() => setTheme(themeName === 'dark' ? 'light' : 'dark')}
                role="switch"
                aria-checked={themeName === 'light'}
                aria-label={themeName === 'light'
                  ? t('settings.appearance.switchToDark', 'Switch to dark theme')
                  : t('settings.appearance.switchToLight', 'Switch to light theme')}
              />
              <Spacer />
              <span style={{ fontSize: 13, opacity: 0.7 }}>
                {themeName === 'light'
                  ? t('settings.appearance.light', 'Light')
                  : t('settings.appearance.dark', 'Dark')}
              </span>
            </Row>
          </Fieldset>

          <>
            {/* Model Store */}
            <Fieldset>
              <Legend>{t('settings.modelStore.legend', 'Model Store')}</Legend>
              <Row>
                <FieldLabel>
                  {t('settings.common.enabled', 'Enabled')}
                  <InfoTooltip
                    filled
                    content={t(
                      'settings.modelStore.enabledTooltip',
                      'When enabled, model store allows specification of a single cluster attached storage device where downloaded models will be saved.',
                    )}
                  />
                </FieldLabel>
                <Toggle $on={modelStoreDraft.enabled} onClick={() => update({ enabled: !modelStoreDraft.enabled })} />
              </Row>
              {modelStoreDraft.enabled && (
                <>
                  <Row>
                    <FieldLabel>{t('settings.modelStore.storeHost', 'Store host')}</FieldLabel>
                    <StyledField
                      size="sm"
                      value={modelStoreDraft.store_host}
                      onChange={(e) => update({ store_host: (e.target as HTMLInputElement).value })}
                      placeholder={t('settings.modelStore.storeHostPlaceholder', 'hostname or node_id')}
                    />
                  </Row>
                  <Row>
                    <FieldLabel>{t('settings.modelStore.httpHost', 'HTTP host')}</FieldLabel>
                    <StyledField
                      size="sm"
                      value={modelStoreDraft.store_http_host}
                      onChange={(e) => update({ store_http_host: (e.target as HTMLInputElement).value })}
                      placeholder={t('settings.modelStore.httpHostPlaceholder', 'defaults to store host')}
                    />
                  </Row>
                  <Row>
                    <FieldLabel>{t('settings.modelStore.port', 'Port')}</FieldLabel>
                    <StyledField
                      size="sm"
                      type="number"
                      value={String(modelStoreDraft.store_port)}
                      onChange={(e) => update({ store_port: parseInt((e.target as HTMLInputElement).value) || 58080 })}
                      style={{ maxWidth: 80 }}
                    />
                  </Row>
                  <Row>
                    <FieldLabel>{t('settings.modelStore.storePath', 'Store path')}</FieldLabel>
                    <StyledField
                      size="sm"
                      value={modelStoreDraft.store_path}
                      onChange={(e) => update({ store_path: (e.target as HTMLInputElement).value })}
                      placeholder={t('settings.modelStore.storePathPlaceholder', '/path/to/models')}
                    />
                  </Row>
                </>
              )}
            </Fieldset>

            {/* Download */}
            <Fieldset>
              <Legend>{t('settings.download.legend', 'Download')}</Legend>
              <Row>
                <FieldLabel>
                  {t('settings.download.allowHuggingFaceFallback', 'Allow HuggingFace fallback')}
                  <InfoTooltip
                    filled
                    content={t(
                      'settings.download.allowHuggingFaceFallbackTooltip',
                      'When enabled, nodes can download models directly from HuggingFace if the model is not in the store. Disable for air-gapped clusters where all models must be pre-loaded into the store.',
                    )}
                  />
                </FieldLabel>
                <Toggle
                  $on={modelStoreDraft.download.allow_hf_fallback}
                  onClick={() => updateDownload({ allow_hf_fallback: !modelStoreDraft.download.allow_hf_fallback })}
                />
              </Row>
              <Row>
                <FieldLabel>
                  {t('settings.experiments.sttRealtime', 'Realtime STT')}
                  <InfoTooltip
                    content={t(
                      'settings.experiments.sttRealtimeTooltip',
                      'Enables the experimental realtime STT provider for mounted models with validated streaming-session support.',
                    )}
                  />
                </FieldLabel>
                <Toggle
                  type="button"
                  $on={experimentsDraft.stt_realtime}
                  aria-pressed={experimentsDraft.stt_realtime}
                  aria-label={t('settings.experiments.sttRealtime', 'Realtime STT')}
                  onClick={() =>
                    setExperimentsDraft((prev) => ({
                      ...prev,
                      stt_realtime: !prev.stt_realtime,
                    }))
                  }
                />
              </Row>
            </Fieldset>

            {/* Staging */}
            <Fieldset>
              <Legend>{t('settings.staging.legend', 'Staging')}</Legend>
              <Row>
                <FieldLabel>
                  {t('settings.common.enabled', 'Enabled')}
                  <InfoTooltip
                    filled
                    content={t(
                      'settings.staging.enabledTooltip',
                      'When enabled, worker nodes copy model files from the store to a local cache directory before loading. This gives MLX a local filesystem path for fast access. Disable only on the store host to load directly from the store path.',
                    )}
                  />
                </FieldLabel>
                <Toggle
                  $on={modelStoreDraft.staging.enabled}
                  onClick={() => updateStaging({ enabled: !modelStoreDraft.staging.enabled })}
                />
              </Row>
              {modelStoreDraft.staging.enabled && (
                <>
                  <Row>
                    <FieldLabel>{t('settings.staging.cachePath', 'Cache path')}</FieldLabel>
                    <StyledField
                      size="sm"
                      value={modelStoreDraft.staging.node_cache_path}
                      onChange={(e) => updateStaging({ node_cache_path: (e.target as HTMLInputElement).value })}
                      placeholder={t('settings.staging.cachePathPlaceholder', '~/.skulk/staging')}
                    />
                  </Row>
                  <Row>
                    <FieldLabel>
                      {t('settings.staging.cleanupOnDeactivate', 'Cleanup on deactivate')}
                      <InfoTooltip
                        filled
                        content={t(
                          'settings.staging.cleanupOnDeactivateTooltip',
                          'When on (default), idle staged copies beyond the keep-recent budget (about 40 GiB) are removed when instances stop and at node startup, keeping the cache warm but bounded. In-use models are always kept. Turn off to keep every staged copy (unbounded) and reclaim disk only with the purge action.',
                        )}
                      />
                    </FieldLabel>
                    <Toggle
                      $on={modelStoreDraft.staging.cleanup_on_deactivate}
                      onClick={() => updateStaging({ cleanup_on_deactivate: !modelStoreDraft.staging.cleanup_on_deactivate })}
                    />
                  </Row>
                </>
              )}
            </Fieldset>

            {configPath && (
              <ConfigPath>
                {t('settings.configPath', 'Config: {configPath}', { configPath })}
              </ConfigPath>
            )}
          </>

          {/* Inference — always shown, not gated on model_store config */}
          <Fieldset>
            <Legend>{t('settings.inference.legend', 'Inference')}</Legend>
            <FieldLabel>
              {t('settings.inference.kvCacheBackend', 'KV Cache Backend')}
              <InfoTooltip
                filled
                content={
                  t(
                    'settings.inference.kvCacheTooltip',
                    'Default - No cache quantization. Best baseline quality, highest memory use.\nOptiQ - Rotation-based quantization via mlx-optiq. Best long-context quality.\nTurboQuant Adaptive - Quantizes middle KV layers, keeps edge layers in FP16. Proven stable.\nTurboQuant - Quantizes all KV layers. Most aggressive compression, higher quality risk.\nMLX Quantized - MLX built-in cache quantization.\n\nTakes effect on next model launch. Incompatible models fall back to Default automatically.',
                  )
                }
              />
            </FieldLabel>
            <Select value={kvBackend} onChange={(e) => setKvBackend(e.target.value)} disabled={!!envOverride}>
              <option value="default">{t('settings.inference.defaultOption', 'Default (no quantization)')}</option>
              <option value="optiq">{t('settings.inference.optiqOption', 'OptiQ (rotation-based)')}</option>
              <option value="turboquant_adaptive">{t('settings.inference.turboquantAdaptiveOption', 'TurboQuant Adaptive')}</option>
              <option value="turboquant">{t('settings.inference.turboquantOption', 'TurboQuant')}</option>
              <option value="mlx_quantized">
                {t('settings.inference.mlxQuantizedOption', 'MLX Quantized (requires SKULK_KV_CACHE_BITS env)')}
              </option>
            </Select>
            {envOverride ? (
              <HintText>
                {t(
                  'settings.inference.envOverrideHint',
                  'Overridden by SKULK_KV_CACHE_BACKEND environment variable. Remove the env var to configure here.',
                )}
              </HintText>
            ) : (
              <HintText>
                {t(
                  'settings.inference.changeHint',
                  'Changes take effect on the next model launch. Models with incompatible architectures (GQA, non-power-of-two head_dim) will automatically fall back to default.',
                )}
              </HintText>
            )}
          </Fieldset>

          {/* HuggingFace */}
          <Fieldset>
            <Legend>{t('settings.huggingFace.legend', 'HuggingFace')}</Legend>
            <Row>
              <FieldLabel>
                {t('settings.huggingFace.apiToken', 'API Token')}
                <InfoTooltip
                  filled
                  content={t(
                    'settings.huggingFace.apiTokenTooltip',
                    'Your HuggingFace API token enables faster downloads, higher rate limits, and access to gated models. Get one at huggingface.co/settings/tokens',
                  )}
                />
              </FieldLabel>
              <StyledField
                size="sm"
                type="password"
                value={hfToken}
                onChange={(e) => setHfToken((e.target as HTMLInputElement).value)}
                placeholder={effective?.has_hf_token
                  ? t('settings.huggingFace.tokenSetPlaceholder', 'Token is set - enter new to replace')
                  : t('settings.huggingFace.tokenPlaceholder', 'hf_...')}
              />
            </Row>
            <HintText>
              {effective?.has_hf_token ? t('settings.huggingFace.tokenConfigured', 'Token is configured. ') : ''}
              {t('settings.huggingFace.syncHint', 'Synced to all nodes. Env var HF_TOKEN takes precedence if set.')}
            </HintText>
          </Fieldset>

          {/* Logging */}
          <Fieldset>
            <Legend>{t('settings.logging.legend', 'Logging')}</Legend>
            <Row>
              <FieldLabel>
                {t('settings.common.enabled', 'Enabled')}
                <InfoTooltip
                  filled
                  content={t(
                    'settings.logging.enabledTooltip',
                    'When enabled, nodes emit structured JSON logs on stdout for collection by Vector. Requires an ingest URL to be set.',
                  )}
                />
              </FieldLabel>
              <Toggle $on={loggingDraft.enabled} onClick={() => setLoggingDraft(prev => ({ ...prev, enabled: !prev.enabled }))} />
            </Row>
            {loggingDraft.enabled && (
              <>
                <Row>
                  <FieldLabel>{t('settings.logging.ingestUrl', 'Ingest URL')}</FieldLabel>
                  <StyledField
                    size="sm"
                    value={loggingDraft.ingest_url}
                    onChange={(e) => setLoggingDraft(prev => ({ ...prev, ingest_url: (e.target as HTMLInputElement).value }))}
                    placeholder={t('settings.logging.ingestUrlPlaceholder', 'http://host:9428/insert/jsonline?_stream_fields=...')}
                  />
                </Row>
                <HintText>
                  {t(
                    'settings.logging.syncHint',
                    'Settings are synced to all nodes. Nodes will start shipping logs when saved.',
                  )}
                </HintText>
              </>
            )}
          </Fieldset>

          {/* Field telemetry: consent lives in skulk.yaml (survives restarts,
              synced like other settings). Both switches stay permanently
              available here; the first-run modal only acquires the initial
              choice. */}
          {telemetryDraft && (
            <Fieldset>
              <Legend>{t('settings.telemetry.legend', 'Telemetry')}</Legend>
              <Row>
                <FieldLabel>
                  {t('settings.telemetry.perf', 'Performance telemetry')}
                  <InfoTooltip
                    filled
                    content={t(
                      'settings.telemetry.perfTooltip',
                      'Anonymous performance and reliability samples (model id, hardware class, timing, token counts, failure classes). Never prompts, outputs, or machine identity. Public only as aggregates.',
                    )}
                  />
                </FieldLabel>
                <Toggle
                  $on={telemetryDraft.consent === 'enabled'}
                  onClick={() =>
                    setTelemetryDraft(prev =>
                      prev && {
                        ...prev,
                        consent: prev.consent === 'enabled' ? 'disabled' : 'enabled',
                        install_id: prev.install_id || generateInstallId(),
                      },
                    )
                  }
                />
              </Row>
              <Row>
                <FieldLabel>
                  {t('settings.telemetry.diagnostics', 'Crash diagnostics')}
                  <InfoTooltip
                    filled
                    content={t(
                      'settings.telemetry.diagnosticsTooltip',
                      'Separate consent for scrubbed crash reports, kept privately for 90 days. Enabling telemetry never enables this.',
                    )}
                  />
                </FieldLabel>
                <Toggle
                  $on={telemetryDraft.diagnostics_consent === 'enabled'}
                  onClick={() =>
                    setTelemetryDraft(prev =>
                      prev && {
                        ...prev,
                        diagnostics_consent:
                          prev.diagnostics_consent === 'enabled' ? 'disabled' : 'enabled',
                        install_id: prev.install_id || generateInstallId(),
                      },
                    )
                  }
                />
              </Row>
              {telemetryDraft.install_id && (
                <>
                  <HintText>
                    {t('settings.telemetry.installId', 'Install id (your deletion key): ')}
                    {telemetryDraft.install_id}
                  </HintText>
                  <Row>
                    <SecondaryButton
                      type="button"
                      onClick={() =>
                        setTelemetryDraft(prev => prev && { ...prev, install_id: generateInstallId() })
                      }
                    >
                      {t('settings.telemetry.rotate', 'Rotate id')}
                    </SecondaryButton>
                    <SecondaryButton
                      type="button"
                      onClick={() => setTelemetryDraft(prev => prev && { ...prev, install_id: '' })}
                    >
                      {t('settings.telemetry.clear', 'Clear id')}
                    </SecondaryButton>
                  </Row>
                  <HintText>
                    {t(
                      'settings.telemetry.rotateHint',
                      'Rotating or clearing disowns previously sent samples; a cleared id regenerates on save while consent is enabled.',
                    )}
                  </HintText>
                </>
              )}
              <HintText>
                {t(
                  'settings.telemetry.previewHint',
                  'Inspect exactly what would be sent at GET /v1/telemetry/preview.',
                )}
              </HintText>
            </Fieldset>
          )}

          {/* Experiments: only shown when the node runs with
              SKULK_ENABLE_EXPERIMENTAL_MODE. This is the fabric's staging
              area for in-development features: each such feature adds its own
              toggle here so its UX is built alongside it. */}
          {effective?.experimental_mode_enabled && (
            <Fieldset>
              <Legend>
                {t('settings.experiments.legend', 'Experiments')}
                <InfoTooltip
                  filled
                  content={t(
                    'settings.experiments.legendTooltip',
                    'Opt-in, in-development features. This section is visible because this node runs with SKULK_ENABLE_EXPERIMENTAL_MODE. Features here are unfinished and off by default.',
                  )}
                />
              </Legend>
              <Row>
                <FieldLabel>
                  {t('settings.experiments.ttsStreaming', 'TTS streaming')}
                  <InfoTooltip
                    content={t(
                      'settings.experiments.ttsStreamingTooltip',
                      'Enables the experimental stream=true path for /v1/audio/speech. Leave off until a mounted TTS model has passed streaming validation.',
                    )}
                  />
                </FieldLabel>
                <Toggle
                  type="button"
                  $on={experimentsDraft.tts_streaming}
                  aria-pressed={experimentsDraft.tts_streaming}
                  aria-label={t('settings.experiments.ttsStreaming', 'TTS streaming')}
                  onClick={() =>
                    setExperimentsDraft((prev) => ({
                      ...prev,
                      tts_streaming: !prev.tts_streaming,
                    }))
                  }
                />
              </Row>
            </Fieldset>
          )}
        </Body>

        <Footer>
          <Spacer />
          <Button variant="outline" size="md" onClick={onClose}>
            {t('common.cancel', 'Cancel')}
          </Button>
          <Button variant="primary" size="md" loading={saving} onClick={handleSave} disabled={loading}>
            {t('common.save', 'Save')}
          </Button>
        </Footer>
      </Drawer>
    </>
  );
}
