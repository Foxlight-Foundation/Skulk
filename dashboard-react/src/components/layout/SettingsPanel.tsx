import { Toggle } from '../common/Toggle';
import { useCallback, useEffect, useRef, useState } from 'react';
import styled from 'styled-components';
import { generateInstallId } from './TelemetryConsentModal';
import {
  useConfig,
  type StoreConfig,
  type FullConfig,
  type IntelligentFabricConfig,
  type LoggingConfig,
  type TelemetryConfig,
} from '../../hooks/useConfig';
import {
  DEFAULT_MODEL_STORE_PORT,
  normalizeStoreConfig,
} from './modelStoreConfig';
import { Button } from '../common/Button';
import { Field } from '../common/Field';
import { InfoTooltip } from '../common/InfoTooltip';
import { addToast } from '../../hooks/useToast';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import { uiActions } from '../../store/slices/uiSlice';
import type { ThemeName } from '../../theme';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { DevicesPanel } from './DevicesPanel';
import { RightDrawer } from '../common/RightDrawer';
import { CollapsibleSection } from '../common/Surfaces';
import { FiArrowLeft, FiChevronRight, FiSmartphone } from 'react-icons/fi';

/** Visibility and dismissal for the draft-owning Settings drawer. */
export interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
}

/* ---- styles ---- */

const Body = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 0;
`;

const Sections = styled.div`
  border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 12px;
  background: ${({ theme }) => theme.colors.surface};
`;
const DevicesEntry = styled.button`
  display: flex; align-items: center; gap: 12px; padding: 12px 14px; margin-bottom: 14px;
  width: 100%; border-radius: 12px; border: 1px solid ${({ theme }) => theme.colors.borderControl};
  background: ${({ theme }) => theme.colors.surface}; color: ${({ theme }) => theme.colors.text};
  cursor: pointer; text-align: left; font: 600 14px ${({ theme }) => theme.fonts.body};
  > svg:first-child { width: 36px; height: 36px; padding: 10px; border-radius: 10px; flex-shrink: 0; background: ${({ theme }) => theme.colors.selected}; }
  > svg:last-child { flex-shrink: 0; margin-left: auto; }
  span { min-width: 0; overflow-wrap: anywhere; }
  small { display: block; margin-top: 2px; font-size: 12.5px; font-weight: 400; color: ${({ theme }) => theme.colors.textSecondary}; }
  &:hover { border-color: ${({ theme }) => theme.colors.borderStrong}; }
`;
const Footer = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 14px 20px;
  border-top: 1px solid ${({ theme }) => theme.colors.border};
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
  white-space: normal;
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

const StyledField = styled(Field)`
  flex: 1;
  min-width: 0;
`;

const Select = styled.select`
  width: 100%;
  box-sizing: border-box;
  background: ${({ theme }) => theme.colors.surfaceHover};
  color: ${({ theme }) => theme.colors.text};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 9px 10px;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  outline: none;
  cursor: pointer;

  &:focus {
    outline: none;
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
  color: ${({ theme }) => theme.colors.subtleText};
  font-style: italic;
`;

const ConfigPath = styled.div`
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.subtleText};
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
  color: ${({ theme }) => theme.colors.subtleText};
`;

const Spacer = styled.span`
  flex: 1;
`;

/* ---- component ---- */

/** Retain an unsaved configuration draft while visiting immediate device actions. */
export function SettingsPanel({ open, onClose }: SettingsPanelProps) {
  const { t } = useSkulkTranslation();
  const { fullConfig, effective, configPath, loading, saving, error, fetchConfig, saveFullConfig } = useConfig(
    t('settings.errors.fetchConfigFailed', 'Failed to fetch config'),
  );
  const themeName = useAppSelector((s) => s.ui.theme);
  const dispatch = useAppDispatch();
  const [themeDraft, setThemeDraft] = useState<ThemeName>(themeName);
  const seeded = useRef(false);
  const [devicesOpen, setDevicesOpen] = useState(false);
  const [width, setWidth] = useState(420);
  const [devicesWidth, setDevicesWidth] = useState(720);
  const devicesEntry = useRef<HTMLButtonElement>(null);
  const devicesBack = useRef<HTMLButtonElement>(null);
  const returnToSettings = () => {
    setDevicesOpen(false);
    requestAnimationFrame(() => devicesEntry.current?.focus());
  };
  const [expanded, setExpanded] = useState<Record<string, boolean>>(() => {
    try {
      const value: unknown = JSON.parse(localStorage.getItem('skulk-settings-sections') ?? '{}');
      if (value && typeof value === 'object' && !Array.isArray(value)) return Object.fromEntries(Object.entries(value).filter(([, entry]) => typeof entry === 'boolean'));
    } catch { /* Storage is optional for disclosure preferences. */ }
    return {};
  });
  const section = (key: string) => ({ open: expanded[key] ?? false, onOpenChange: (value: boolean) => {
    setExpanded(previous => {
      const next = { ...previous, [key]: value };
      try { localStorage.setItem('skulk-settings-sections', JSON.stringify(next)); } catch { /* Keep the current session usable when storage is unavailable. */ }
      return next;
    });
  } });
  const [draft, setDraft] = useState<StoreConfig | null>(null);
  const [kvBackend, setKvBackend] = useState('default');
  const [hfToken, setHfToken] = useState('');
  const [telemetryDraft, setTelemetryDraft] = useState<TelemetryConfig | null>(null);
  const [fabricDraft, setFabricDraft] = useState<IntelligentFabricConfig>({
    enabled: false,
  });
  const [loggingDraft, setLoggingDraft] = useState<LoggingConfig>({
    enabled: false, ingest_url: '',
  });
  // Fetch config when panel opens
  useEffect(() => {
    if (!open) return;
    fetchConfig();
  }, [open, fetchConfig]);

  // Seed draft from fetched config — use effective value for KV backend
  const envOverride = effective != null
    && effective.kv_cache_backend !== 'default'
    && effective.kv_cache_backend !== (fullConfig?.inference?.kv_cache_backend ?? 'default');
  useEffect(() => {
    if (!open) { seeded.current = false; return; }
    if (seeded.current || !fullConfig || loading) return;
    seeded.current = true;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- Seed an editable snapshot once after the external query completes; later refreshes must not overwrite drafts.
    setThemeDraft(themeName);
    setDraft(
      fullConfig?.model_store
        ? normalizeStoreConfig(fullConfig.model_store)
        : null,
    );
    setKvBackend(effective?.kv_cache_backend ?? fullConfig?.inference?.kv_cache_backend ?? 'default');
    setHfToken(fullConfig?.hf_token ?? '');
    setFabricDraft({
      enabled: fullConfig?.intelligent_fabric?.enabled ?? false,
      steward_models: fullConfig?.intelligent_fabric?.steward_models,
    });
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
  }, [open, fullConfig, effective, loading, themeName]);

  const modelStoreDraft = normalizeStoreConfig(draft);

  const update = useCallback((patch: Partial<StoreConfig>) => {
    setDraft((prev) => ({ ...normalizeStoreConfig(prev), ...patch }));
  }, []);

  const updateDownload = useCallback((patch: Partial<StoreConfig['download']>) => {
    setDraft((prev) => {
      const base = normalizeStoreConfig(prev);
      return { ...base, download: { ...base.download, ...patch } };
    });
  }, []);

  const updateStaging = useCallback((patch: Partial<StoreConfig['staging']>) => {
    setDraft((prev) => {
      const base = normalizeStoreConfig(prev);
      return { ...base, staging: { ...base.staging, ...patch } };
    });
  }, []);

  const handleSave = useCallback(async () => {
    // An enabled store with a blank host or path is refused by the server
    // (an empty host interpolates into unusable store URLs and once shipped
    // a fleet that could not place any new model), so surface the problem
    // here instead of persisting a config the API will 422.
    if (draft?.enabled && (!draft.store_host.trim() || !draft.store_path.trim())) {
      addToast({
        type: 'error',
        message: t(
          'settings.toasts.storeIdentityRequired',
          'Model store is enabled but the store host or store path is blank - name them, or disable the store',
        ),
      });
      return;
    }
    // Base on the last fetched config to avoid dropping sections
    const updated: FullConfig = { ...(fullConfig ?? {}) };
    if (draft) updated.model_store = draft;
    updated.inference = { kv_cache_backend: kvBackend };
    // Include logging config
    updated.logging = { ...loggingDraft };
    updated.intelligent_fabric = { ...fabricDraft };
    // Retain compatibility with nodes that still expose the retired model-trust
    // field, but never write the inert legacy allow-list back through Settings.
    delete updated.model_trust;
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
    // Only send hf_token when user entered a new one
    if (hfToken && hfToken !== '') updated.hf_token = hfToken;
    const configSaved = await saveFullConfig(updated);
    if (configSaved) {
      dispatch(uiActions.setTheme(themeDraft));
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
  }, [draft, fullConfig, hfToken, kvBackend, loggingDraft, fabricDraft, telemetryDraft, themeDraft, dispatch, onClose, saveFullConfig, t]);

  // eslint-disable-next-line react-hooks/set-state-in-effect -- Closing the modal resets its nested navigation for the next opening.
  useEffect(() => { if (!open) setDevicesOpen(false); }, [open]);

  if (!open) return null;

  return (
    <RightDrawer open={open} onClose={onClose} width={devicesOpen ? devicesWidth : width} minWidth={360} maxWidth={720} onWidthChange={devicesOpen ? setDevicesWidth : setWidth}
      ariaLabel={devicesOpen ? t('devices.title', 'Devices & pairing') : t('settings.title', 'Settings')}
      title={devicesOpen ? t('devices.title', 'Devices & pairing') : t('settings.title', 'Settings')}
      headerLeading={devicesOpen ? <Button ref={devicesBack} variant="ghost" size="sm" aria-label={t('devices.backToSettings', 'Back to Settings')} onClick={returnToSettings}><FiArrowLeft /></Button> : undefined}
      closeLabel={t('settings.close', 'Close settings')} resizeLabel={t('settings.resize', 'Resize settings')}>
      {devicesOpen ? <DevicesPanel /> : null}
        <Body style={{ display: devicesOpen ? 'none' : undefined }}>
          {loading && <LoadingText>{t('settings.loadingConfig', 'Loading config...')}</LoadingText>}
          {error && <ErrorText>{error}</ErrorText>}

          <DevicesEntry ref={devicesEntry} aria-label={t('devices.title', 'Devices & pairing')} onClick={() => { setDevicesOpen(true); requestAnimationFrame(() => devicesBack.current?.focus()); }}><FiSmartphone aria-hidden /><span>{t('devices.title', 'Devices & pairing')}<small>{t('devices.manageHint', 'Manage paired devices and invitations')}</small></span><FiChevronRight size={14} aria-hidden /></DevicesEntry>

          <Sections>
          {/* Appearance */}
          <CollapsibleSection {...section('appearance')} title={t('settings.appearance.legend', 'Appearance')} summary={themeDraft === 'dark' ? 'Night' : 'Noon Ridge'}>
            <Row>
              <FieldLabel>{t('settings.appearance.colorTheme', 'Color theme')}</FieldLabel>
              <Toggle
                $on={themeDraft === 'light'}
                onClick={() => setThemeDraft(themeDraft === 'dark' ? 'light' : 'dark')}
                role="switch"
                aria-checked={themeDraft === 'light'}
                aria-label={themeDraft === 'light'
                  ? t('settings.appearance.switchToDark', 'Switch to dark theme')
                  : t('settings.appearance.switchToLight', 'Switch to light theme')}
              />
              <Spacer />
              <span style={{ fontSize: 13, opacity: 0.7 }}>
                {themeDraft === 'light'
                  ? t('settings.appearance.noonRidge', 'Noon Ridge')
                  : t('settings.appearance.night', 'Night')}
              </span>
            </Row>
          </CollapsibleSection>

          <>
            {/* Model Store */}
            <CollapsibleSection {...section('modelStore')} title={t('settings.modelStore.legend', 'Model Store')} summary={modelStoreDraft.enabled ? modelStoreDraft.store_host : t('common.off', 'Off')}>
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
                <Toggle aria-label={t('settings.common.enabled', 'Enabled')} $on={modelStoreDraft.enabled} onClick={() => update({ enabled: !modelStoreDraft.enabled })} />
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
                      onChange={(e) => update({ store_port: parseInt((e.target as HTMLInputElement).value) || DEFAULT_MODEL_STORE_PORT })}
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
            </CollapsibleSection>

            {/* Download */}
            <CollapsibleSection {...section('download')} title={t('settings.download.legend', 'Download')} summary={modelStoreDraft.download.allow_hf_fallback ? t('common.on', 'On') : t('common.off', 'Off')}>
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
                <Toggle aria-label={t('settings.common.enabled', 'Enabled')}
                  $on={modelStoreDraft.download.allow_hf_fallback}
                  onClick={() => updateDownload({ allow_hf_fallback: !modelStoreDraft.download.allow_hf_fallback })}
                />
              </Row>
            </CollapsibleSection>

            {/* Staging */}
            <CollapsibleSection {...section('staging')} title={t('settings.staging.legend', 'Staging')} summary={modelStoreDraft.staging.enabled ? modelStoreDraft.staging.node_cache_path : t('common.off', 'Off')}>
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
                <Toggle aria-label={t('settings.common.enabled', 'Enabled')}
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
                    <Toggle aria-label={t('settings.common.enabled', 'Enabled')}
                      $on={modelStoreDraft.staging.cleanup_on_deactivate}
                      onClick={() => updateStaging({ cleanup_on_deactivate: !modelStoreDraft.staging.cleanup_on_deactivate })}
                    />
                  </Row>
                </>
              )}
            </CollapsibleSection>

          </>

          {/* Inference — always shown, not gated on model_store config */}
          <CollapsibleSection {...section('inference')} title={t('settings.inference.legend', 'Inference')} summary={({ default: 'Default', optiq: 'OptiQ', turboquant_adaptive: 'TurboQuant Adaptive', turboquant: 'TurboQuant', mlx_quantized: 'MLX Quantized' } as Record<string, string>)[kvBackend] ?? kvBackend}>
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
          </CollapsibleSection>

          {/* HuggingFace */}
          <CollapsibleSection {...section('huggingFace')} title={t('settings.huggingFace.legend', 'HuggingFace')} summary={hfToken || effective?.has_hf_token ? t('settings.tokenSet', 'Token set') : t('settings.tokenNotSet', 'Not set')}>
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
          </CollapsibleSection>

          {/* Logging */}
          <CollapsibleSection {...section('logging')} title={t('settings.logging.legend', 'Logging')} summary={loggingDraft.enabled ? t('common.on', 'On') : t('common.off', 'Off')}>
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
              <Toggle aria-label={t('settings.common.enabled', 'Enabled')} $on={loggingDraft.enabled} onClick={() => setLoggingDraft(prev => ({ ...prev, enabled: !prev.enabled }))} />
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
          </CollapsibleSection>

          {/* Intelligent fabric: Skulk's resident operator cognition. The toggle is the
              whole surface; model preference stays config-file-only until
              the cards-DB arc gives model pickers a proper home. */}
          <CollapsibleSection {...section('intelligentFabric')} title={t('settings.intelligentFabric.legend', 'Intelligent Fabric')} summary={fabricDraft.enabled ? t('common.on', 'On') : t('common.off', 'Off')}>
            <Row>
              <FieldLabel>
                {t('settings.common.enabled', 'Enabled')}
                <InfoTooltip
                  filled
                  content={t(
                    'settings.intelligentFabric.enabledTooltip',
                    'Keeps Skulk available as the fabric itself: ask about cluster health, models, and diagnostics from the Skulk page. Skulk can prepare basic actions, but each one requires your separate approval.',
                  )}
                />
              </FieldLabel>
              <Toggle aria-label={t('settings.common.enabled', 'Enabled')} $on={fabricDraft.enabled} onClick={() => setFabricDraft(prev => ({ ...prev, enabled: !prev.enabled }))} />
            </Row>
            {fabricDraft.enabled && (
              <HintText>
                {t(
                  'settings.intelligentFabric.syncHint',
                  'Skulk prepares its resident intelligence automatically after saving; the first start downloads its model. Turning it off removes that system placement.',
                )}
              </HintText>
            )}
          </CollapsibleSection>

          {/* Field telemetry: consent lives in skulk.yaml (survives restarts,
              synced like other settings). Both switches stay permanently
              available here; the first-run modal only acquires the initial
              choice. */}
          {telemetryDraft && (
            <CollapsibleSection {...section('telemetry')} title={t('settings.telemetry.legend', 'Telemetry')} summary={telemetryDraft?.consent}>
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
                <Toggle aria-label={t('settings.common.enabled', 'Enabled')}
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
                <Toggle aria-label={t('settings.common.enabled', 'Enabled')}
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
            </CollapsibleSection>
          )}

          </Sections>
          {configPath && <ConfigPath style={{ marginTop: 16, overflowWrap: 'anywhere' }}>{t('settings.configPath', 'Config: {configPath}', { configPath })}</ConfigPath>}
        </Body>

        <Footer style={{ display: devicesOpen ? 'none' : undefined }}>
          <Spacer />
          <Button variant="outline" size="md" onClick={onClose}>
            {t('common.cancel', 'Cancel')}
          </Button>
          <Button variant="solid" size="md" loading={saving} onClick={handleSave} disabled={loading || !fullConfig}>
            {t('settings.saveChanges', 'Save changes')}
          </Button>
        </Footer>
    </RightDrawer>
  );
}
