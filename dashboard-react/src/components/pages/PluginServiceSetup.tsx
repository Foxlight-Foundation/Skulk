import { useEffect, useRef, type ReactNode } from 'react';
import styled from 'styled-components';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { apiSlice } from '../../store/api';
import { useAppDispatch } from '../../store/hooks';
import { useGetPluginServiceQuery, useStartPluginServiceSetupMutation } from '../../store/endpoints/plugins';
import { Button } from '../common/Button';
import { Spinner } from '../common/Spinner';

/** Props for {@link PluginServiceSetup}. */
export interface PluginServiceSetupProps {
  /** Whether this browser has direct owner access to the host (localhost or Tailscale). */
  direct: boolean;
  /** Page heading shown above the setup notice while the manager is not ready. */
  header?: ReactNode;
  /** What to show once the host's plugin manager is ready. */
  children: ReactNode;
}

/**
 * Gate the Plugins page on the host's plugin manager.
 *
 * The first time the page opens on a host with owner access, the host sets up
 * its plugin manager by itself (a per-user service, no terminal and no
 * administrator password) and the page shows progress until it is ready. A
 * failure shows the host's reason with a retry; a paired browser is told the
 * setup happens from the host itself.
 */
export function PluginServiceSetup({ direct, header, children }: PluginServiceSetupProps) {
  const { t } = useSkulkTranslation();
  const dispatch = useAppDispatch();
  const service = useGetPluginServiceQuery(undefined, { pollingInterval: 2000, skipPollingIfUnfocused: true });
  const [startSetup, starting] = useStartPluginServiceSetupMutation();
  const startedAutomatically = useRef(false);
  const state = service.data?.state;
  const wasReady = useRef(false);

  useEffect(() => {
    // Set up once, on the first visit, only where the owner is present.
    if (state === 'absent' && direct && !startedAutomatically.current) {
      startedAutomatically.current = true;
      void startSetup();
    }
  }, [state, direct, startSetup]);

  useEffect(() => {
    // The inventory answered "unavailable" until now; read it again once ready.
    if (state === 'ready' && !wasReady.current) {
      wasReady.current = true;
      dispatch(apiSlice.util.invalidateTags(['Plugins', 'PluginCatalog']));
    }
  }, [state, dispatch]);

  // An older host without the status route still shows its plugins as before.
  if (service.isError || state === 'ready') return <>{children}</>;
  if (service.isLoading || !service.data) return <>{header}<Notice role="status"><Spinner size={16} /> {t('plugins.service.checking', 'Checking this host’s plugin service…')}</Notice></>;

  const retry = <Button type="button" disabled={starting.isLoading} onClick={() => void startSetup()}>{t('plugins.service.retry', 'Try again')}</Button>;
  const status = service.data;
  if (status.state === 'setting_up' || (status.state === 'absent' && direct)) {
    return <>{header}<Notice role="status" data-testid="plugin-service-setting-up">
      <h2><Spinner size={16} /> {t('plugins.service.settingUp', 'Setting up plugins on this host')}</h2>
      <p>{t('plugins.service.settingUpDetail', 'This happens once and takes a few minutes. You can leave this page; setup continues on the host.')}</p>
      {status.progress ? <p>{status.progress}</p> : null}
    </Notice></>;
  }
  if (status.state === 'absent') {
    return <>{header}<Notice role="status">
      <h2>{t('plugins.service.needsHost', 'Plugins are not set up on this host yet')}</h2>
      <p>{t('plugins.service.needsHostDetail', 'Open Plugins on the host itself, or over Tailscale, to set them up. It takes a few minutes and happens once.')}</p>
    </Notice></>;
  }
  if (status.state === 'unsupported') {
    return <>{header}<Notice role="alert">
      <h2>{t('plugins.service.unsupported', 'This host cannot run plugins')}</h2>
      {status.error ? <p>{status.error}</p> : null}
    </Notice></>;
  }
  return <>{header}<Notice role="alert" data-testid="plugin-service-problem">
    <h2>{status.state === 'failed' ? t('plugins.service.failed', 'Plugin setup did not finish') : t('plugins.service.unavailable', 'The plugin service is not answering')}</h2>
    {status.error ? <p>{status.error}</p> : null}
    {direct ? retry : <p>{t('plugins.service.retryOnHost', 'Retry from the host itself, or over Tailscale.')}</p>}
  </Notice></>;
}

const Notice = styled.div`
  padding: 24px; margin: 20px 0; border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 14px; background: ${({ theme }) => theme.colors.surface};
  h2 { display: flex; align-items: center; gap: 10px; font-size: 16px; margin-bottom: 8px; color: ${({ theme }) => theme.colors.text}; }
  p { font-size: 14px; line-height: 1.6; color: ${({ theme }) => theme.colors.textSecondary}; margin-bottom: 12px; }
`;
