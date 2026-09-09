import { useState } from 'react';
import { useSkulkTranslation } from '../../i18n/tolgee';
import {
  useLazyGetRuntimeReleaseQuery, useGetRuntimeInstallationQuery,
  useInstallRuntimeReleaseMutation, useActivateRuntimeReleaseMutation,
  type ManagedRuntime,
} from '../../store/endpoints/plugins';
import { Button } from '../common/Button';

/** Review, stage and explicitly activate one trusted source without replaying a lost request. */
export function RuntimeReleasePanel({ runtime }: { runtime: ManagedRuntime }) {
  const { t } = useSkulkTranslation();
  const [inspect, inspection] = useLazyGetRuntimeReleaseQuery();
  const installation = useGetRuntimeInstallationQuery(runtime.plugin_id, { pollingInterval: 3000, skipPollingIfUnfocused: true });
  const [install, installing] = useInstallRuntimeReleaseMutation();
  const [activate, activating] = useActivateRuntimeReleaseMutation();
  const [submitted, setSubmitted] = useState<string | null>(null);
  const [activationSubmitted, setActivationSubmitted] = useState(false);
  const [acceptedDigest, setAcceptedDigest] = useState<string | null>(null);
  const [rollback, setRollback] = useState(false);
  const [notice, setNotice] = useState('');
  const operation = installation.currentData?.operation;
  const review = inspection.currentData ?? operation?.review;
  const permissionsAccepted = !!review && acceptedDigest === review.runtime_digest;
  const busy = installing.isLoading || activating.isLoading || inspection.isFetching;
  const pending = operation && ['accepted', 'downloading', 'staging'].includes(operation.state);
  const uncertain = submitted && operation?.request.operation_id !== submitted;
  const staged = operation?.state === 'staged' && operation.review.runtime_digest === review?.runtime_digest;
  const lifecyclePending = runtime.operation_state === 'accepted' || runtime.operation_state === 'applying' || runtime.operation_state === 'recovery_required';
  const alreadyActive = runtime.enabled && runtime.selected_digest === review?.runtime_digest;
  const stateLabel = operation ? {
    accepted: t('plugins.installAccepted', 'Accepted'),
    downloading: t('plugins.installDownloading', 'Downloading verified artifacts'),
    staging: t('plugins.installStaging', 'Preparing isolated runtime'),
    staged: t('plugins.installStaged', 'Ready for activation'),
    recovery_required: t('plugins.installRecovery', 'Installation requires local recovery'),
  }[operation.state] : '';
  const inspectRelease = async () => {
    setAcceptedDigest(null);
    setNotice('');
    try { await inspect(runtime.plugin_id).unwrap(); }
    catch { setNotice(t('plugins.releaseUnavailable', 'Release inspection failed. Check the owner-configured source, credential and publisher trust.')); }
  };
  const installRelease = async () => {
    if (!review) return;
    const id = crypto.randomUUID().replaceAll('-', '');
    setSubmitted(id);
    setNotice('');
    try {
      await install({ pluginId: runtime.plugin_id, request: { operation_id: id, runtime_digest: review.runtime_digest, expected_source_revision: review.source_revision } }).unwrap();
    } catch {
      setNotice(t('plugins.installUncertain', 'The installation response was not confirmed. Read installation status before another action.'));
    }
  };
  const activateRelease = async () => {
    if (!review || !permissionsAccepted) return;
    setActivationSubmitted(true);
    setNotice('');
    try {
      await activate({ pluginId: runtime.plugin_id, operationId: crypto.randomUUID().replaceAll('-', ''), expectedRevision: runtime.selection_revision, runtimeDigest: review.runtime_digest, rollback }).unwrap();
    } catch {
      setNotice(t('plugins.activationUncertain', 'The activation response was not confirmed. Check the retained lifecycle operation above before another action.'));
    }
  };
  return <section aria-label={t('plugins.releaseInstallation', 'Release installation')}>
    <Button type="button" disabled={busy || !!pending} onClick={() => void inspectRelease()}>{t('plugins.inspectRelease', 'Inspect configured release')}</Button>
    {installation.error ? <p role="status">{t('plugins.installStatusUnavailable', 'Installation status is unavailable. No request has been repeated.')}</p> : null}
    {review ? <>
      <h4>{review.bundle_id} {review.version}</h4>
      <p>{review.publisher} · {review.platform} · Python {review.python_requires}</p>
      <p>{t('plugins.releaseSize', 'Download size')}: {(review.artifact_bytes / 1048576).toFixed(1)} MiB</p>
      <p>{t('plugins.releaseExpiry', 'Release expires')}: {new Date(review.expires_at * 1000).toLocaleString()}</p>
      <p>{t('plugins.releaseDigest', 'Release identity')}: <code>{review.runtime_digest}</code></p>
      <ul>{review.permissions.map((permission) => <li key={permission}>{permission}</li>)}</ul>
      <Button type="button" disabled={busy || !!pending || !!uncertain || !!installation.error || staged || operation?.state === 'recovery_required'} onClick={() => void installRelease()}>{t('plugins.installRelease', 'Download and install')}</Button>
      {staged ? <>
        <p><label><input type="checkbox" checked={permissionsAccepted} onChange={(event) => setAcceptedDigest(event.target.checked ? review.runtime_digest : null)} /> {t('plugins.acceptReleasePermissions', 'I accept the permissions listed for this release.')}</label></p>
        <p><label><input type="checkbox" checked={rollback} onChange={(event) => setRollback(event.target.checked)} /> {t('plugins.allowReleaseRollback', 'Allow rollback to this older retained release.')}</label></p>
        <Button type="button" disabled={busy || !permissionsAccepted || activationSubmitted || lifecyclePending || alreadyActive || !!installation.error} onClick={() => void activateRelease()}>{t('plugins.activateRelease', 'Activate release')}</Button>
      </> : null}
    </> : null}
    {operation ? <p role="status">{stateLabel} · {(operation.downloaded_bytes / 1048576).toFixed(1)} MiB</p> : null}
    <Button type="button" disabled={installation.isFetching || busy} onClick={() => void installation.refetch()}>{t('plugins.refreshInstallation', 'Refresh installation status')}</Button>
    {notice && !(operation?.request.operation_id === submitted && operation.state === 'staged') ? <p role="status">{notice}</p> : null}
    <p>{t('plugins.installSeparateApproval', 'Installation and activation do not approve paid capacity. Provider setup and owner spending approval remain separate.')}</p>
  </section>;
}
