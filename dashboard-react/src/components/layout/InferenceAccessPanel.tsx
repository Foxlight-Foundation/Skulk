import { useCallback, useEffect, useRef, useState } from 'react';
import styled from 'styled-components';
import { operatorSession } from '../../auth/operatorSession';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { copyToClipboard } from '../../utils/clipboard';
import {
  inferenceAccessRequest, inferenceGrantKey, inferenceListingGrant, selectInferenceGrant,
  type InferenceAccessPolicy, type InferenceClientCreated, type InferenceClients,
  type InferenceGrant, type InferenceListing, type InferencePrincipal,
} from '../../utils/inferenceAccess';
import { Button } from '../common/Button';
import { Field } from '../common/Field';
import { Surface } from '../common/Surfaces';

const Body = styled.div`padding: 20px; display: flex; flex-direction: column; gap: 16px; overflow-y: auto; min-width: 0;`;
const Group = styled(Surface)`display: flex; flex-direction: column; gap: 12px; h3, p { margin: 0; } p, small { color: ${({ theme }) => theme.colors.textSecondary}; line-height: 1.5; }`;
const Choice = styled.label`display: flex; align-items: flex-start; gap: 10px; cursor: pointer; input { margin-top: 4px; accent-color: ${({ theme }) => theme.colors.gold}; } span { min-width: 0; overflow-wrap: anywhere; } small { display: block; margin-top: 4px; }`;
const Row = styled.div`display: flex; gap: 8px; align-items: center; flex-wrap: wrap;`;
const Select = styled.select`width: 100%; padding: 9px; color: ${({ theme }) => theme.colors.text}; background: ${({ theme }) => theme.colors.surface}; border: 1px solid ${({ theme }) => theme.colors.borderControl}; border-radius: 8px; font: inherit;`;
const ErrorText = styled.p`color: ${({ theme }) => theme.colors.error}; margin: 0;`;

interface Snapshot {
  owner: InferencePrincipal;
  clients: InferenceClients;
  policy: InferenceAccessPolicy;
  listings: InferenceListing[];
}

/** Immediate, explicit caller enrollment; independent of the general Settings draft. */
export function InferenceAccessPanel() {
  const { t } = useSkulkTranslation();
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [policy, setPolicy] = useState<InferenceAccessPolicy | null>(null);
  const [selected, setSelected] = useState('direct-owner');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [label, setLabel] = useState('');
  const [issued, setIssued] = useState<InferenceClientCreated | null>(null);
  const mounted = useRef(false);

  const load = useCallback(async (signal?: AbortSignal) => {
    setBusy(true); setError(''); setNotice('');
    try {
      const [owner, clients, current, listings] = await Promise.all([
        inferenceAccessRequest<InferencePrincipal>('/v1/inference-access/owner', 'GET', undefined, signal),
        inferenceAccessRequest<InferenceClients>('/v1/inference-access/clients', 'GET', undefined, signal),
        inferenceAccessRequest<InferenceAccessPolicy>('/v1/inference-access/policy', 'GET', undefined, signal),
        inferenceAccessRequest<InferenceListing[]>('/v1/native-inference', 'GET', undefined, signal),
      ]);
      if (mounted.current && !signal?.aborted) {
        setSnapshot({ owner, clients, policy: current, listings }); setPolicy(current);
      }
    } catch (failure) {
      if (mounted.current && !signal?.aborted) setError(failure instanceof Error ? failure.message : t('inferenceAccess.loadFailed', 'Could not load inference access.'));
    } finally { if (mounted.current && !signal?.aborted) setBusy(false); }
  }, [t]);

  useEffect(() => {
    mounted.current = true; const controller = new AbortController(); void load(controller.signal);
    return () => { mounted.current = false; controller.abort(); };
  }, [load]);

  const principal: InferencePrincipal | undefined = selected === 'direct-owner' ? snapshot?.owner
    : snapshot?.owner && { identifier: selected, kind: 'client', issuer_node_id: snapshot.owner.issuer_node_id };
  const samePrincipal = (candidate: InferencePrincipal) => candidate.identifier === principal?.identifier && candidate.kind === principal?.kind && candidate.issuer_node_id === principal?.issuer_node_id;
  const access = policy?.callers.find(entry => samePrincipal(entry.principal));
  const obsolete = policy?.callers.filter(entry => entry.principal.issuer_node_id !== snapshot?.owner.issuer_node_id
    || (entry.principal.kind === 'client' && !snapshot?.clients.clients.some(client => `client:${client.identifier}` === entry.principal.identifier))) ?? [];
  const choices = snapshot?.listings.map(listing => ({ grant: inferenceListingGrant(listing), title: listing.descriptor.title, available: true })) ?? [];
  const currentKeys = new Set(choices.map(choice => inferenceGrantKey(choice.grant)));
  for (const grant of access?.grants ?? []) {
    if (!currentKeys.has(inferenceGrantKey(grant))) choices.push({ grant, title: `${grant.capability_id} ${grant.version}`, available: false });
  }

  const choose = (grant: InferenceGrant, checked: boolean) => {
    if (!policy || !principal) return;
    setNotice('');
    const updated = { principal, grants: selectInferenceGrant(access?.grants ?? [], grant, checked), maximum_seconds: access?.maximum_seconds ?? null };
    setPolicy({ ...policy, callers: [...policy.callers.filter(entry => !samePrincipal(entry.principal)), updated] });
  };

  const save = async () => {
    if (!snapshot || !policy) return;
    setBusy(true); setError(''); setNotice('');
    try {
      const current = await inferenceAccessRequest<InferenceAccessPolicy>('/v1/inference-access/policy', 'PUT', { expected_revision: snapshot.policy.revision, enabled: policy.enabled, callers: policy.callers });
      if (mounted.current) { setSnapshot({ ...snapshot, policy: current }); setPolicy(current); setNotice(t('inferenceAccess.saved', 'Inference access saved.')); }
    } catch (failure) { if (mounted.current) setError(failure instanceof Error ? failure.message : t('inferenceAccess.saveFailed', 'Could not save access. Reload before trying again.')); }
    finally { if (mounted.current) setBusy(false); }
  };

  const create = async () => {
    if (!snapshot || !label.trim()) return;
    setBusy(true); setError(''); setNotice('');
    try {
      const result = await inferenceAccessRequest<InferenceClientCreated>('/v1/inference-access/clients', 'POST', { expected_revision: snapshot.clients.revision, label: label.trim() });
      if (mounted.current) {
        setIssued(result); setLabel(''); setSelected(`client:${result.client.identifier}`);
        setSnapshot({ ...snapshot, clients: { revision: result.revision, clients: [...snapshot.clients.clients, result.client] } });
      }
    } catch (failure) { if (mounted.current) setError(failure instanceof Error ? failure.message : t('inferenceAccess.createFailed', 'Client creation outcome is unavailable. Reload the inventory before trying again.')); }
    finally { if (mounted.current) setBusy(false); }
  };

  const revoke = async () => {
    if (!snapshot || !selected.startsWith('client:')) return;
    setBusy(true); setError(''); setNotice('');
    try {
      const clients = await inferenceAccessRequest<InferenceClients>(`/v1/inference-access/clients/${encodeURIComponent(selected.slice(7))}`, 'DELETE', { expected_revision: snapshot.clients.revision });
      if (mounted.current) { setSnapshot({ ...snapshot, clients }); setIssued(null); setSelected('direct-owner'); setNotice(t('inferenceAccess.revoked', 'Client key revoked. Its requests can no longer authenticate.')); }
    } catch (failure) { if (mounted.current) setError(failure instanceof Error ? failure.message : t('inferenceAccess.revokeFailed', 'Could not confirm revocation. Reload before trying again.')); }
    finally { if (mounted.current) setBusy(false); }
  };

  return <Body>
    <p>{t('inferenceAccess.intro', 'Choose who may use installed inference capabilities from this node. Connecting a provider account does not authorize a caller. Changes here take effect immediately when saved.')}</p>
    {error && <ErrorText role="alert">{error}</ErrorText>}
    {notice && <p role="status">{notice}</p>}
    <Row><Button variant="outline" size="sm" disabled={busy} onClick={() => void load()}>{t('inferenceAccess.reload', 'Reload access')}</Button>
      {operatorSession.snapshot().mode !== 'direct' && <Button variant="outline" size="sm" disabled={busy} onClick={() => { operatorSession.useDirectAccess(); setSnapshot(null); setPolicy(null); setIssued(null); void load(); }}>{t('inferenceAccess.direct', 'Use direct host access')}</Button>}</Row>
    {busy && !snapshot && <p role="status">{t('inferenceAccess.loading', 'Loading inference access…')}</p>}
    {snapshot && policy && <>
      <Group><h3>{t('inferenceAccess.permissionTitle', 'Caller permissions')}</h3>
        <Choice><input type="checkbox" checked={policy.enabled} disabled={busy} onChange={event => { setPolicy({ ...policy, enabled: event.target.checked }); setNotice(''); }} /><span>{t('inferenceAccess.enabled', 'Enable approved inference access')}<small>{t('inferenceAccess.enableHint', 'Only the callers and exact capabilities you approve below are permitted. Turning this off pauses all enrolled callers.')}</small></span></Choice>
        <label>{t('inferenceAccess.caller', 'Caller')}<Select aria-label={t('inferenceAccess.caller', 'Caller')} disabled={busy} value={selected} onChange={event => setSelected(event.target.value)}>
          <option value="direct-owner">{t('inferenceAccess.dashboard', 'This dashboard (direct owner)')}</option>
          {snapshot.clients.clients.map(client => <option key={client.identifier} value={`client:${client.identifier}`}>{client.label}</option>)}
        </Select></label>
        <p>{t('inferenceAccess.reviewHint', 'Select the capabilities this caller may use. External processing may send request content to a provider. A capability update requires approval of its new release.')}</p>
        {choices.length === 0 && <p>{t('inferenceAccess.noChoices', 'No native inference capabilities are available. Start a compatible installed capability in Plugins, then reload.')}</p>}
        {choices.map(({ grant, title, available }) => <Choice key={inferenceGrantKey(grant)}><input type="checkbox" disabled={busy} checked={access?.grants.some(existing => inferenceGrantKey(existing) === inferenceGrantKey(grant)) ?? false} onChange={event => choose(grant, event.target.checked)} /><span>{title}<small>{grant.behavior === 'inference' ? t('inferenceAccess.inferenceEffect', 'Inference') : t('inferenceAccess.readEffect', 'Read')} · {grant.processing_domain === 'external' ? t('inferenceAccess.external', 'External processing') : grant.processing_domain === 'private_site' ? t('inferenceAccess.privateSite', 'Private site') : t('inferenceAccess.local', 'Local processing')}{!available && ` · ${t('inferenceAccess.unavailable', 'Not currently available; review or remove this grant')}`}</small></span></Choice>)}
        {access?.maximum_seconds != null && <p>{t('inferenceAccess.ceiling', 'Existing request duration ceiling (seconds):')} {access.maximum_seconds}</p>}
        <Row><Button disabled={busy || JSON.stringify(policy) === JSON.stringify(snapshot.policy)} onClick={() => void save()}>{t('inferenceAccess.save', 'Save inference access')}</Button>
          {selected.startsWith('client:') && <Button variant="outline" disabled={busy} onClick={() => void revoke()}>{t('inferenceAccess.revoke', 'Revoke client key')}</Button>}</Row>
        {obsolete.map(entry => <Row key={JSON.stringify(entry.principal)}><span>{t('inferenceAccess.obsolete', 'Retained permissions for an unavailable caller:')} {entry.principal.identifier}</span><Button variant="outline" size="sm" disabled={busy} onClick={() => setPolicy({ ...policy, callers: policy.callers.filter(candidate => candidate !== entry) })}>{t('inferenceAccess.remove', 'Remove retained permissions')}</Button></Row>)}
      </Group>
      <Group><h3>{t('inferenceAccess.clientTitle', 'API clients')}</h3><p>{t('inferenceAccess.clientHint', 'Create a separate key for each application. A new key has no capability permissions; select that caller and save its permissions above.')}</p>
        <label>{t('inferenceAccess.clientLabel', 'Application name')}<Field aria-label={t('inferenceAccess.clientLabel', 'Application name')} value={label} maxLength={128} disabled={busy} onChange={event => setLabel(event.target.value)} /></label>
        <Button variant="outline" disabled={busy || !label.trim() || issued !== null} onClick={() => void create()}>{t('inferenceAccess.create', 'Create client key')}</Button>
      </Group>
    </>}
    {issued && <Group><h3>{t('inferenceAccess.keyTitle', 'Save this key now')}</h3><p>{t('inferenceAccess.keyHint', 'This key is shown once. Store it securely in your API client. It is cleared when dismissed or when you leave this panel.')}</p>
      <Field aria-label={t('inferenceAccess.key', 'New client key')} readOnly type="password" value={issued.token} autoComplete="off" />
      <Row><Button variant="outline" onClick={() => { void copyToClipboard(issued.token).then(() => setNotice(t('inferenceAccess.copied', 'Client key copied.')), () => setError(t('inferenceAccess.copyFailed', 'Copy failed. Save the key from the field before dismissing it.'))); }}>{t('inferenceAccess.copy', 'Copy client key')}</Button><Button onClick={() => setIssued(null)}>{t('inferenceAccess.dismiss', 'I saved the key')}</Button></Row>
    </Group>}
  </Body>;
}
