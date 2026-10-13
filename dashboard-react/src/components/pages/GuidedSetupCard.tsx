import { useEffect, useRef, useState } from 'react';
import styled from 'styled-components';
import { FiCheckCircle } from 'react-icons/fi';
import { operatorSession } from '../../auth/operatorSession';
import { useSkulkTranslation } from '../../i18n/tolgee';
import {
  pluginRefusalDetail,
  refusalSentence,
  useGetNodeCredentialsQuery,
  useGetNodeSetupActionsQuery,
  useResumeNodeSetupMutation,
  useStartNodeSetupMutation,
  type NodeAddress,
  type SetupOperation,
} from '../../store/endpoints/plugins';
import { randomHex32 } from '../../utils/randomIds';
import { Button } from '../common/Button';
import { Spinner } from '../common/Spinner';
import { PluginConfigurationFields } from './PluginConfigurationFields';
import { guidedAction } from './guidedSetup';

/** Props for one node's guided setup. */
export interface GuidedSetupCardProps {
  /** The plugin node whose guided setup action this runs. */
  address: NodeAddress;
  /** Called once the setup finished, when the plugin asked to be turned on then. */
  onTurnOn: () => void;
}

type Phase = 'form' | 'working' | 'done' | 'failed';

/** True when a required form value is still missing, so Set up stays disabled. */
function missingRequired(schema: Record<string, unknown>, values: Record<string, unknown>): boolean {
  const required = Array.isArray(schema.required) ? schema.required : [];
  return required.some((name) => typeof name === 'string' && (values[name] === undefined || values[name] === null || values[name] === ''));
}

/**
 * The few steps a newcomer needs, instead of the plugin's whole settings.
 *
 * A plugin marks one setup action as guided and names the credentials it
 * needs first. This card asks for those credentials, shows the action's
 * ordinary fields with its Advanced ones collapsed, and runs the action in
 * one step. When the plugin asked for it, the plugin is turned on afterwards.
 * Credential values go straight to the host and never enter app state.
 */
export function GuidedSetupCard({ address, onTurnOn }: GuidedSetupCardProps) {
  const { t } = useSkulkTranslation();
  const [phase, setPhase] = useState<Phase>('form');
  const [operationId, setOperationId] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [values, setValues] = useState<Record<string, unknown>>({});
  const keys = useRef<Record<string, HTMLInputElement | null>>({});
  const actions = useGetNodeSetupActionsQuery(address, { pollingInterval: phase === 'working' ? 2000 : 0 });
  const credentials = useGetNodeCredentialsQuery(address);
  const [startSetup] = useStartNodeSetupMutation();
  const [resumeSetup] = useResumeNodeSetupMutation();
  // The action stays bound once started: after it completes the plugin no
  // longer offers it, and this card shows how it ended instead.
  const [bound, setBound] = useState(() => guidedAction(actions.currentData));
  const offered = guidedAction(actions.currentData);
  if (offered && (!bound || bound.action.schemaDigest !== offered.action.schemaDigest) && phase === 'form') setBound(offered);
  const guided = phase === 'form' ? offered : bound;

  const operation: SetupOperation | undefined = operationId
    ? actions.currentData?.operations.find((item) => item.operationId === operationId)
    : undefined;
  // Follow the operation to its end; render-time adjustment keeps the phase
  // in step with the host without an effect.
  if (phase === 'working' && operation?.phase === 'complete') {
    setPhase('done');
  } else if (phase === 'working' && operation?.phase === 'failed') {
    setPhase('failed');
    setProblem(operation.correctiveAction ?? t('plugins.guided.failed', 'Setup did not finish. Try again.'));
  }

  // Turning on is the parent's, so it starts after this render commits, and
  // exactly once however often the parent renders afterwards.
  const turnOn = phase === 'done' && bound?.hint.turnOn === true;
  const turnedOn = useRef(false);
  useEffect(() => {
    if (!turnOn || turnedOn.current) return;
    turnedOn.current = true;
    onTurnOn();
  }, [turnOn, onTurnOn]);

  if (!guided) return null;
  const { action, hint } = guided;
  const needed = hint.credentials
    .map((id) => credentials.currentData?.credentials.find((item) => item.credentialId === id))
    .filter((item) => item !== undefined);

  const saveCredentials = async (): Promise<boolean> => {
    for (const credential of needed) {
      const input = keys.current[credential.credentialId];
      let value = input?.value.trim() ?? '';
      if (!value) {
        if (credential.ready) continue;
        setProblem(t('plugins.guided.keyRequired', 'Add your {title} first.', { title: credential.title }));
        return false;
      }
      if (input) input.value = '';
      const current = await credentials.refetch().unwrap();
      // The secret-bearing body bypasses Redux actions and query caches.
      const body = JSON.stringify({
        operation: 'replace', operationId: randomHex32(), credentialId: credential.credentialId,
        expectedRevision: current.revision, expectedSchemaDigest: current.schemaDigest, value,
      });
      value = '';
      const response = await operatorSession.fetch(
        `/v1/plugins/${encodeURIComponent(address.pluginId)}/nodes/${encodeURIComponent(address.nodeId)}/credentials`,
        {
          method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Skulk-Dashboard': 'pairing-v1' },
          credentials: 'same-origin', cache: 'no-store', redirect: 'error', body, signal: AbortSignal.timeout(35000),
        },
      );
      if (!response.ok) {
        setProblem(t('plugins.guided.keyNotSaved', 'Your {title} was not saved. Check it and try again.', { title: credential.title }));
        return false;
      }
    }
    return true;
  };

  const start = async () => {
    setProblem(null);
    setPhase('working');
    try {
      if (!(await saveCredentials())) { setPhase('form'); return; }
      await credentials.refetch();
      // Saving a credential moves the fences the action is bound to.
      const fresh = await actions.refetch().unwrap();
      const current = guidedAction(fresh);
      if (!current) {
        setPhase('form');
        setProblem(t('plugins.guided.notOffered', 'This plugin is already set up.'));
        return;
      }
      setBound(current);
      const id = randomHex32();
      setOperationId(id);
      await startSetup({
        ...address,
        mutation: {
          operationId: id,
          actionId: current.action.actionId,
          expectedRevision: fresh.revision,
          expectedSchemaDigest: fresh.schemaDigest,
          expectedCredentialRevision: fresh.credentialRevision,
          expectedCredentialSchemaDigest: fresh.credentialSchemaDigest,
          expectedActionSchemaDigest: current.action.schemaDigest,
          expectedRequiresApproval: current.action.requiresApproval ?? false,
          values,
        },
      }).unwrap();
    } catch (error: unknown) {
      setPhase('form');
      setOperationId(null);
      setProblem(refusalSentence(pluginRefusalDetail(error)) ?? t('plugins.guided.notStarted', 'Setup was not started. Try again.'));
    }
  };

  const retry = async () => {
    if (!operationId) return;
    setProblem(null);
    setPhase('working');
    try {
      await resumeSetup({ ...address, operationId }).unwrap();
    } catch (error: unknown) {
      setPhase('failed');
      setProblem(refusalSentence(pluginRefusalDetail(error)) ?? t('plugins.guided.notStarted', 'Setup was not started. Try again.'));
    }
  };

  const working = phase === 'working';
  return <Card aria-labelledby="guided-setup-title">
    <h3 id="guided-setup-title">{action.title}</h3>
    <p>{action.description}</p>
    {phase === 'done' ? <Done role="status">
      <FiCheckCircle aria-hidden="true" />
      <span>
        {hint.turnOn ? t('plugins.guided.doneTurningOn', 'Set up. Turning it on now.') : t('plugins.guided.done', 'Set up.')}
        {operation?.correctiveAction ? ` ${operation.correctiveAction}` : ''}
      </span>
    </Done> : <>
      {needed.map((credential) => <Field key={credential.credentialId}>
        <span>{credential.title}</span>
        {credential.ready ? <small>{t('plugins.guided.keySaved', 'Saved. Enter a new one only to replace it.')}</small> : <small>{credential.description}</small>}
        <input
          ref={(element) => { keys.current[credential.credentialId] = element; }}
          type="password" autoComplete="new-password" maxLength={4096} disabled={working}
          aria-label={credential.title}
          placeholder={credential.ready ? t('plugins.guided.keyReplace', 'Leave empty to keep the saved one') : t('plugins.guided.keyPaste', 'Paste it here')}
        />
      </Field>)}
      <PluginConfigurationFields schema={action.parametersSchema} values={values} onChange={setValues} disabled={working} />
      <ActionRow>
        <Button variant="solid" loading={working} disabled={working || missingRequired(action.parametersSchema, values)} onClick={() => void (phase === 'failed' ? retry() : start())}>
          {working ? t('plugins.guided.working', 'Setting up…') : phase === 'failed' ? t('plugins.guided.retry', 'Try again') : t('plugins.guided.start', 'Set up')}
        </Button>
        {working ? <Spinner size={16} /> : null}
      </ActionRow>
      {working ? <Muted role="status">{t('plugins.guided.workingHelp', 'This takes about a minute. It starts nothing that costs money.')}</Muted> : null}
    </>}
    {problem ? <Problem role="alert">{problem}</Problem> : null}
  </Card>;
}

const Card = styled.section`
  display: flex; flex-direction: column; gap: 12px; padding: 18px; border-radius: ${({ theme }) => theme.radii.lg};
  border: 1px solid ${({ theme }) => theme.colors.border}; background: ${({ theme }) => theme.colors.surface};
  h3 { margin: 0; font-size: 16px; color: ${({ theme }) => theme.colors.text}; }
  > p { margin: 0; font-size: 13.5px; line-height: 1.5; color: ${({ theme }) => theme.colors.textSecondary}; }
`;
const Field = styled.label`
  display: grid; gap: 6px; font-size: 14px; color: ${({ theme }) => theme.colors.text};
  small { color: ${({ theme }) => theme.colors.metadataText}; font-size: 12.5px; line-height: 1.45; }
  input { width: 100%; min-width: 0; box-sizing: border-box; padding: 9px; border-radius: ${({ theme }) => theme.radii.sm};
    border: 1px solid ${({ theme }) => theme.colors.border}; background: ${({ theme }) => theme.colors.surface}; color: ${({ theme }) => theme.colors.text}; }
`;
const ActionRow = styled.div`display: flex; align-items: center; gap: 10px;`;
const Muted = styled.p`margin: 0; font-size: 13px; color: ${({ theme }) => theme.colors.metadataText};`;
const Done = styled.p`
  margin: 0; display: flex; gap: 8px; align-items: flex-start; font-size: 14px; color: ${({ theme }) => theme.colors.text};
  svg { flex: none; margin-top: 2px; color: ${({ theme }) => theme.colors.healthy}; }
`;
const Problem = styled.p`
  margin: 0; padding: 10px 12px; border-radius: ${({ theme }) => theme.radii.md}; font-size: 13.5px; line-height: 1.5;
  border: 1px solid ${({ theme }) => theme.colors.borderDanger}; background: ${({ theme }) => theme.colors.errorBg}; color: ${({ theme }) => theme.colors.text};
`;
