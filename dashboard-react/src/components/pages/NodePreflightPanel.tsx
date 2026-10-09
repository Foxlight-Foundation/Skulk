import { useEffect, useState } from 'react';
import styled from 'styled-components';
import { FiAlertCircle, FiCheckCircle } from 'react-icons/fi';
import { useSkulkTranslation } from '../../i18n/tolgee';
import {
  pluginRefusalDetail, pluginRequestRefused, useConfigurePluginNodeMutation, useLazyGetNodeConfigurationQuery,
  useLazyGetNodePreflightQuery, type NodeAddress,
} from '../../store/endpoints/plugins';
import { Button } from '../common/Button';
import { humanizeKey } from './pluginSettingsSchema';
import { startNodeAgain } from './turnOnCapability';

/** Props for one node's setup checks. */
export interface NodePreflightPanelProps extends NodeAddress {
  /**
   * Run the checks again whenever this number changes to a new positive
   * value, as a caller does after the host refuses to turn the node on.
   */
  runRequest?: number;
  /**
   * The node is on but stopped waiting for setup. When its checks then pass,
   * offer to start it again: it stopped on a check that failed while it was
   * starting and does not retry by itself.
   */
  offerStartAgain?: boolean;
}

/**
 * Run explicit setup observations without treating a prior success as enable authority.
 *
 * The plugin names each check by a stable code; the common ones read in
 * plain words, and any other code is shown as its words rather than raw.
 */
export function NodePreflightPanel({ pluginId, nodeId, runRequest = 0, offerStartAgain = false }: NodePreflightPanelProps) {
  const { t } = useSkulkTranslation();
  const [run, query] = useLazyGetNodePreflightQuery();
  const [readConfiguration] = useLazyGetNodeConfigurationQuery();
  const [configure] = useConfigurePluginNodeMutation();
  const [starting, setStarting] = useState(false);
  const [startOutcome, setStartOutcome] = useState<{ text: string; problem: boolean } | null>(null);
  const startAgain = async () => {
    setStarting(true);
    setStartOutcome(null);
    const outcome = await startNodeAgain({ readConfiguration: (address) => readConfiguration(address, false), configure },
      { pluginId, nodeId }, pluginRefusalDetail, pluginRequestRefused);
    setStarting(false);
    if (outcome.kind === 'started') {
      setStartOutcome({ text: t('plugins.startAgainStarted', 'Starting it again. Its status updates here in a moment.'), problem: false });
    } else if (outcome.kind === 'off') {
      setStartOutcome({ text: t('plugins.startAgainTurnedOff', 'It is turned off now, so it was left off. Turn it on from its settings to start it.'), problem: false });
    } else {
      const { detail } = outcome.refusal;
      setStartOutcome({ text: detail ? t('plugins.startAgainRefusedReason', 'It did not start. The host said: {reason}', { reason: detail })
        : t('plugins.startAgainRefused', 'It did not start. Run the checks again to see what is missing.'), problem: true });
    }
  };
  const report = query.currentData;
  useEffect(() => {
    if (runRequest > 0) void run({ pluginId, nodeId });
    // Only a new request runs the checks; the address is fixed for one panel.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runRequest]);
  const checkLabel = (code: string): string => ({
    configuration: t('plugins.check.configuration', 'Settings'),
    runtime: t('plugins.check.runtime', 'Runtime'),
    durable_storage: t('plugins.check.durableStorage', 'Storage'),
    credentials: t('plugins.check.credentials', 'Credentials'),
    observation_current: t('plugins.check.observationCurrent', 'Status report'),
  } as Record<string, string>)[code] ?? humanizeKey(code);
  const passed = report?.checks.filter((check) => check.passed).length ?? 0;
  return <Panel aria-label={t('plugins.preflight', 'Setup checks')}>
    <div>
      <Button type="button" variant="outline" size="sm" disabled={query.isFetching} loading={query.isFetching} onClick={() => void run({ pluginId, nodeId })}>
        {query.isFetching ? t('plugins.checkingSetup', 'Checking setup…') : report ? t('plugins.checkAgain', 'Check again') : t('plugins.checkSetup', 'Check setup')}
      </Button>
    </div>
    {query.error ? <Problem role="alert">{t('plugins.checksUnavailable', 'Setup checks are unavailable. Check plugin access and service health, then retry.')}</Problem> : null}
    {report && !query.error ? <>
      <Summary>
        {passed === report.checks.length
          ? t('plugins.checksAllPassed', 'All {count} checks passed.', { count: report.checks.length })
          : t('plugins.checksSomeFailed', '{failed} of {count} checks need attention.', { failed: report.checks.length - passed, count: report.checks.length })}
        {' '}<Meta>{t('plugins.checkedAt', 'Checked {time}, against settings revision {revision}.', { time: new Date(report.observedAt * 1000).toLocaleString(), revision: report.revision })}</Meta>
      </Summary>
      <Checklist>{report.checks.map((check) => <Check key={check.code} $passed={check.passed}>
        <Mark aria-hidden="true" $passed={check.passed}>{check.passed ? <FiCheckCircle /> : <FiAlertCircle />}</Mark>
        <div>
          <CheckTitle><strong>{checkLabel(check.code)}</strong><span>{check.passed ? t('plugins.checkPassed', 'Passed') : t('plugins.checkFailed', 'Needs attention')}</span></CheckTitle>
          {check.correctiveAction ? <p>{check.correctiveAction}</p> : null}
        </div>
      </Check>)}</Checklist>
      <Meta as="p">{t('plugins.checksAreObservations', 'These are observations. Turning it on runs fresh checks and does not approve spending.')}</Meta>
      {offerStartAgain && passed === report.checks.length ? <StartAgain>
        <p>{t('plugins.startAgainDetail', 'Its checks pass now, but it stopped when a check failed while it was starting, for example while Skulk was updating, and it does not retry by itself. Starting it again runs the checks once more and starts it.')}</p>
        <div><Button type="button" variant="solid" size="sm" disabled={starting} loading={starting} onClick={() => void startAgain()}>{t('plugins.startAgain', 'Start it again')}</Button></div>
      </StartAgain> : null}
    </> : null}
    {startOutcome ? <Outcome role={startOutcome.problem ? 'alert' : 'status'} $problem={startOutcome.problem}>{startOutcome.text}</Outcome> : null}
  </Panel>;
}

const Panel = styled.section`display: flex; flex-direction: column; gap: 12px; min-width: 0; overflow-wrap: anywhere;`;
const StartAgain = styled.div`
  display: flex; flex-direction: column; gap: 8px;
  p { margin: 0; font-size: 13px; line-height: 1.5; color: ${({ theme }) => theme.colors.textSecondary}; }
`;
const Outcome = styled.p<{ $problem: boolean }>`
  margin: 0; font-size: 13px; line-height: 1.5;
  color: ${({ theme, $problem }) => $problem ? theme.colors.error : theme.colors.text};
`;
const Problem = styled.p`margin: 0; font-size: 13px; line-height: 1.5; color: ${({ theme }) => theme.colors.error};`;
const Summary = styled.p`margin: 0; font-size: 13.5px; line-height: 1.5; color: ${({ theme }) => theme.colors.text};`;
const Meta = styled.span`font-size: 12px; line-height: 1.5; margin: 0; color: ${({ theme }) => theme.colors.metadataText};`;
const Checklist = styled.ul`
  list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column;
  border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: ${({ theme }) => theme.radii.lg}; overflow: hidden;
`;
const Check = styled.li<{ $passed: boolean }>`
  display: grid; grid-template-columns: 20px minmax(0, 1fr); gap: 12px; align-items: start; padding: 12px 14px;
  background: ${({ theme, $passed }) => $passed ? 'transparent' : theme.colors.liveBg};
  & + & { border-top: 1px solid ${({ theme }) => theme.colors.border}; }
  p { margin: 4px 0 0; font-size: 13px; line-height: 1.5; color: ${({ theme }) => theme.colors.textSecondary}; }
`;
const Mark = styled.span<{ $passed: boolean }>`
  display: inline-flex; align-items: center; justify-content: center; height: 20px; font-size: 17px;
  color: ${({ theme, $passed }) => $passed ? theme.colors.healthy : theme.colors.liveText};
`;
const CheckTitle = styled.div`
  display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  strong { font-size: 14px; font-weight: 600; color: ${({ theme }) => theme.colors.text}; }
  span { font-size: 12.5px; color: ${({ theme }) => theme.colors.textSecondary}; }
`;
