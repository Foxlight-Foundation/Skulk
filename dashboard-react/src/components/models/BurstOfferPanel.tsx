import { skipToken } from '@reduxjs/toolkit/query';
import { Fragment, useRef, useState } from 'react';
import styled from 'styled-components';
import { useSkulkTranslation } from '../../i18n/tolgee';
import {
  SETTLED_BURST_PHASES,
  useApproveBurstMutation,
  useGetBurstOperationQuery,
  useGetBurstProvidersQuery,
  useRecommendBurstMutation,
  useReleaseBurstMutation,
  type BurstOperationAddress,
  type BurstPhase,
  type BurstRecommendation,
} from '../../store/endpoints/burst';
import { pluginAccessRefused } from '../../store/endpoints/plugins';
import { randomHex32 } from '../../utils/randomIds';
import { Button } from '../common/Button';

/** The model and workload a remote machine would be sized for. */
export interface BurstOfferPanelProps {
  modelId: string;
  contextTokens: number;
}

const Panel = styled.section`
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid ${({ theme }) => theme.colors.border};
  overflow-wrap: anywhere;
  h4 { margin: 0 0 8px; font-size: inherit; }
  p { margin: 8px 0; }
`;
const Facts = styled.dl`
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 4px 12px;
  margin: 8px 0;
  dt { font-weight: 600; }
  dd { margin: 0; }
`;
const Actions = styled.div`display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px;`;

const storageKey = (modelId: string) => `skulk-burst-operation-v1:${encodeURIComponent(modelId)}`;

/** The rental this browser approved for the model, if one is still being followed. */
function retainedOperation(modelId: string): BurstOperationAddress | null {
  try {
    const raw = localStorage.getItem(storageKey(modelId));
    if (!raw) return null;
    const value = JSON.parse(raw) as Partial<BurstOperationAddress>;
    return typeof value.pluginId === 'string' && typeof value.nodeId === 'string'
      && typeof value.operationId === 'string' && /^[a-f0-9]{32}$/.test(value.operationId)
      ? { pluginId: value.pluginId, nodeId: value.nodeId, operationId: value.operationId } : null;
  } catch { return null; }
}

/** Show an exact six-decimal dollar string with cents, keeping smaller amounts. */
export function formatUsd(value: string): string {
  const [whole, fraction = ''] = value.split('.');
  const trimmed = fraction.replace(/0+$/, '');
  return `$${whole}.${trimmed.length >= 2 ? trimmed : trimmed.padEnd(2, '0')}`;
}

const when = (seconds: number) => new Date(seconds * 1000).toLocaleString();

/**
 * Offer, approve and follow remote capacity for a model the cluster cannot run.
 *
 * Shown only when the cluster's own assessment says the model needs external
 * capacity. Nothing is rented until the operator approves an offer; the
 * approval's operation ID is saved before it is sent, so a lost reply is
 * followed, never approved twice.
 */
export function BurstOfferPanel({ modelId, contextTokens }: BurstOfferPanelProps) {
  const { t } = useSkulkTranslation();
  const providers = useGetBurstProvidersQuery(undefined, { refetchOnMountOrArgChange: true });
  const [recommend, recommending] = useRecommendBurstMutation();
  const [approve, approving] = useApproveBurstMutation();
  const [release, releasing] = useReleaseBurstMutation();
  const [recommendation, setRecommendation] = useState<BurstRecommendation | null>(null);
  const [address, setAddress] = useState<BurstOperationAddress | null>(() => retainedOperation(modelId));
  const [notice, setNotice] = useState('');
  const submitting = useRef(false);
  const followed = useGetBurstOperationQuery(address ?? skipToken, { refetchOnMountOrArgChange: true });
  const operation = followed.currentData;
  const settled = operation ? SETTLED_BURST_PHASES.includes(operation.phase) : false;
  // Coming up, poll briskly; once ready, slowly, so an idle release or the
  // rental's expiry still shows here.
  useGetBurstOperationQuery(address && !settled ? address : skipToken, {
    pollingInterval: operation?.phase === 'ready' ? 15000 : 3000,
    skipPollingIfUnfocused: true,
  });

  const remember = (value: BurstOperationAddress | null) => {
    // Only the lookup address is stored, before the approval is sent; reload
    // always reads progress fresh.
    try {
      if (value) localStorage.setItem(storageKey(modelId), JSON.stringify(value));
      else localStorage.removeItem(storageKey(modelId));
    } catch {
      if (value) setNotice(t('burst.keepOperationId', 'Browser storage is unavailable. Keep operation {id} to follow this rental.', { id: value.operationId }));
    }
    setAddress(value);
  };

  const askForOffer = async () => {
    setNotice('');
    setRecommendation(null);
    try {
      setRecommendation(await recommend({ modelId, contextTokens }).unwrap());
    } catch {
      setNotice(t('burst.offerFailed', 'The burst provider could not answer. It may be busy; try again in a moment.'));
    }
  };

  const approveOffer = async () => {
    const reference = recommendation?.reference;
    const revision = recommendation?.approvalRevision;
    if (submitting.current || address || !reference || !revision) return;
    submitting.current = true;
    setNotice('');
    const target = { pluginId: reference.pluginId, nodeId: reference.nodeId, operationId: randomHex32() };
    remember(target);
    try {
      await approve({ operationId: target.operationId, reference, approvalRevision: revision }).unwrap();
    } catch {
      setNotice(t('burst.approvalUnconfirmed', 'The approval reply was not confirmed. Its progress is followed below; do not approve again.'));
    } finally { submitting.current = false; }
  };

  const releaseRental = async () => {
    if (!address || submitting.current) return;
    submitting.current = true;
    setNotice('');
    try { await release(address).unwrap(); void followed.refetch(); }
    catch { setNotice(t('burst.releaseFailed', 'The release was not confirmed. The provider may be busy; try again. The rental still ends on its own at its latest end time.')); }
    finally { submitting.current = false; }
  };

  const phaseLabels: Record<BurstPhase, string> = {
    approving: t('burst.phase.approving', 'Approving'),
    refused: t('burst.phase.refused', 'Refused'),
    attention: t('burst.phase.attention', 'Needs attention'),
    uncertain: t('burst.phase.uncertain', 'Uncertain'),
    starting: t('burst.phase.starting', 'Starting the machine'),
    installing: t('burst.phase.installing', 'Installing Skulk'),
    joining: t('burst.phase.joining', 'Joining the cluster'),
    downloading: t('burst.phase.downloading', 'Downloading the model'),
    loading: t('burst.phase.loading', 'Loading the model'),
    ready: t('burst.phase.ready', 'Ready'),
    releasing: t('burst.phase.releasing', 'Releasing'),
    released: t('burst.phase.released', 'Released'),
  };

  const title = <h4>{t('burst.title', 'Remote capacity')}</h4>;

  if (address) {
    return <Panel aria-label={t('burst.rentalTitle', 'Remote rental')}>
      {title}
      <div role="status" aria-live="polite">
        {operation ? <>
          <p><strong>{phaseLabels[operation.phase]}</strong></p>
          <p>{operation.message}</p>
          {operation.resourceExpiresAt ? <p>{t('burst.endsBy', 'The rental ends by {time} at the latest.', { time: when(operation.resourceExpiresAt) })}</p> : null}
        </> : followed.isError
          ? <p>{t('burst.followUnavailable', 'Progress is unavailable right now; it is checked again shortly.')}</p>
          : <p>{t('burst.following', 'Reading the rental’s progress…')}</p>}
      </div>
      {notice ? <p role="alert">{notice}</p> : null}
      <Actions>
        {operation && !['released', 'refused', 'releasing'].includes(operation.phase)
          ? <Button type="button" disabled={releasing.isLoading} onClick={() => void releaseRental()}>{t('burst.release', 'Release the machine')}</Button> : null}
        {operation && ['released', 'refused'].includes(operation.phase)
          ? <Button type="button" onClick={() => { remember(null); setRecommendation(null); }}>{t('burst.done', 'Done')}</Button> : null}
      </Actions>
    </Panel>;
  }

  const offer = recommendation?.state === 'offer' ? recommendation.offer : null;
  const available = providers.currentData?.providers.filter((provider) => provider.state === 'available') ?? [];
  const firstProvider = providers.currentData?.providers[0];

  return <Panel aria-label={t('burst.title', 'Remote capacity')}>
    {title}
    {providers.isError
      ? <p>{pluginAccessRefused(providers.error)
        ? t('burst.accessRefused', 'Remote offers need plugin access on this connection.')
        : t('burst.providersUnavailable', 'Burst providers are unavailable right now.')}</p>
      : !providers.currentData
        ? <p>{t('burst.checking', 'Checking for burst providers…')}</p>
        : !firstProvider
          ? <p>{t('burst.noProvider', 'No burst provider is installed. A capability that rents remote machines can provide capacity for this model; install one from the Plugins page.')}</p>
          : available.length === 0
            ? <p>{firstProvider.message}</p>
            : null}
    {recommendation && !offer ? <p role="status">{recommendation.message}</p> : null}
    {offer && recommendation ? <section aria-label={t('burst.offerTitle', 'Remote offer')}>
      <p>{recommendation.message}</p>
      <Facts>
        <dt>{t('burst.gpu', 'Machine')}</dt><dd>{t('burst.gpuValue', '{gpu} in {region}', { gpu: offer.gpu, region: offer.region })}</dd>
        <dt>{t('burst.hourly', 'Quoted price')}</dt><dd>{t('burst.perHour', '{amount} per hour', { amount: formatUsd(offer.hourlyPriceUsd) })}</dd>
        <dt>{t('burst.maximumHourly', 'Most per hour')}</dt><dd>{formatUsd(offer.maximumHourlyPriceUsd)}</dd>
        <dt>{t('burst.maximumTotal', 'Most it can cost')}</dt><dd>{formatUsd(offer.maximumTotalUsd)}</dd>
        <dt>{t('burst.offerExpires', 'Approve by')}</dt><dd>{when(offer.planExpiresAt)}</dd>
        <dt>{t('burst.rentalEnds', 'Rental ends by')}</dt><dd>{when(offer.resourceExpiresAt)}</dd>
        {recommendation.review.map((fact, index) => <Fragment key={index}><dt>{fact.label}</dt><dd>{fact.value}</dd></Fragment>)}
      </Facts>
      <p>{t('burst.idleRelease', 'The machine is released after its model has been idle for five minutes, or when you release it.')}</p>
    </section> : null}
    {notice ? <p role="alert">{notice}</p> : null}
    <Actions>
      {offer
        ? <Button type="button" disabled={approving.isLoading || offer.planExpiresAt * 1000 <= Date.now()} onClick={() => void approveOffer()}>
          {t('burst.approve', 'Approve and rent for at most {amount}', { amount: formatUsd(offer.maximumTotalUsd) })}
        </Button>
        : null}
      {available.length > 0
        ? <Button type="button" disabled={recommending.isLoading || approving.isLoading} onClick={() => void askForOffer()}>
          {recommending.isLoading ? t('burst.asking', 'Asking for an offer…') : recommendation ? t('burst.askAgain', 'Ask for a new offer') : t('burst.getOffer', 'Get a remote offer')}
        </Button>
        : null}
    </Actions>
  </Panel>;
}
