import { useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';
import { useSkulkTranslation } from '../../i18n/tolgee';
import {
  installOperationIds, pluginRefusalDetail, pluginRequestRefused, useGetCatalogSourceQuery, useGetInstallOperationsQuery, useGetManagedRuntimesQuery,
  useGetPluginCatalogQuery, useInstallFromCatalogMutation, useInstallRuntimeReleaseMutation, useLazyGetRuntimeInstallationQuery,
  useLazyGetRuntimeSourceStatusQuery, useRecoverRuntimeInstallationMutation, useSelectBuiltinCatalogMutation, type CatalogListing, type RuntimeRelease,
} from '../../store/endpoints/plugins';
import { randomHex32 } from '../../utils/randomIds';
import { Button } from '../common/Button';
import { Spinner } from '../common/Spinner';
import { CapabilitySetupPanel } from './CapabilitySetupPanel';
import { CatalogConnectPanel } from './CatalogConnectPanel';
import { CatalogInstallProgress, type SetupTarget } from './CatalogInstallProgress';
import { CatalogOfferCard } from './CatalogOfferCard';
import { CatalogRetryReviewPanel, CatalogReviewPanel } from './CatalogReviewPanel';
import {
  catalogOffers, displayTitle, interruptedInstalls, isConsentNeeded, isFollowed, isStartRefusal, journeyInProgress, readJourneys, startCatalogInstall,
  startInstallRetry, type BoundJourney, type CatalogOffer, type InstallStartRefusal, type RetryRequest,
} from './catalogJourney';

type BrowseView =
  | { kind: 'list' }
  | { kind: 'connect' }
  | { kind: 'review'; offer: CatalogOffer; listing: CatalogListing }
  | { kind: 'retry-review'; request: RetryRequest; review: RuntimeRelease }
  | {
    kind: 'install'; title: string; publisher: string; sequence: number; transferBytes: number; updating: boolean; journey: BoundJourney | null; refusal: InstallStartRefusal | null;
    /** The stopped install a retry from this view would retry again. */
    retryOf?: RetryRequest;
  }
  | { kind: 'setup'; target: SetupTarget };

// Only a confirmed binding has an operation to follow; an unconfirmed one is
// continued by installing the same bundle again, and a stopped one by its retry.
function resumedView(): BrowseView {
  const journey = readJourneys().find(isFollowed);
  return journey
    ? { kind: 'install', title: journey.title, publisher: journey.publisher, sequence: journey.sequence, transferBytes: journey.transferBytes, updating: false, journey, refusal: null }
    : { kind: 'list' };
}

/** A retry asked for from another view, such as a card under Installed; `id` tells one click from the next. */
export interface BrowseRetryHandoff { id: number; request: RetryRequest }

/** Props for the Browse view of the Plugins page. */
export interface PluginCatalogBrowseProps {
  /** Open an installed plugin's settings under Installed. */
  onManage?: (pluginId: string) => void;
  /** A retry to start as Browse opens; it is taken once, then `onRetryTaken` is called. */
  retry?: BrowseRetryHandoff | null;
  /** Called when Browse takes the handed-off retry, so it is not started again. */
  onRetryTaken?: () => void;
}

function retryOfJourney(journey: BoundJourney, updating: boolean): RetryRequest {
  return { pluginId: journey.pluginId, title: journey.title, publisher: journey.publisher, sequence: journey.sequence, transferBytes: journey.transferBytes, updating };
}

/**
 * Browse the host's signed catalog: review a release, install or update it,
 * and see what it still needs. A host on the built-in Foxlight store lists it
 * with nothing to paste; a private catalog is added behind its own control,
 * and a host on one can return to the store. An install this browser left
 * unfinished is picked up where it stopped, and an install the host reports
 * stopped is resumed on its own installation rather than installed again.
 */
export function PluginCatalogBrowse({ onManage, retry, onRetryTaken }: PluginCatalogBrowseProps = {}) {
  const { t } = useSkulkTranslation();
  const [view, setView] = useState<BrowseView>(resumedView);
  const source = useGetCatalogSourceQuery();
  const configured = !!source.data?.configured;
  const onStore = !!source.data?.builtin_store;
  const storeAvailable = !!source.data?.builtin_store_available;
  const [selectStore, storeSwitch] = useSelectBuiltinCatalogMutation();
  const [storeNotice, setStoreNotice] = useState('');
  const catalog = useGetPluginCatalogQuery(undefined, { skip: !configured || view.kind === 'connect' });
  const runtimes = useGetManagedRuntimesQuery();
  const operationIds = installOperationIds(runtimes.data?.installations);
  const installs = useGetInstallOperationsQuery(operationIds, { skip: operationIds.length === 0 });
  const [bind] = useInstallFromCatalogMutation();
  const [install] = useInstallRuntimeReleaseMutation();
  const [readInstall] = useLazyGetRuntimeInstallationQuery();
  const [readSource] = useLazyGetRuntimeSourceStatusQuery();
  const [recover] = useRecoverRuntimeInstallationMutation();
  const offers = useMemo(() => catalog.data
    ? catalogOffers(catalog.data, runtimes.data?.installations ?? [], interruptedInstalls(runtimes.data?.installations ?? [], installs.data))
    : [], [catalog.data, runtimes.data, installs.data]);
  // Offers wait for the installations and their installs: an offer made
  // before a stopped install is known would register a second installation.
  const installsKnown = operationIds.length === 0 || installs.data !== undefined || !!installs.error;
  const inventoryKnown = (runtimes.data !== undefined || !!runtimes.error) && installsKnown;

  // One consent starts one install, even if the button is clicked twice.
  const starting = useRef(false);
  const beginInstall = async (offer: CatalogOffer, listing: CatalogListing) => {
    if (starting.current) return;
    starting.current = true;
    const base = { kind: 'install' as const, title: displayTitle(offer.entry), publisher: offer.entry.publisher, sequence: offer.entry.sequence, transferBytes: offer.entry.transfer_bytes, updating: offer.state === 'update' };
    setView({ ...base, journey: null, refusal: null });
    const outcome = await startCatalogInstall({ bind, install }, offer, listing, pluginRefusalDetail, randomHex32, pluginRequestRefused);
    starting.current = false;
    setView(isStartRefusal(outcome) ? { ...base, journey: null, refusal: outcome } : { ...base, transferBytes: outcome.transferBytes, journey: outcome, refusal: null });
  };
  // A retry follows the same progress a fresh install does, through
  // activation, once this browser holds consent to the release it retries.
  const beginRetry = async (request: RetryRequest, reviewedDigest: string | null = null) => {
    if (starting.current) return;
    starting.current = true;
    const base = { kind: 'install' as const, title: request.title, publisher: request.publisher, sequence: request.sequence, transferBytes: request.transferBytes, updating: request.updating, retryOf: request };
    setView({ ...base, journey: null, refusal: null });
    const outcome = await startInstallRetry({
      readInstall: (pluginId) => readInstall(pluginId, false), readSource: (pluginId) => readSource(pluginId, false), recover,
    }, request, reviewedDigest, pluginRefusalDetail, pluginRequestRefused);
    starting.current = false;
    if (isConsentNeeded(outcome)) setView({ kind: 'retry-review', request, review: outcome.review });
    else if (isStartRefusal(outcome)) setView({ ...base, journey: null, refusal: outcome });
    else setView({ ...base, title: outcome.title, publisher: outcome.publisher, sequence: outcome.sequence, transferBytes: outcome.transferBytes, journey: outcome, refusal: null });
  };
  // A retry handed over from Installed starts once per click, even if this
  // view mounts again before the handoff is cleared.
  const takenRetry = useRef<number | null>(null);
  useEffect(() => {
    if (!retry || takenRetry.current === retry.id) return;
    takenRetry.current = retry.id;
    onRetryTaken?.();
    void beginRetry(retry.request);
    // beginRetry reads current state through refs and stable RTK triggers.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [retry]);
  const toList = () => {
    setView({ kind: 'list' });
    // A query that has not started yet (Browse opened on a resumed install)
    // has nothing to refetch; it starts when the list needs it.
    if (!catalog.isUninitialized) void catalog.refetch();
    if (!runtimes.isUninitialized) void runtimes.refetch();
    if (!installs.isUninitialized) void installs.refetch();
  };
  // Opening the panel reads the source again: the store's first catalog read
  // records it, so the revision this page loaded with may be behind.
  const openConnect = () => {
    setStoreNotice('');
    setView({ kind: 'connect' });
    void source.refetch();
  };
  const switchToStore = async () => {
    if (!source.data || storeSwitch.isLoading) return;
    setStoreNotice('');
    try {
      await selectStore({ expected_revision: source.data.revision }).unwrap();
    } catch (error) {
      setStoreNotice(pluginRefusalDetail(error) ?? t('plugins.catalog.useStoreFailed', 'The host did not switch to the Foxlight store. Try again.'));
    }
  };
  const resume = (journey: BoundJourney, updating: boolean) => setView({
    kind: 'install', title: journey.title, publisher: journey.publisher, sequence: journey.sequence, transferBytes: journey.transferBytes, updating, journey, refusal: null,
  });
  // Read when the list renders: leaving an install's progress returns here.
  const journeys = view.kind === 'list' ? readJourneys() : [];

  if (view.kind === 'install') {
    const retryOf = view.journey ? retryOfJourney(view.journey, view.updating) : view.retryOf;
    // Each attempt is its own progress: a retry follows the same operation again.
    return <CatalogInstallProgress key={view.journey ? `${view.journey.pluginId}/${view.journey.startedAt}` : view.refusal ? 'refused' : 'starting'}
      title={view.title} publisher={view.publisher} sequence={view.sequence} transferBytes={view.transferBytes}
      updating={view.updating} journey={view.journey} refusal={view.refusal} onDone={(target) => setView({ kind: 'setup', target })} onBack={toList}
      onRetry={retryOf ? () => void beginRetry(retryOf) : undefined} />;
  }
  if (view.kind === 'retry-review') {
    return <CatalogRetryReviewPanel title={view.request.title} updating={view.request.updating} review={view.review} onCancel={toList}
      onRetry={() => void beginRetry(view.request, view.review.runtime_digest)} />;
  }
  if (view.kind === 'setup') return <CapabilitySetupPanel target={view.target} onManage={onManage} onBack={toList} />;
  if (view.kind === 'review') {
    return <CatalogReviewPanel offer={view.offer} onCancel={toList} onInstall={() => void beginInstall(view.offer, view.listing)} />;
  }
  if (source.isLoading) return <Centered><Spinner size={22} /></Centered>;
  if (source.error) return <Notice role="alert">{pluginRefusalDetail(source.error) ?? t('plugins.catalog.sourceUnavailable', 'The catalog settings of this host could not be read.')}</Notice>;
  if (!source.data) return null;
  if (!configured || view.kind === 'connect') {
    return <CatalogConnectPanel status={source.data} replacesStore={onStore} onConnected={() => { setView({ kind: 'list' }); void source.refetch(); }}
      onCancel={configured ? () => setView({ kind: 'list' }) : undefined} />;
  }
  const updated = catalog.data ? new Date(catalog.data.created_at * 1000).toLocaleDateString() : '';
  return <section aria-labelledby="catalog-browse-title">
    <Heading>
      <div>
        <h2 id="catalog-browse-title">{t('plugins.catalog.browseTitle', 'Browse capabilities')}</h2>
        {catalog.data ? <p>{onStore
          ? t('plugins.catalog.storeLead', 'From the Foxlight capability store. Signatures verified; listing updated {date}.', { date: updated })
          : t('plugins.catalog.browseLead', 'From the {publisher} catalog. Signature verified; listing updated {date}.', { publisher: catalog.data.publisher, date: updated })}</p> : null}
      </div>
      <SourceActions>
        {onStore
          ? <Button variant="ghost" size="sm" onClick={openConnect}>{t('plugins.catalog.addPrivate', 'Add a private catalog')}</Button>
          : <>
            {storeAvailable ? <Button variant="outline" size="sm" disabled={storeSwitch.isLoading} onClick={() => void switchToStore()}>{t('plugins.catalog.useStore', 'Use the Foxlight store')}</Button> : null}
            <Button variant="ghost" size="sm" onClick={openConnect}>{t('plugins.catalog.changeCatalog', 'Change catalog')}</Button>
          </>}
      </SourceActions>
    </Heading>
    {storeNotice ? <Notice role="alert">{storeNotice}</Notice> : null}
    {catalog.isFetching && !catalog.data ? <Centered><Spinner size={22} /><span>{t('plugins.catalog.reading', 'Reading and verifying the catalog…')}</span></Centered> : null}
    {catalog.error ? <Notice role="alert">
      {pluginRefusalDetail(catalog.error) ?? t('plugins.catalog.readFailed', 'The catalog could not be read.')}
      <Button variant="outline" size="sm" disabled={catalog.isFetching} onClick={() => void catalog.refetch()}>{t('plugins.catalog.retry', 'Try again')}</Button>
    </Notice> : null}
    {catalog.data && !inventoryKnown ? <Centered><Spinner size={22} /></Centered> : null}
    {catalog.data && inventoryKnown && offers.length === 0 ? <p>{t('plugins.catalog.empty', 'This catalog lists nothing yet.')}</p> : null}
    <Offers>{catalog.data && inventoryKnown ? offers.map((offer) => {
      const progress = journeyInProgress(offer.entry.bundle_id, offer.entry.sequence, journeys);
      const stopped = offer.retry;
      return <CatalogOfferCard key={offer.entry.bundle_id} offer={offer} progress={progress}
        onResume={() => progress && resume(progress, offer.state === 'update')}
        onRetry={() => {
          if (!stopped) return;
          void beginRetry({
            pluginId: stopped.pluginId, title: displayTitle(offer.entry), publisher: stopped.review.publisher, sequence: stopped.review.sequence,
            transferBytes: stopped.review.artifact_bytes, updating: stopped.updating,
          });
        }}
        onReview={() => setView({ kind: 'review', offer, listing: catalog.data! })}
        onSetUp={() => offer.installed && setView({ kind: 'setup', target: { pluginId: offer.installed.plugin_id, title: displayTitle(offer.entry) } })} />;
    }) : null}</Offers>
  </section>;
}

const Heading = styled.div`
  display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin-bottom: 16px;
  h2 { font-size: 22px; margin: 0; letter-spacing: -.01em; color: ${({ theme }) => theme.colors.text}; }
  p { margin: 6px 0 0; font-size: 14px; color: ${({ theme }) => theme.colors.textSecondary}; }
`;
const SourceActions = styled.div`display: flex; flex-wrap: wrap; gap: 8px;`;
const Offers = styled.div`display: flex; flex-direction: column; gap: 12px;`;
const Centered = styled.div`display: flex; align-items: center; gap: 12px; padding: 24px 4px; color: ${({ theme }) => theme.colors.textSecondary}; font-size: 14px;`;
const Notice = styled.div`
  display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; padding: 14px 16px; margin-bottom: 12px; border-radius: 12px; font-size: 14px; line-height: 1.5;
  border: 1px solid ${({ theme }) => theme.colors.borderDanger}; background: ${({ theme }) => theme.colors.errorBg}; color: ${({ theme }) => theme.colors.text};
`;
