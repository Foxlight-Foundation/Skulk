import { useMemo, useRef, useState } from 'react';
import styled from 'styled-components';
import { useSkulkTranslation } from '../../i18n/tolgee';
import {
  pluginRefusalDetail, pluginRequestRefused, useGetCatalogSourceQuery, useGetManagedRuntimesQuery, useGetPluginCatalogQuery,
  useInstallFromCatalogMutation, useInstallRuntimeReleaseMutation, type CatalogListing,
} from '../../store/endpoints/plugins';
import { randomHex32 } from '../../utils/randomIds';
import { Button } from '../common/Button';
import { Spinner } from '../common/Spinner';
import { CapabilitySetupPanel } from './CapabilitySetupPanel';
import { CatalogConnectPanel } from './CatalogConnectPanel';
import { CatalogInstallProgress, type SetupTarget } from './CatalogInstallProgress';
import { CatalogOfferCard } from './CatalogOfferCard';
import { CatalogReviewPanel } from './CatalogReviewPanel';
import {
  catalogOffers, displayTitle, isBound, isStartRefusal, readJourneys, startCatalogInstall,
  type BoundJourney, type CatalogOffer, type InstallStartRefusal,
} from './catalogJourney';

type BrowseView =
  | { kind: 'list' }
  | { kind: 'connect' }
  | { kind: 'review'; offer: CatalogOffer; listing: CatalogListing }
  | { kind: 'install'; title: string; publisher: string; sequence: number; transferBytes: number; updating: boolean; journey: BoundJourney | null; refusal: InstallStartRefusal | null }
  | { kind: 'setup'; target: SetupTarget };

// Only a confirmed binding has an operation to follow; an unconfirmed one is
// continued by installing the same bundle again.
function resumedView(): BrowseView {
  const journey = readJourneys().find(isBound);
  return journey
    ? { kind: 'install', title: journey.title, publisher: journey.publisher, sequence: journey.sequence, transferBytes: journey.transferBytes, updating: false, journey, refusal: null }
    : { kind: 'list' };
}

/** Props for the Browse view of the Plugins page. */
export interface PluginCatalogBrowseProps {
  /** Open an installed plugin's settings under Installed. */
  onManage?: (pluginId: string) => void;
}

/**
 * Browse the host's signed catalog: connect one, review a release, install
 * or update it, and see what it still needs. An install this browser left
 * unfinished is picked up where it stopped.
 */
export function PluginCatalogBrowse({ onManage }: PluginCatalogBrowseProps = {}) {
  const { t } = useSkulkTranslation();
  const [view, setView] = useState<BrowseView>(resumedView);
  const source = useGetCatalogSourceQuery();
  const configured = !!source.data?.configured;
  const catalog = useGetPluginCatalogQuery(undefined, { skip: !configured || view.kind === 'connect' });
  const runtimes = useGetManagedRuntimesQuery();
  const [bind] = useInstallFromCatalogMutation();
  const [install] = useInstallRuntimeReleaseMutation();
  const offers = useMemo(() => catalog.data ? catalogOffers(catalog.data, runtimes.data?.installations ?? []) : [], [catalog.data, runtimes.data]);

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
  const toList = () => { setView({ kind: 'list' }); void catalog.refetch(); void runtimes.refetch(); };

  if (view.kind === 'install') {
    return <CatalogInstallProgress title={view.title} publisher={view.publisher} sequence={view.sequence} transferBytes={view.transferBytes}
      updating={view.updating} journey={view.journey} refusal={view.refusal} onDone={(target) => setView({ kind: 'setup', target })} onBack={toList} />;
  }
  if (view.kind === 'setup') return <CapabilitySetupPanel target={view.target} onManage={onManage} onBack={toList} />;
  if (view.kind === 'review') {
    return <CatalogReviewPanel offer={view.offer} onCancel={toList} onInstall={() => void beginInstall(view.offer, view.listing)} />;
  }
  if (source.isLoading) return <Centered><Spinner size={22} /></Centered>;
  if (source.error) return <Notice role="alert">{pluginRefusalDetail(source.error) ?? t('plugins.catalog.sourceUnavailable', 'The catalog settings of this host could not be read.')}</Notice>;
  if (!source.data) return null;
  if (!configured || view.kind === 'connect') {
    return <CatalogConnectPanel status={source.data} onConnected={() => { setView({ kind: 'list' }); void source.refetch(); }}
      onCancel={configured ? () => setView({ kind: 'list' }) : undefined} />;
  }
  return <section aria-labelledby="catalog-browse-title">
    <Heading>
      <div>
        <h2 id="catalog-browse-title">{t('plugins.catalog.browseTitle', 'Browse capabilities')}</h2>
        {catalog.data ? <p>{t('plugins.catalog.browseLead', 'From the {publisher} catalog. Signature verified; listing updated {date}.', { publisher: catalog.data.publisher, date: new Date(catalog.data.created_at * 1000).toLocaleDateString() })}</p> : null}
      </div>
      <Button variant="ghost" size="sm" onClick={() => setView({ kind: 'connect' })}>{t('plugins.catalog.changeCatalog', 'Change catalog')}</Button>
    </Heading>
    {catalog.isFetching && !catalog.data ? <Centered><Spinner size={22} /><span>{t('plugins.catalog.reading', 'Reading and verifying the catalog…')}</span></Centered> : null}
    {catalog.error ? <Notice role="alert">
      {pluginRefusalDetail(catalog.error) ?? t('plugins.catalog.readFailed', 'The catalog could not be read.')}
      <Button variant="outline" size="sm" disabled={catalog.isFetching} onClick={() => void catalog.refetch()}>{t('plugins.catalog.retry', 'Try again')}</Button>
    </Notice> : null}
    {catalog.data && offers.length === 0 ? <p>{t('plugins.catalog.empty', 'This catalog lists nothing yet.')}</p> : null}
    <Offers>{catalog.data ? offers.map((offer) => <CatalogOfferCard key={offer.entry.bundle_id} offer={offer}
      onReview={() => setView({ kind: 'review', offer, listing: catalog.data! })}
      onSetUp={() => offer.installed && setView({ kind: 'setup', target: { pluginId: offer.installed.plugin_id, title: displayTitle(offer.entry) } })} />) : null}</Offers>
  </section>;
}

const Heading = styled.div`
  display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin-bottom: 16px;
  h2 { font-size: 22px; margin: 0; letter-spacing: -.01em; color: ${({ theme }) => theme.colors.text}; }
  p { margin: 6px 0 0; font-size: 14px; color: ${({ theme }) => theme.colors.textSecondary}; }
`;
const Offers = styled.div`display: flex; flex-direction: column; gap: 12px;`;
const Centered = styled.div`display: flex; align-items: center; gap: 12px; padding: 24px 4px; color: ${({ theme }) => theme.colors.textSecondary}; font-size: 14px;`;
const Notice = styled.div`
  display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; padding: 14px 16px; margin-bottom: 12px; border-radius: 12px; font-size: 14px; line-height: 1.5;
  border: 1px solid ${({ theme }) => theme.colors.borderDanger}; background: ${({ theme }) => theme.colors.errorBg}; color: ${({ theme }) => theme.colors.text};
`;
