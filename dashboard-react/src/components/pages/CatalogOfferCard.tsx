import styled from 'styled-components';
import { FiExternalLink } from 'react-icons/fi';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { Button } from '../common/Button';
import { Monogram, StatusPill } from '../common/Surfaces';
import { canSpendMoney, displayTitle, formatMegabytes, monogram, platformLabel, type BoundJourney, type CatalogOffer } from './catalogJourney';
import type { InstalledCardAction } from './installedCapability';

/** Props for one catalog card: the offer and what its button does. */
export interface CatalogOfferCardProps {
  offer: CatalogOffer;
  /** Review the offered release before installing or updating it. */
  onReview: () => void;
  /** Open the installed plugin's own page, where it is set up or managed. */
  onSetUp: () => void;
  /**
   * What an installed release's card offers, from what its nodes report now:
   * open its screen, set it up, or manage it. Manage when not given.
   */
  installedAction?: InstalledCardAction;
  /** The install of this release this browser is following, if any. */
  progress?: BoundJourney | null;
  /** Return to that install's progress. */
  onResume?: () => void;
  /** Retry the stopped install of this bundle, when the offer is a retry. */
  onRetry?: () => void;
  /** Hold a new install while an installation's install status is unknown: it may be a stopped install of this bundle. */
  installPaused?: boolean;
}

/** One bundle a catalog offers, with the facts a newcomer decides on at a glance. */
export function CatalogOfferCard({ offer, onReview, onSetUp, installedAction = { kind: 'manage' }, progress = null, onResume, onRetry, installPaused = false }: CatalogOfferCardProps) {
  const { t } = useSkulkTranslation();
  const { entry, installed, state } = offer;
  const title = displayTitle(entry);
  const installedSequence = installed?.release?.sequence ?? null;
  // A release being installed is shown as such: offering it again would start
  // a second installation beside the first.
  const busy = progress !== null && state !== 'installed';
  const fit = busy ? <StatusPill tone="live">{state === 'update' ? t('plugins.catalog.updatingPill', 'Updating') : t('plugins.catalog.installingPill', 'Installing')}</StatusPill> : {
    available: <StatusPill tone="healthy">{t('plugins.catalog.fitsHost', 'Fits this host')}</StatusPill>,
    update: <StatusPill tone="live">{t('plugins.catalog.updateAvailable', 'Update available')}</StatusPill>,
    installed: <StatusPill tone="healthy">{t('plugins.catalog.installed', 'Installed')}</StatusPill>,
    unfit: <StatusPill tone="neutral">{t('plugins.catalog.notForHost', 'Not built for this host')}</StatusPill>,
    retry: offer.retry?.stopped === false
      ? <StatusPill tone="live">{t('plugins.catalog.installingPill', 'Installing')}</StatusPill>
      : <StatusPill tone="live">{t('plugins.catalog.needsRetry', 'Install needs a retry')}</StatusPill>,
  }[state];
  const opens = entry.surfaces.length > 0 ? t('plugins.catalog.opens', 'Opens {surfaces}.', { surfaces: entry.surfaces.join(', ') }) : null;
  const primary = busy ? <Button variant="primary" size="sm" onClick={onResume}>{t('plugins.catalog.showProgress', 'Show progress')}</Button> : {
    available: <Button variant="primary" size="sm" disabled={installPaused} onClick={onReview}>{t('plugins.catalog.reviewInstall', 'Review and install')}</Button>,
    update: <Button variant="primary" size="sm" onClick={onReview}>{t('plugins.catalog.reviewUpdate', 'Review update')}</Button>,
    // A running plugin opens its screen in a new tab, as its own page does; one
    // that needs the owner is set up; anything else is managed on its page.
    installed: installedAction.kind === 'open'
      ? <OpenLink href={installedAction.surface.url} target="_blank" rel="noopener noreferrer" title={t('plugins.catalog.opensInNewTab', 'Opens in a new tab')}>
        {t('plugins.setup.open', 'Open {surface}', { surface: installedAction.surface.title })}<FiExternalLink aria-hidden="true" />
      </OpenLink>
      : installedAction.kind === 'set-up'
        ? <Button variant="primary" size="sm" onClick={onSetUp}>{t('plugins.catalog.setUp', 'Set up')}</Button>
        : <Button variant="outline" size="sm" onClick={onSetUp}>{t('plugins.catalog.manage', 'Manage')}</Button>,
    unfit: null,
    // Another install would register a second installation beside the
    // stopped one, so the stopped one is resumed instead.
    retry: <Button variant="primary" size="sm" onClick={onRetry}>{offer.retry?.stopped === false ? t('plugins.catalog.showProgress', 'Show progress') : t('plugins.catalog.resumeInstall', 'Resume install')}</Button>,
  }[state];
  return <Card>
    <Mark aria-hidden="true">{monogram(title)}</Mark>
    <div>
      <Row><h2>{title}</h2>{fit}{canSpendMoney(entry) ? <StatusPill tone="neutral">{t('plugins.catalog.canSpend', 'Can spend money, with approval')}</StatusPill> : <StatusPill tone="neutral">{t('plugins.catalog.noSpending', 'Cannot spend money')}</StatusPill>}</Row>
      {opens ? <Description>{opens}</Description> : null}
      {state === 'update' && installedSequence !== null ? <Description>{t('plugins.catalog.updateFrom', 'Release {installed} is installed; release {offered} is built for this host.', { installed: installedSequence, offered: entry.sequence })}</Description> : null}
      {state === 'retry' && offer.retry ? <Description>{offer.retry.stopped
        ? t('plugins.catalog.retryHelp', 'Installing release {sequence} stopped before it finished. Resume it to finish on the same installation; nothing is installed beside it.', { sequence: offer.retry.review.sequence })
        : t('plugins.catalog.underWayHelp', 'Release {sequence} is being installed on this host. Follow it to finish on the same installation; nothing is installed beside it.', { sequence: offer.retry.review.sequence })}</Description> : null}
      {state === 'unfit' ? <Description>{t('plugins.catalog.unfitHelp', 'Every listed release was built for a different Skulk build or platform. Its publisher has to build one for this host.')}</Description> : null}
      <Metadata>
        <span>{t('plugins.catalog.by', 'by {publisher}', { publisher: entry.publisher })}</span>
        <span>{t('plugins.catalog.version', '{version} (release {sequence})', { version: entry.bundle_version, sequence: entry.sequence })}</span>
        <span>{formatMegabytes(entry.transfer_bytes)}</span>
        <span>{platformLabel(entry)}</span>
      </Metadata>
    </div>
    <Actions>{primary}</Actions>
  </Card>;
}

const Card = styled.article`
  padding: 18px 20px; border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 14px;
  background: ${({ theme }) => theme.colors.surface}; display: grid; grid-template-columns: 48px minmax(0, 1fr) auto; gap: 16px; align-items: center;
  color: ${({ theme }) => theme.colors.text};
  h2 { font-size: 17px; font-weight: 600; margin: 0; overflow-wrap: anywhere; }
  @media (max-width: 600px) { grid-template-columns: 44px minmax(0, 1fr); align-items: start; > div:last-child { grid-column: 1 / -1; } }
`;
const Mark = styled(Monogram)`border-radius: 12px; border: 1px solid ${({ theme }) => theme.colors.border}; font-size: 15px;`;
const Row = styled.div`display: flex; align-items: center; flex-wrap: wrap; gap: 10px; min-width: 0;`;
const Description = styled.p`margin: 6px 0 0; color: ${({ theme }) => theme.colors.textSecondary}; font-size: 13.5px; line-height: 1.5;`;
const Metadata = styled.div`
  display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 10px; font: 11.5px ${({ theme }) => theme.fonts.mono}; color: ${({ theme }) => theme.colors.metadataText};
  > span { overflow-wrap: anywhere; min-width: 0; }
`;
const Actions = styled.div`display: flex; align-items: center; gap: 8px;`;
// Styled as the card's primary button: opening a running plugin is its next step.
const OpenLink = styled.a`
  box-sizing: border-box; display: inline-flex; align-items: center; gap: 6px; min-height: 30px; padding: 0 10px;
  border-radius: ${({ theme }) => theme.radii.md}; border: 1px solid ${({ theme }) => theme.colors.goldDim};
  color: ${({ theme }) => theme.colors.accentText}; background: transparent; text-decoration: none;
  font: ${({ theme }) => theme.fontSizes.xs} ${({ theme }) => theme.fonts.body}; white-space: nowrap;
  &:hover { background: ${({ theme }) => theme.colors.goldBg}; }
  &:focus-visible { outline: none; box-shadow: ${({ theme }) => theme.colors.focusRing}; }
`;
