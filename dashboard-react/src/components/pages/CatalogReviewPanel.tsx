import { useState } from 'react';
import styled from 'styled-components';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { Button } from '../common/Button';
import { SectionLabel, Surface } from '../common/Surfaces';
import { canSpendMoney, capabilityName, displayTitle, formatMegabytes, platformLabel, type CatalogOffer } from './catalogJourney';

/** Props for reviewing one catalog release before consenting to it. */
export interface CatalogReviewPanelProps {
  offer: CatalogOffer;
  /** Start the install with the reviewed release; only called after consent. */
  onInstall: () => void;
  onCancel: () => void;
}

/**
 * Everything the signed listing says about a release, with one consent.
 *
 * The host checks again at install that the release it downloads is exactly
 * the one reviewed here, so the permissions accepted are the ones it gets.
 */
export function CatalogReviewPanel({ offer, onInstall, onCancel }: CatalogReviewPanelProps) {
  const { t } = useSkulkTranslation();
  const [accepted, setAccepted] = useState(false);
  const { entry, installed, state } = offer;
  const title = displayTitle(entry);
  const updating = state === 'update' && installed !== null;
  const installedSequence = installed?.release?.sequence;
  return <Panel aria-labelledby="catalog-review-title">
    <h2 id="catalog-review-title">{updating ? t('plugins.catalog.reviewUpdateTitle', 'Review the update to {title}', { title }) : t('plugins.catalog.reviewTitle', 'Review {title}', { title })}</h2>
    <Lead>{t('plugins.catalog.reviewLead', 'Everything below comes from the release’s signed listing.')}</Lead>
    {updating ? <Callout>{installedSequence !== undefined
      ? t('plugins.catalog.updateKeeps', 'This replaces release {installed} with release {offered} on the same installation. Its settings and saved work stay.', { installed: installedSequence, offered: entry.sequence })
      : t('plugins.catalog.updateKeepsUnknown', 'This replaces the installed release with release {offered} on the same installation. Its settings and saved work stay.', { offered: entry.sequence })}</Callout> : null}
    <Facts>
      <dt>{t('plugins.catalog.publisher', 'Publisher')}</dt><dd>{entry.publisher}</dd>
      <dt>{t('plugins.catalog.versionLabel', 'Version')}</dt><dd>{t('plugins.catalog.versionValue', '{version}, release {sequence}', { version: entry.bundle_version, sequence: entry.sequence })}</dd>
      <dt>{t('plugins.catalog.download', 'Download')}</dt><dd>{formatMegabytes(entry.transfer_bytes)}</dd>
      <dt>{t('plugins.catalog.runsOn', 'Runs on')}</dt><dd>{platformLabel(entry)}</dd>
      <dt>{t('plugins.catalog.validUntil', 'Release valid until')}</dt><dd>{new Date(entry.expires_at * 1000).toLocaleDateString()}</dd>
    </Facts>
    <Section>
      <SectionLabel>{t('plugins.catalog.allowedTo', 'What it will be allowed to do')}</SectionLabel>
      <ul>{entry.permissions.map((permission) => <li key={permission}>{permission}</li>)}</ul>
    </Section>
    {entry.surfaces.length > 0 ? <Section>
      <SectionLabel>{t('plugins.catalog.screens', 'Screens it opens')}</SectionLabel>
      <ul>{entry.surfaces.map((surface) => <li key={surface}>{surface}</li>)}</ul>
    </Section> : null}
    {entry.descriptors.length > 0 ? <Section>
      <SectionLabel>{t('plugins.catalog.adds', 'Capabilities it adds')}</SectionLabel>
      <Chips>{entry.descriptors.map((descriptor) => <code key={descriptor} title={descriptor}>{capabilityName(descriptor)}</code>)}</Chips>
      {entry.operations ? <Small>{t('plugins.catalog.durable', 'Its long jobs are recorded on this host and survive a restart.')}</Small> : null}
    </Section> : null}
    <Section>
      <SectionLabel>{t('plugins.catalog.money', 'Money')}</SectionLabel>
      <p>{canSpendMoney(entry)
        ? t('plugins.catalog.moneyBillable', 'Some of its actions can spend money. Each one asks for your approval first.')
        : t('plugins.catalog.moneyNone', 'It cannot spend money. Its actions run on your own hardware.')}</p>
    </Section>
    <Section>
      <SectionLabel>{t('plugins.catalog.models', 'Models')}</SectionLabel>
      <p>{t('plugins.catalog.modelsNote', 'Any models it uses are downloaded separately, from the Model Store, which shows each model’s licence before download.')}</p>
    </Section>
    <Consent>
      <input id="catalog-consent" type="checkbox" checked={accepted} onChange={(event) => setAccepted(event.target.checked)} />
      <label htmlFor="catalog-consent">{t('plugins.catalog.accept', 'I accept these permissions for this release.')}</label>
    </Consent>
    <Actions>
      <Button variant="ghost" onClick={onCancel}>{t('common.cancel', 'Cancel')}</Button>
      <Button variant="primary" disabled={!accepted} onClick={onInstall}>{updating ? t('plugins.catalog.update', 'Update') : t('plugins.catalog.install', 'Install')}</Button>
    </Actions>
  </Panel>;
}

const Panel = styled(Surface)`
  max-width: 720px; padding: 28px; display: flex; flex-direction: column; gap: 14px;
  h2 { font-size: 22px; margin: 0; letter-spacing: -.01em; color: ${({ theme }) => theme.colors.text}; }
`;
const Lead = styled.p`margin: -6px 0 0; color: ${({ theme }) => theme.colors.textSecondary}; font-size: 14px;`;
const Callout = styled.p`
  margin: 0; padding: 12px 14px; border-radius: 10px; font-size: 13.5px; line-height: 1.5;
  border: 1px solid ${({ theme }) => theme.colors.borderLive}; background: ${({ theme }) => theme.colors.liveBg}; color: ${({ theme }) => theme.colors.text};
`;
const Facts = styled.dl`
  display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 8px 20px; margin: 0; padding: 16px; font-size: 14px;
  border-radius: 12px; border: 1px solid ${({ theme }) => theme.colors.border}; background: ${({ theme }) => theme.colors.surfaceHover};
  dt { color: ${({ theme }) => theme.colors.textSecondary}; }
  dd { margin: 0; color: ${({ theme }) => theme.colors.text}; overflow-wrap: anywhere; }
`;
const Section = styled.section`
  display: flex; flex-direction: column; gap: 8px;
  ul { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 4px; font-size: 14px; color: ${({ theme }) => theme.colors.text}; }
  p { margin: 0; font-size: 14px; line-height: 1.5; color: ${({ theme }) => theme.colors.text}; }
`;
const Chips = styled.div`
  display: flex; flex-wrap: wrap; gap: 6px;
  code { padding: 3px 8px; border-radius: 6px; border: 1px solid ${({ theme }) => theme.colors.border}; background: ${({ theme }) => theme.colors.surfaceHover}; font: 12px ${({ theme }) => theme.fonts.mono}; color: ${({ theme }) => theme.colors.textSecondary}; }
`;
const Small = styled.p`&& { font-size: 12.5px; color: ${({ theme }) => theme.colors.metadataText}; }`;
const Consent = styled.div`
  display: flex; align-items: flex-start; gap: 10px; padding: 14px 16px; border-radius: 12px; border: 1px solid ${({ theme }) => theme.colors.borderStrong};
  input { margin-top: 3px; } label { font-size: 14px; font-weight: 600; color: ${({ theme }) => theme.colors.text}; }
`;
const Actions = styled.div`display: flex; justify-content: flex-end; flex-wrap: wrap; gap: 8px;`;
