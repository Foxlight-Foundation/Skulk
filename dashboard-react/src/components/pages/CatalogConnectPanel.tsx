import { useState } from 'react';
import styled from 'styled-components';
import { operatorSession } from '../../auth/operatorSession';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { apiSlice } from '../../store/api';
import { useAppDispatch } from '../../store/hooks';
import { Button } from '../common/Button';
import { SectionLabel, Surface } from '../common/Surfaces';
import { pluginRefusalDetail, type CatalogSourceUpdate, type RuntimeSourceStatus } from '../../store/endpoints/plugins';
import { decodeInvitation, isCatalogAddress, type CatalogInvitation } from './catalogJourney';

/** Props for connecting this host to a capability catalog. */
export interface CatalogConnectPanelProps {
  /** Current catalog source readiness; its revision fences the change. */
  status: RuntimeSourceStatus;
  /** Called once the host has accepted the catalog source. */
  onConnected: () => void;
  /** Leave without changing anything; offered when a catalog is already configured. */
  onCancel?: () => void;
  /** The host reads the built-in Foxlight store now; a private catalog would replace it. */
  replacesStore?: boolean;
}

const DAY_SECONDS = 86_400;

/**
 * Connect this host to a catalog from an invitation code, or from the same
 * facts entered by hand. Nothing installs here: the host only learns where to
 * look and which key signs the listings it will show.
 */
export function CatalogConnectPanel({ status, onConnected, onCancel, replacesStore = false }: CatalogConnectPanelProps) {
  const { t } = useSkulkTranslation();
  const dispatch = useAppDispatch();
  const [busy, setBusy] = useState(false);
  const [code, setCode] = useState('');
  const [manual, setManual] = useState(false);
  const [address, setAddress] = useState('');
  const [publisher, setPublisher] = useState('');
  const [publicKey, setPublicKey] = useState('');
  const [days, setDays] = useState('365');
  const [notice, setNotice] = useState('');
  // Expiry is judged, and a typed trust period counted, from when the panel opened.
  const [openedAt] = useState(() => Math.floor(Date.now() / 1000));
  const invitation = code.trim() ? decodeInvitation(code, openedAt) : null;
  const manualInvitation: CatalogInvitation | null = manual && isCatalogAddress(address.trim())
    && /^[a-z0-9][a-z0-9._-]{0,63}$/.test(publisher.trim()) && /^[0-9a-f]{64}$/.test(publicKey.trim().toLowerCase())
    && Number.isInteger(Number(days)) && Number(days) > 0 && Number(days) <= 3650
    ? { baseUrl: address.trim(), publisher: publisher.trim(), publicKey: publicKey.trim().toLowerCase(), trustExpiresAt: openedAt + Number(days) * DAY_SECONDS }
    : null;
  const chosen = manual ? manualInvitation : invitation;
  const connect = async () => {
    if (!chosen || busy) return;
    setNotice('');
    setBusy(true);
    // An invitation may carry the catalog's credential. Like a release source's,
    // it must not enter RTK mutation arguments, action history or cached errors,
    // so the request goes straight to the host and only the cache is invalidated.
    const update: CatalogSourceUpdate = {
      expected_revision: status.revision,
      base_url: chosen.baseUrl,
      trust: { revision: (status.trust_revision ?? 0) + 1, expires_at: chosen.trustExpiresAt, publishers: { [chosen.publisher]: chosen.publicKey } },
      ...(chosen.token ? { token: chosen.token } : {}),
    };
    let body = JSON.stringify(update);
    let refusal: string | null = null;
    try {
      const response = await operatorSession.fetch('/v1/plugins/managed/catalog/source', {
        method: 'POST', credentials: 'same-origin', cache: 'no-store', redirect: 'error',
        headers: { 'Content-Type': 'application/json', 'X-Skulk-Dashboard': 'pairing-v1' },
        body, signal: AbortSignal.timeout(35000),
      });
      body = '';
      if (!response.ok) {
        // The host's refusal sentence names no credential; read it for the operator.
        refusal = pluginRefusalDetail({ status: response.status, data: await response.json().catch(() => null) });
        throw new Error('catalog source update refused');
      }
      setCode('');
      dispatch(apiSlice.util.invalidateTags(['PluginCatalog']));
      onConnected();
    } catch {
      setNotice(refusal ?? t('plugins.catalog.connectUnconfirmed', 'The catalog was not connected. Check the code, then try again.'));
    } finally {
      body = '';
      setBusy(false);
    }
  };
  return <Panel aria-labelledby="catalog-connect-title">
    <h2 id="catalog-connect-title">{replacesStore ? t('plugins.catalog.addPrivate', 'Add a private catalog') : t('plugins.catalog.connectTitle', 'Connect a capability catalog')}</h2>
    <Lead>{replacesStore
      ? t('plugins.catalog.privateLead', 'Paste the invitation code you were given. This host then reads that catalog instead of the Foxlight store; you can switch back at any time.')
      : t('plugins.catalog.connectLead', 'Paste the invitation code you were given. It names the catalog and the key its releases are signed with.')}</Lead>
    {!manual ? <>
      <Label htmlFor="catalog-invitation">{t('plugins.catalog.invitationCode', 'Invitation code')}</Label>
      <CodeInput id="catalog-invitation" value={code} rows={3} spellCheck={false} autoComplete="off" autoCapitalize="off"
        placeholder="skulk-catalog:…" onChange={(event) => setCode(event.target.value)} />
      {code.trim() && !invitation ? <Warning role="status">{t('plugins.catalog.invitationInvalid', 'This is not an invitation code this dashboard can read, or it has expired.')}</Warning> : null}
    </> : <Fields>
      <Label htmlFor="catalog-address">{t('plugins.catalog.address', 'Catalog address')}</Label>
      <Input id="catalog-address" value={address} placeholder="https://catalog.example/" onChange={(event) => setAddress(event.target.value)} />
      <Label htmlFor="catalog-publisher">{t('plugins.catalog.publisherName', 'Publisher')}</Label>
      <Input id="catalog-publisher" value={publisher} placeholder="foxlight" onChange={(event) => setPublisher(event.target.value)} />
      <Label htmlFor="catalog-key">{t('plugins.catalog.publisherKey', 'Publisher key (Ed25519, hexadecimal)')}</Label>
      <Input id="catalog-key" value={publicKey} spellCheck={false} onChange={(event) => setPublicKey(event.target.value)} />
      <Label htmlFor="catalog-days">{t('plugins.catalog.trustDays', 'Trust this key for (days)')}</Label>
      <Input id="catalog-days" value={days} inputMode="numeric" onChange={(event) => setDays(event.target.value)} />
    </Fields>}
    {chosen ? <Facts aria-label={t('plugins.catalog.connectFacts', 'What this host will trust')}>
      <SectionLabel>{t('plugins.catalog.willTrust', 'This host will read')}</SectionLabel>
      <dl>
        <dt>{t('plugins.catalog.addressShort', 'Catalog')}</dt><dd>{new URL(chosen.baseUrl).host}</dd>
        <dt>{t('plugins.catalog.publisherShort', 'Signed by')}</dt><dd>{chosen.publisher} · {chosen.publicKey.slice(0, 8)}…{chosen.publicKey.slice(-8)}</dd>
        <dt>{t('plugins.catalog.trustedUntil', 'Trusted until')}</dt><dd>{new Date(chosen.trustExpiresAt * 1000).toLocaleDateString()}</dd>
        {chosen.token ? <><dt>{t('plugins.catalog.access', 'Access')}</dt><dd>{t('plugins.catalog.accessIncluded', 'An access credential is included and stored on this host only.')}</dd></> : null}
      </dl>
    </Facts> : null}
    <Note>{t('plugins.catalog.connectNote', 'The code only tells this host where to look and whose signature to accept. Every release is still verified against that key before anything installs.')}</Note>
    {notice ? <Warning role="alert">{notice}</Warning> : null}
    <Actions>
      <Button variant="primary" disabled={!chosen || busy} onClick={() => void connect()}>{t('plugins.catalog.connect', 'Connect')}</Button>
      {onCancel ? <Button variant="ghost" onClick={onCancel}>{t('common.cancel', 'Cancel')}</Button> : null}
      <Button variant="ghost" onClick={() => { setManual(!manual); setNotice(''); }}>{manual ? t('plugins.catalog.useCode', 'Use an invitation code instead') : t('plugins.catalog.enterDetails', 'Enter the catalog details instead')}</Button>
    </Actions>
  </Panel>;
}

const Panel = styled(Surface)`
  max-width: 640px; padding: 28px; display: flex; flex-direction: column; gap: 10px;
  h2 { font-size: 22px; margin: 0; letter-spacing: -.01em; color: ${({ theme }) => theme.colors.text}; }
`;
const Lead = styled.p`margin: 0 0 8px; color: ${({ theme }) => theme.colors.textSecondary}; font-size: 14.5px; line-height: 1.5;`;
const Label = styled.label`font-size: 13px; font-weight: 600; color: ${({ theme }) => theme.colors.text}; margin-top: 6px;`;
const inputStyle = `width: 100%; box-sizing: border-box; border-radius: 10px; padding: 10px 12px; font-size: 14px;`;
const CodeInput = styled.textarea`
  ${inputStyle} resize: vertical; font-family: ${({ theme }) => theme.fonts.mono}; font-size: 13px;
  background: ${({ theme }) => theme.colors.surfaceHover}; color: ${({ theme }) => theme.colors.text}; border: 1px solid ${({ theme }) => theme.colors.borderControl};
`;
const Input = styled.input`
  ${inputStyle} background: ${({ theme }) => theme.colors.surfaceHover}; color: ${({ theme }) => theme.colors.text}; border: 1px solid ${({ theme }) => theme.colors.borderControl};
`;
const Fields = styled.div`display: flex; flex-direction: column; gap: 6px;`;
const Facts = styled.section`
  margin-top: 6px; padding: 14px 16px; border-radius: 12px; border: 1px solid ${({ theme }) => theme.colors.border}; background: ${({ theme }) => theme.colors.surfaceHover};
  dl { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 6px 16px; margin: 10px 0 0; font-size: 13.5px; }
  dt { color: ${({ theme }) => theme.colors.textSecondary}; }
  dd { margin: 0; overflow-wrap: anywhere; color: ${({ theme }) => theme.colors.text}; }
`;
const Note = styled.p`margin: 4px 0 0; font-size: 12.5px; line-height: 1.5; color: ${({ theme }) => theme.colors.metadataText};`;
const Warning = styled.p`margin: 4px 0 0; font-size: 13px; line-height: 1.5; color: ${({ theme }) => theme.colors.warningOnSurface};`;
const Actions = styled.div`display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;`;
