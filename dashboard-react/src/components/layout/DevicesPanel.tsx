import { useState } from 'react';
import styled from 'styled-components';
import { FiSearch, FiSmartphone } from 'react-icons/fi';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { useGetOperatorDevicesQuery, useRevokeOperatorDeviceMutation, type OperatorDevice } from '../../store/endpoints/devices';
import { pairingInvitationQueryErrorDetail } from '../../store/endpoints/pairing';
import { operatorSession } from '../../auth/operatorSession';
import { Button } from '../common/Button';
import { Field } from '../common/Field';
import { StatusPill, SectionLabel } from '../common/Surfaces';
import { PairingSettings } from './PairingSettings';

const Content = styled.div`
  overflow-y: auto; min-height: 0; flex: 1; container-type: inline-size;
`;
const Columns = styled.div`
  display: grid; grid-template-columns: minmax(0, 1fr) 280px; min-height: 100%;
  @container (max-width: 620px) { grid-template-columns: minmax(0, 1fr); min-height: 0; }
`;
const Column = styled.section`
  min-width: 0; padding: 20px; display: flex; flex-direction: column; gap: 16px;
  &:first-child { border-right: 1px solid ${({ theme }) => theme.colors.border}; }
  &:last-child { background: ${({ theme }) => theme.colors.surface}; }
  @container (max-width: 620px) { padding: 16px; &:first-child { border-right: 0; border-bottom: 1px solid ${({ theme }) => theme.colors.border}; } }
`;
const Row = styled.article`
  padding: 11px 14px;
  & + & { border-top: 1px solid ${({ theme }) => theme.colors.borderLight}; }
  display: grid; grid-template-columns: 32px minmax(0, 1fr) auto; gap: 12px; align-items: center;
  > svg { flex-shrink: 0; width: 32px; height: 32px; padding: 6px; border-radius: 8px; background: ${({ theme }) => theme.colors.selected}; }
`;
const Detail = styled.div`min-width: 0; flex: 1; display: flex; flex-direction: column; gap: 4px; overflow-wrap: anywhere;
  strong { font-size: 14px; font-weight: 600; }`;
const Meta = styled.p`margin: 0; font: 11.5px ${({ theme }) => theme.fonts.mono}; color: ${({ theme }) => theme.colors.metadataText}; line-height: 1.5;`;
const DeviceList = styled.div`
  border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 12px;
  background: ${({ theme }) => theme.colors.surface}; overflow: hidden;
`;
const DeviceHeading = styled.div`
  display: flex; align-items: center; flex-wrap: wrap; gap: 6px;
  > span { font-size: 10px; padding: 1px 6px; }
`;
const DeviceMetadata = styled(Meta)`overflow: hidden; text-overflow: ellipsis; white-space: nowrap;`;
const SearchField = styled(Field)`background: ${({ theme }) => theme.colors.surface};`;
const Actions = styled.div`display: flex; gap: 8px; flex-wrap: wrap;`;

/** Safe device inventory row with explicit confirmation before immediate revocation. */
export function DeviceRow({ device, busy, onRevoke }: { device: OperatorDevice; busy: boolean; onRevoke: () => void }) {
  const { t } = useSkulkTranslation();
  const [confirming, setConfirming] = useState(false);
  return <Row>
    <FiSmartphone size={24} aria-hidden="true" />
    <Detail>
      <DeviceHeading><strong>{device.name}</strong>
      <StatusPill tone={device.state === 'revoked' ? 'danger' : 'neutral'}>{device.state === 'revoked' ? t('devices.revoked', 'Revoked') : t('devices.paired', 'Paired')}{device.current ? ` · ${t('devices.thisDevice', 'This device')}` : ''}</StatusPill></DeviceHeading>
      <DeviceMetadata title={device.deviceId}>{device.deviceId}</DeviceMetadata>
      <Meta>{t('devices.pairedAt', 'Paired {date}', { date: new Date(device.pairedAt).toLocaleDateString() })}</Meta>
      {device.state === 'active' && confirming && <>
        <Meta>{device.current ? t('devices.confirmCurrent', 'Revoking this device ends your current session.') : t('devices.confirmRevoke', 'This device will need a new invitation to reconnect.')}</Meta>
        <Actions><Button variant="danger" size="sm" loading={busy} onClick={onRevoke}>{t('devices.confirm', 'Revoke access')}</Button><Button size="sm" disabled={busy} onClick={() => setConfirming(false)}>{t('common.cancel', 'Cancel')}</Button></Actions>
      </>}
    </Detail>
    {device.state === 'active' && !confirming && <Button size="sm" variant="outline" disabled={busy} onClick={() => setConfirming(true)}>{t('devices.revoke', 'Revoke')}</Button>}
  </Row>;
}

/** Device and invitation operations are immediate, independent of the Settings draft. */
export function DevicesPanel() {
  const { t } = useSkulkTranslation();
  const [search, setSearch] = useState('');
  const [invitationHost, setInvitationHost] = useState<HTMLDivElement | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const devices = useGetOperatorDevicesQuery(undefined, { pollingInterval: 15000 });
  const [revoke, revocation] = useRevokeOperatorDeviceMutation();
  const onRevoke = async (device: OperatorDevice) => {
    setFailure(null);
    try {
      await revoke(device.deviceId).unwrap();
      if (device.current) operatorSession.disconnect();
    } catch (error) { setFailure(pairingInvitationQueryErrorDetail(error) ?? t('devices.revokeFailed', 'Could not revoke access. Refresh the list before retrying.')); }
  };
  const filtered = devices.data?.devices.filter(device => `${device.name} ${device.deviceId}`.toLocaleLowerCase().includes(search.toLocaleLowerCase())) ?? [];
  return <Content><Columns>
    <Column aria-label={t('devices.pairedDevices', 'Paired devices')}>
      <SearchField icon={<FiSearch size={14} />} aria-label={t('devices.search', 'Search devices')} placeholder={t('devices.search', 'Search devices')} value={search} onChange={event => setSearch(event.target.value)} />
      <Meta style={{ fontFamily: 'inherit', fontSize: 12 }}>{t('devices.stateHint', 'Pairing grants access; it does not indicate whether a device is online.')}</Meta>
      <SectionLabel>{t('devices.pairedDevices', 'Paired devices')}</SectionLabel>
      {devices.isLoading && <Meta role="status">{t('devices.loading', 'Loading devices…')}</Meta>}
      {devices.isError && <><Meta role="alert">{pairingInvitationQueryErrorDetail(devices.error) ?? t('devices.unavailable', 'Device inventory is unavailable.')}</Meta><Button onClick={() => devices.refetch()}>{t('common.retry', 'Retry')}</Button></>}
      {failure && <Meta role="alert">{failure}</Meta>}
      {!devices.isLoading && !devices.isError && filtered.length === 0 && <Meta>{t('devices.empty', 'No devices found.')}</Meta>}
      {filtered.length > 0 && <DeviceList>{filtered.map(device => <DeviceRow key={device.deviceId} device={device} busy={revocation.isLoading} onRevoke={() => void onRevoke(device)} />)}</DeviceList>}
      <div ref={setInvitationHost} />
    </Column>
    <Column><PairingSettings invitationHost={invitationHost} /></Column>
  </Columns></Content>;
}
