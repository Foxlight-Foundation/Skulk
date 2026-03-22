/**
 * SystemWarningsBanner
 *
 * Shows dismissible warnings for:
 * - macOS version mismatches across cluster nodes
 * - Thunderbolt 5 nodes without RDMA enabled
 *
 * Extracted from the inline logic in +page.svelte.
 */
import React, { useState } from 'react';
import styled from 'styled-components';
import { useTranslate } from '@tolgee/react';
import { useTopologyStore } from '../../stores/topologyStore';

// ─── Styled components ────────────────────────────────────────────────────────

const BannerStack = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 4px 8px 0;
`;

const Warning = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 8px 12px;
  border-radius: ${({ theme }) => theme.radius.md};
  background: oklch(0.2 0.06 75 / 0.8);
  border: 1px solid oklch(0.45 0.12 75 / 0.5);
  font-size: 11px;
  letter-spacing: 0.04em;
`;

const WarningIcon = styled.span`
  flex-shrink: 0;
  color: oklch(0.75 0.18 75);
  margin-top: 1px;
`;

const WarningBody = styled.div`
  flex: 1;
`;

const WarningTitle = styled.div`
  font-weight: 700;
  color: oklch(0.85 0.12 75);
  text-transform: uppercase;
  letter-spacing: 0.07em;
  margin-bottom: 2px;
`;

const WarningText = styled.div`
  color: oklch(0.75 0.06 75);
  line-height: 1.4;
`;

const DismissButton = styled.button`
  flex-shrink: 0;
  background: none;
  border: none;
  color: oklch(0.6 0.06 75);
  cursor: pointer;
  font-size: 12px;
  padding: 0 4px;
  line-height: 1;
  &:hover { color: oklch(0.85 0.12 75); }
`;

// ─── Component ─────────────────────────────────────────────────────────────────

export const SystemWarningsBanner: React.FC = () => {
  const { t } = useTranslate();
  const [macOsDismissed, setMacOsDismissed] = useState(false);
  const [tb5Dismissed, setTb5Dismissed] = useState(false);

  const getMacosVersionMismatch = useTopologyStore((s) => s.getMacosVersionMismatch);
  const getHasTb5WithoutRdma = useTopologyStore((s) => s.getHasTb5WithoutRdma);

  const macOsMismatch = getMacosVersionMismatch();
  const hasTb5 = getHasTb5WithoutRdma();

  const showMacOs = !macOsDismissed && macOsMismatch !== null;
  const showTb5 = !tb5Dismissed && hasTb5;

  if (!showMacOs && !showTb5) return null;

  return (
    <BannerStack>
      {showMacOs && (
        <Warning role="alert">
          <WarningIcon>⚠</WarningIcon>
          <WarningBody>
            <WarningTitle>{t('warnings.macos_mismatch_title')}</WarningTitle>
            <WarningText>
              {t('warnings.macos_mismatch_body')}{' '}
              {macOsMismatch!.map((n) => `${n.friendlyName} (${n.version})`).join(', ')}
            </WarningText>
          </WarningBody>
          <DismissButton
            onClick={() => setMacOsDismissed(true)}
            aria-label={t('warnings.dismiss')}
            title={t('warnings.dismiss')}
          >
            ✕
          </DismissButton>
        </Warning>
      )}

      {showTb5 && (
        <Warning role="alert">
          <WarningIcon>⚠</WarningIcon>
          <WarningBody>
            <WarningTitle>{t('warnings.tb5_no_rdma_title')}</WarningTitle>
            <WarningText>{t('warnings.tb5_no_rdma_body')}</WarningText>
          </WarningBody>
          <DismissButton
            onClick={() => setTb5Dismissed(true)}
            aria-label={t('warnings.dismiss')}
            title={t('warnings.dismiss')}
          >
            ✕
          </DismissButton>
        </Warning>
      )}
    </BannerStack>
  );
};
