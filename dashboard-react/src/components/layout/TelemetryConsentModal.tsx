import { useCallback, useEffect, useState } from 'react';
import styled, { keyframes } from 'styled-components';
import { useConfig, type FullConfig, type TelemetryConfig } from '../../hooks/useConfig';
import { useSkulkTranslation } from '../../i18n/tolgee';

/**
 * First-run consent modal for opt-in field telemetry.
 *
 * Shown once per browser (a localStorage seen-marker enforces no-nag) and
 * only while the fleet consent state is still `unasked`. Dismissing without
 * choosing leaves the fleet Setting `unasked` (nothing is collected) but
 * never re-prompts this browser; another operator's browser still gets its
 * one ask. The only path to any collection is an explicit toggle here or in
 * Settings, where both switches remain permanently available.
 */

/** Browser-local no-nag marker (never synced; consent itself lives in skulk.yaml). */
export const TELEMETRY_CONSENT_SEEN_KEY = 'skulk-telemetry-consent-seen';

const fadeIn = keyframes`
  from { opacity: 0; }
  to { opacity: 1; }
`;

const riseIn = keyframes`
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: none; }
`;

const Backdrop = styled.div`
  position: fixed;
  inset: 0;
  z-index: 60;
  background: ${({ theme }) => theme.colors.overlay};
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  animation: ${fadeIn} 160ms ease;
`;

const Card = styled.div`
  width: min(560px, calc(100vw - 32px));
  max-height: calc(100vh - 64px);
  overflow-y: auto;
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: 14px;
  padding: 24px;
  animation: ${riseIn} 200ms ease;
`;

const Title = styled.h2`
  margin: 0 0 6px;
  font-size: 1.05rem;
`;

const Body = styled.p`
  margin: 0 0 14px;
  font-size: 0.86rem;
  line-height: 1.55;
  opacity: 0.85;
`;

const FactList = styled.ul`
  margin: 0 0 14px;
  padding-left: 18px;
  font-size: 0.82rem;
  line-height: 1.5;
  opacity: 0.8;
`;

const ToggleRow = styled.label`
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 12px;
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: 10px;
  margin-bottom: 10px;
  cursor: pointer;
  font-size: 0.85rem;

  input {
    margin-top: 3px;
  }
`;

const ToggleHint = styled.span`
  display: block;
  font-size: 0.76rem;
  opacity: 0.65;
  margin-top: 2px;
`;

const Actions = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 16px;
`;

const Button = styled.button<{ $primary?: boolean }>`
  padding: 8px 16px;
  border-radius: 8px;
  font-size: 0.84rem;
  cursor: pointer;
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: ${({ $primary, theme }) => ($primary ? theme.colors.gold : 'transparent')};
  color: ${({ $primary, theme }) => ($primary ? theme.colors.textOnAccent : 'inherit')};
  font-weight: ${({ $primary }) => ($primary ? 600 : 400)};
`;

/** UUID even on non-secure origins (LAN HTTP dashboards lack crypto.randomUUID). */
export function generateInstallId(): string {
  // Without Web Crypto entirely, return empty: the API backfills an id
  // server-side whenever consent is enabled without one.
  if (typeof crypto === 'undefined' || typeof crypto.getRandomValues !== 'function') return '';
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function markSeen(): void {
  try {
    localStorage.setItem(TELEMETRY_CONSENT_SEEN_KEY, new Date().toISOString());
  } catch {
    // Private-mode storage failures only cost the no-nag guarantee.
  }
}

function hasSeen(): boolean {
  try {
    return localStorage.getItem(TELEMETRY_CONSENT_SEEN_KEY) != null;
  } catch {
    return false;
  }
}

/** First-run field-telemetry consent dialog; renders nothing once decided or seen. */
export function TelemetryConsentModal() {
  const { t } = useSkulkTranslation();
  const { fullConfig, loading, saving, saveFullConfig } = useConfig(
    t('telemetry.errors.fetchConfigFailed', 'Failed to fetch config'),
  );
  const [telemetryOn, setTelemetryOn] = useState(false);
  const [diagnosticsOn, setDiagnosticsOn] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  // Decided ONCE at mount: re-deriving from the marker after the effect
  // below stamps it would unmount the dialog on the next re-render (a
  // checkbox click) before the operator can save.
  const [eligible] = useState(() => !hasSeen());

  const consent = fullConfig?.telemetry?.consent ?? 'unasked';
  // fullConfig must be LOADED: saving over a failed fetch would PUT a
  // config missing every other section.
  const visible = !loading && fullConfig != null && !dismissed && consent === 'unasked' && eligible;

  // Showing the modal is what stamps the no-nag marker: even a hard refresh
  // mid-decision never asks this browser twice.
  useEffect(() => {
    if (visible) markSeen();
  }, [visible]);

  const close = useCallback(() => {
    markSeen();
    setDismissed(true);
  }, []);

  const save = useCallback(async () => {
    const telemetry: TelemetryConfig = {
      consent: telemetryOn ? 'enabled' : 'disabled',
      diagnostics_consent: diagnosticsOn ? 'enabled' : 'disabled',
      install_id: telemetryOn || diagnosticsOn ? generateInstallId() : '',
      consented_at: new Date().toISOString(),
      consented_version: '',
      ingest_url:
        fullConfig?.telemetry?.ingest_url ??
        'https://skulk-ledger-ingest.thomastupper92618.workers.dev',
    };
    const updated: FullConfig = { ...(fullConfig ?? {}), telemetry };
    const ok = await saveFullConfig(updated);
    // A failed save must NOT dismiss: the browser is already marked seen,
    // so closing here would strand the operator with no modal and no saved
    // consent. Settings remains the fallback either way.
    if (ok !== false) close();
  }, [close, diagnosticsOn, fullConfig, saveFullConfig, telemetryOn]);

  if (!visible) return null;

  return (
    <Backdrop role="dialog" aria-modal="true" aria-labelledby="telemetry-consent-title">
      <Card>
        <Title id="telemetry-consent-title">{t('telemetry.consent.title', 'Help make Skulk better?')}</Title>
        <Body>
          {t(
            'telemetry.consent.body',
            'Share collected data with Foxlight to help make the product better. This is entirely optional, off by default, and never includes your prompts, outputs, or anything that identifies your machines.',
          )}
        </Body>
        <FactList>
          <li>
            {t(
              'telemetry.consent.collected',
              'If enabled, performance telemetry sends: model id, hardware class (like "apple-m4-24gb"), timing, token counts, and failure classes.',
            )}
          </li>
          <li>
            {t(
              'telemetry.consent.retention',
              'Crash diagnostics (a separate choice) are kept privately for 90 days, then deleted. Telemetry appears publicly only as aggregates.',
            )}
          </li>
          <li>
            {t(
              'telemetry.consent.control',
              'You can turn either off at any time in Settings, inspect exactly what would be sent, and delete everything with your install id.',
            )}
          </li>
        </FactList>
        <ToggleRow>
          <input
            type="checkbox"
            checked={telemetryOn}
            onChange={(e) => setTelemetryOn(e.target.checked)}
          />
          <span>
            {t('telemetry.consent.perfToggle', 'Share performance and reliability telemetry')}
            <ToggleHint>
              {t('telemetry.consent.perfHint', 'Anonymous metrics only; shown as public aggregates.')}
            </ToggleHint>
          </span>
        </ToggleRow>
        <ToggleRow>
          <input
            type="checkbox"
            checked={diagnosticsOn}
            onChange={(e) => setDiagnosticsOn(e.target.checked)}
          />
          <span>
            {t('telemetry.consent.diagToggle', 'Share crash diagnostics')}
            <ToggleHint>
              {t(
                'telemetry.consent.diagHint',
                'Scrubbed error reports, kept privately for debugging; never published.',
              )}
            </ToggleHint>
          </span>
        </ToggleRow>
        <Actions>
          <Button onClick={close} disabled={saving}>
            {t('telemetry.consent.notNow', 'Not now')}
          </Button>
          <Button $primary onClick={() => void save()} disabled={saving}>
            {t('telemetry.consent.save', 'Save choices')}
          </Button>
        </Actions>
      </Card>
    </Backdrop>
  );
}
