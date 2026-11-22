import styled from 'styled-components';
import { useSkulkTranslation } from '../../i18n/tolgee';
import type { NativeCatalogBinding } from '../../store/endpoints/plugins';
import { SectionLabel } from '../common/Surfaces';

/** Authenticated facts shared by catalog, recovery and direct-source consent. */
export interface NativeRuntimeConsentProps { binding?: NativeCatalogBinding | null }

/** Show processing locations before consent, with complete contract identities available. */
export function NativeRuntimeConsent({ binding }: NativeRuntimeConsentProps) {
  const { t } = useSkulkTranslation();
  if (!binding) return null;
  const labels = {
    local: t('plugins.native.local', 'On this host'),
    private_site: t('plugins.native.privateSite', 'Private deployment'),
    external: t('plugins.native.external', 'External services'),
  };
  return <Section aria-label={t('plugins.native.processing', 'Declared processing locations')}>
    <SectionLabel>{t('plugins.native.processing', 'Declared processing locations')}</SectionLabel>
    <ul>{binding.processing_domains.map((domain) => <li key={domain}>{labels[domain]}</li>)}</ul>
    {binding.processing_domains.includes('external') ? <p>{t('plugins.native.externalContent', 'Requests and their content may be sent to external services.')}</p> : null}
    <p>{t('plugins.native.configureAccess', 'Configure account access and inference permissions separately.')}</p>
    <Details>
      <summary>{t('plugins.native.identity', 'Contract identity')}</summary>
      <Facts>
        <dt>{t('plugins.native.manifest', 'Declaration')}</dt><dd><code>{binding.manifest_revision}</code></dd>
        <dt>{t('plugins.native.protocols', 'Runtime / release')}</dt><dd>{binding.runtime_protocol} / {binding.release_protocol}</dd>
        <dt>{t('plugins.native.kernelVersion', 'Kernel version')}</dt><dd>{binding.kernel.version}</dd>
        <dt>{t('plugins.native.kernelSource', 'Kernel source')}</dt><dd><code>{binding.kernel.source_commit}</code></dd>
        {Object.entries(binding.kernel).filter(([key]) => key !== 'version' && key !== 'source_commit').map(([key, value]) => <Fact key={key}>
          <dt>ABI {key}</dt><dd><code>{value}</code></dd>
        </Fact>)}
      </Facts>
    </Details>
  </Section>;
}

const Section = styled.section`
  display: flex; flex-direction: column; gap: 8px; padding: 16px;
  border: 1px solid ${({ theme }) => theme.colors.borderStrong}; border-radius: 12px;
  color: ${({ theme }) => theme.colors.text};
  p, ul { margin: 0; font-size: 14px; line-height: 1.5; }
  ul { padding-left: 20px; }
`;
const Details = styled.details`
  font-size: 13px;
  summary { cursor: pointer; font-weight: 600; padding: 4px 0; }
`;
const Facts = styled.dl`
  display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 8px 16px; margin: 12px 0 0;
  dt { color: ${({ theme }) => theme.colors.textSecondary}; }
  dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }
  code { font: 12px ${({ theme }) => theme.fonts.mono}; }
  @media (max-width: 480px) { grid-template-columns: minmax(0, 1fr); gap: 4px; dd { margin-bottom: 8px; } }
`;
const Fact = styled.div`display: contents;`;
