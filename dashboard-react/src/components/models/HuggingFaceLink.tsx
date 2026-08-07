import styled from 'styled-components';
import { FiExternalLink } from 'react-icons/fi';
import { useSkulkTranslation } from '../../i18n/tolgee';

/**
 * Quiet external-link affordance opening a repository on Hugging Face in a
 * new tab. Every discovery row that maps to exactly one repository renders
 * one, so provenance is always one click away instead of hiding inside the
 * info tooltip.
 */
const LinkButton = styled.a`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  flex-shrink: 0;
  border-radius: ${({ theme }) => theme.radii.sm};
  color: ${({ theme }) => theme.colors.textMuted};
  transition: color 0.15s, background 0.15s;

  &:hover {
    color: ${({ theme }) => theme.colors.gold};
    background: ${({ theme }) => theme.colors.goldBg};
  }
`;

/** Render the Hugging Face link icon for one repository id. */
export function HuggingFaceLink({ repoId, size = 14 }: { repoId: string; size?: number }) {
  const { t } = useSkulkTranslation();
  if (!repoId.includes('/')) return null;
  return (
    <LinkButton
      href={`https://huggingface.co/${repoId}`}
      target="_blank"
      rel="noopener noreferrer"
      title={t('common.openOnHuggingFace', 'Open on HuggingFace')}
      aria-label={t('common.openOnHuggingFace', 'Open on HuggingFace')}
      onClick={(e) => e.stopPropagation()}
    >
      <FiExternalLink size={size} />
    </LinkButton>
  );
}
