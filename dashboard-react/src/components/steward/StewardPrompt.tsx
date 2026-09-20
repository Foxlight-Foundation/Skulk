import styled from 'styled-components';
import { MdAutoAwesome } from 'react-icons/md';
import { useSkulkTranslation } from '../../i18n/tolgee';

const Prompt = styled.button`
  display: flex; align-items: center; gap: 12px;
  width: min(640px, calc(100% - 32px)); min-height: 48px; padding: 8px 16px;
  border: 1px solid ${({ theme }) => theme.colors.borderLive}; border-radius: 14px;
  background: ${({ theme }) => theme.colors.surfaceElevated};
  box-shadow: ${({ theme }) => theme.colors.shadowCard};
  color: ${({ theme }) => theme.colors.subtleText}; text-align: left; font-size: 15px;
  svg { color: ${({ theme }) => theme.colors.live}; flex-shrink: 0; }
  span { flex: 1; }
  kbd { font: 11px ${({ theme }) => theme.fonts.mono}; padding: 2px 6px; border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 4px; }
`;

/** Cluster entry point; opening the conversation never submits a request. */
export function StewardPrompt({ onOpen }: { onOpen: () => void }) {
  const { t } = useSkulkTranslation();
  const shortcut = navigator.platform.includes('Mac') ? '⌘K' : 'Ctrl K';
  return <Prompt type="button" onClick={onOpen} onFocus={event => { if (event.currentTarget.dataset.skulkFocusReturn !== 'true') onOpen(); }}>
    <MdAutoAwesome aria-hidden="true" /><span>{t('steward.prompt', 'Ask Skulk about the cluster…')}</span><kbd>{shortcut}</kbd>
  </Prompt>;
}
