import { useRef } from 'react';
import styled from 'styled-components';
import { MdAutoAwesome } from 'react-icons/md';
import { FiArrowUp } from 'react-icons/fi';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { Button } from '../common/Button';

const Prompt = styled.form`
  display: flex; align-items: center; gap: 12px;
  width: min(640px, calc(100% - 32px)); min-height: 48px; padding: 8px 12px 8px 16px;
  border: 1px solid ${({ theme }) => theme.colors.borderLive}; border-radius: 14px;
  background: ${({ theme }) => theme.colors.surfaceElevated};
  box-shadow: ${({ theme }) => theme.colors.shadowCard};
  color: ${({ theme }) => theme.colors.text};
  > svg { color: ${({ theme }) => theme.colors.live}; flex-shrink: 0; }
  &:focus-within { box-shadow: ${({ theme }) => theme.colors.focusRing}; }
`;
const Input = styled.input`
  all: unset; flex: 1; min-width: 0; font-size: 16px;
  &::placeholder { color: ${({ theme }) => theme.colors.subtleText}; }
  /* The surrounding composer owns the designed focus indication. */
  &:focus-visible { outline: none; }
`;

/** Controlled cluster composer sharing the Steward conversation's draft. */
export interface StewardPromptProps {
  draft: string;
  onDraftChange: (draft: string) => void;
  onSubmit: () => void;
  /** Prevent submission while Steward is unavailable or already answering. */
  disabled?: boolean;
}

/** Submit only on Send or Enter; focusing and typing never open the drawer. */
export function StewardPrompt({ draft, onDraftChange, onSubmit, disabled }: StewardPromptProps) {
  const { t } = useSkulkTranslation();
  const input = useRef<HTMLInputElement>(null);
  const canSend = !disabled && !!draft.trim();
  return <Prompt onSubmit={event => {
    event.preventDefault();
    if (!canSend) return;
    // Return to the editable prompt when the drawer closes, even after clicking Send.
    input.current?.focus();
    onSubmit();
  }}>
    <MdAutoAwesome aria-hidden="true" />
    <Input ref={input} value={draft} onChange={event => onDraftChange(event.target.value)}
      aria-label={t('steward.prompt', 'Ask Skulk about the cluster…')}
      placeholder={t('steward.prompt', 'Ask Skulk about the cluster…')}
      onKeyDown={event => {
        // Enter can confirm an IME candidate without expressing intent to send.
        if (event.key === 'Enter' && (event.nativeEvent.isComposing || event.keyCode === 229)) event.preventDefault();
      }} />
    <Button type="submit" variant="solid" icon disabled={!canSend}
      aria-label={t('chat.form.sendMessage', 'Send message')} title={t('chat.form.sendMessage', 'Send message')}>
      <FiArrowUp aria-hidden="true" />
    </Button>
  </Prompt>;
}
