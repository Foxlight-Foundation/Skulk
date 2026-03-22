/**
 * ChatMessages
 *
 * Scrollable message list with auto-scroll on new messages and a
 * "scroll to bottom" button when the user has scrolled up.
 * Ported from ChatMessages.svelte.
 */
import React, { useRef, useEffect, useState, useCallback } from 'react';
import styled, { keyframes } from 'styled-components';
import { useTranslate } from '@tolgee/react';
import { MessageBubble } from './MessageBubble';
import { useChatStore } from '../../stores/chatStore';

// ─── Styled components ────────────────────────────────────────────────────────

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: none; }
`;

const Scroll = styled.div`
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  padding: 16px;
  gap: 8px;
  scroll-behavior: smooth;
`;

const MessageRow = styled.div`
  animation: ${fadeIn} 200ms ease;
`;

const Empty = styled.div`
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: ${({ theme }) => theme.colors.lightGray};
  font-size: 12px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  opacity: 0.5;
`;

const ScrollButton = styled.button`
  position: absolute;
  bottom: 80px;
  right: 20px;
  background: ${({ theme }) => theme.colors.darkGray};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radius.full};
  color: ${({ theme }) => theme.colors.yellow};
  width: 36px;
  height: 36px;
  font-size: 16px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px oklch(0 0 0 / 0.4);
  transition: background ${({ theme }) => theme.transitions.fast};
  z-index: 10;
  &:hover { background: ${({ theme }) => theme.colors.mediumGray}; }
`;

const Container = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  position: relative;
  overflow: hidden;
`;

// ─── Component ─────────────────────────────────────────────────────────────────

const SCROLL_THRESHOLD = 100;

export const ChatMessages: React.FC = () => {
  const { t } = useTranslate();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const lastCountRef = useRef(0);

  const messages = useChatStore((s) => s.getMessages());
  const isStreaming = useChatStore((s) => s.isStreaming);

  const isNearBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < SCROLL_THRESHOLD;
  }, []);

  const scrollToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, []);

  const updateScrollButton = useCallback(() => {
    setShowScrollButton(!isNearBottom());
  }, [isNearBottom]);

  // Attach scroll listener
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.addEventListener('scroll', updateScrollButton, { passive: true });
    return () => el.removeEventListener('scroll', updateScrollButton);
  }, [updateScrollButton]);

  // Auto-scroll when a new message appears
  useEffect(() => {
    const count = messages.length;
    if (count > lastCountRef.current) {
      requestAnimationFrame(scrollToBottom);
    }
    lastCountRef.current = count;
  }, [messages.length, scrollToBottom]);

  // Keep up with streaming updates if near bottom
  useEffect(() => {
    if (isStreaming && isNearBottom()) {
      requestAnimationFrame(scrollToBottom);
    }
  }, [isStreaming, messages, isNearBottom, scrollToBottom]);

  return (
    <Container>
      {messages.length === 0 ? (
        <Empty>{t('topology.waiting')}</Empty>
      ) : (
        <Scroll ref={scrollRef}>
          {messages.map((msg) => (
            <MessageRow key={msg.id}>
              <MessageBubble message={msg} />
            </MessageRow>
          ))}
        </Scroll>
      )}

      {showScrollButton && (
        <ScrollButton
          onClick={scrollToBottom}
          aria-label="Scroll to bottom"
          title="Scroll to bottom"
        >
          ↓
        </ScrollButton>
      )}
    </Container>
  );
};
