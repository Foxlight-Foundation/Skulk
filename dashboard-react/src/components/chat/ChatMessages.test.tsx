import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { darkTheme } from '../../theme/theme';
import type { ChatMessage } from '../../types/chat';
import { ChatMessages } from './ChatMessages';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}));

let root: Root | null = null;
let container: HTMLDivElement | null = null;

async function renderChatMessages(messages: ChatMessage[]): Promise<void> {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(
      <ThemeProvider theme={darkTheme}>
        <ChatMessages
          messages={messages}
          onDelete={vi.fn()}
          onRegenerate={vi.fn()}
        />
      </ThemeProvider>,
    );
  });
}

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  root = null;
  container = null;
});

describe('ChatMessages', () => {
  it('exposes assistant response content without message chrome', async () => {
    await renderChatMessages([
      {
        id: 'assistant-1',
        role: 'assistant',
        content: 'orange circle | GEM86Z',
        timestamp: 0,
        ttftMs: 1234,
        tps: 56.7,
      },
    ]);

    const fullMessageText = container
      ?.querySelector('[aria-label="Assistant message"]')
      ?.textContent;
    const responseText = container
      ?.querySelector('[data-testid="assistant-response-content"]')
      ?.textContent;

    expect(fullMessageText).toContain('TTFT');
    expect(fullMessageText).toContain('Copy');
    expect(fullMessageText).toContain('Regenerate');
    expect(fullMessageText).toContain('Delete');
    expect(responseText?.trim()).toBe('orange circle | GEM86Z');
  });
});
