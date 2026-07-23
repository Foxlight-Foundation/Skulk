// Copyright 2026 Foxlight Foundation

import { describe, expect, it } from 'vitest';

import type { ChatMessage } from '../../types/chat';
import { buildApiMessages } from './chatApiPayload';

describe('buildApiMessages', () => {
  it('preserves uploaded image bytes as an image_url part before prompt text', () => {
    const imageDataUrl = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB';
    const messages: ChatMessage[] = [{
      id: 'user-1',
      role: 'user',
      content: 'Read the hidden code.',
      timestamp: 1,
      attachments: [{
        id: 'fixture-1',
        name: 'qualification.png',
        type: 'image/png',
        size: 24,
        preview: imageDataUrl,
      }],
    }];

    expect(buildApiMessages(messages)).toEqual([{
      role: 'user',
      content: [
        { type: 'image_url', image_url: { url: imageDataUrl } },
        { type: 'text', text: 'Read the hidden code.' },
      ],
    }]);
  });

  it('keeps text-only history in the scalar content form', () => {
    const messages: ChatMessage[] = [{
      id: 'assistant-1',
      role: 'assistant',
      content: 'Ready.',
      timestamp: 2,
    }];

    expect(buildApiMessages(messages)).toEqual([{
      role: 'assistant',
      content: 'Ready.',
    }]);
  });
});
