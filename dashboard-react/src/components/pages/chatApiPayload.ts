// Copyright 2026 Foxlight Foundation

import type { ChatMessage } from '../../types/chat';

/** One OpenAI-compatible chat message sent by the dashboard. */
export type ApiMessagePayload = Record<string, unknown>;

/**
 * Convert persisted dashboard messages into OpenAI-compatible API messages.
 *
 * Image attachments are emitted as `image_url` content parts before the text
 * part. The data URLs are retained verbatim so the API receives the exact
 * bytes selected by the user.
 */
export function buildApiMessages(messages: ChatMessage[]): ApiMessagePayload[] {
  return messages.map((message) => {
    if (message.attachments?.some((attachment) => attachment.type.startsWith('image/') && attachment.preview)) {
      const parts: Array<{ type: string; text?: string; image_url?: { url: string } }> = [];
      for (const attachment of message.attachments) {
        if (attachment.type.startsWith('image/') && attachment.preview) {
          parts.push({ type: 'image_url', image_url: { url: attachment.preview } });
        }
      }
      if (message.content) {
        parts.push({ type: 'text', text: message.content });
      }
      return { role: message.role, content: parts };
    }
    return { role: message.role, content: message.content };
  });
}
