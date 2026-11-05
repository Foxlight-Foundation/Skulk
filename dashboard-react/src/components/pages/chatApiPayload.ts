// Copyright 2026 Foxlight Foundation

import type { ChatMessage } from '../../types/chat';
import { MAX_CHAT_DOCUMENTS, chatDocumentKind } from '../../store/chatDocuments';

/** One OpenAI-compatible chat message sent by the dashboard. */
export type ApiMessagePayload = Record<string, unknown>;

/**
 * Convert persisted dashboard messages into OpenAI-compatible API messages.
 *
 * Image attachments are emitted as `image_url` content parts before the text
 * part. The data URLs are retained verbatim so the API receives the exact
 * bytes selected by the user.
 */
export function buildApiMessages(
  messages: ChatMessage[], documents: { text: boolean; pdf: boolean } = { text: false, pdf: false },
): ApiMessagePayload[] {
  let documentCount = 0;
  return messages.flatMap((message): ApiMessagePayload[] => {
    if (message.role === 'assistant' && message.continuationMessages) return message.continuationMessages;
    if (message.attachments?.length) {
      const parts: Array<Record<string, unknown>> = [];
      for (const attachment of message.attachments) {
        if (attachment.type.startsWith('image/') && attachment.preview) {
          parts.push({ type: 'image_url', image_url: { url: attachment.preview } });
        } else {
          const kind = chatDocumentKind(attachment.name);
          if (!kind || !documents[kind]) throw new Error('The selected model does not support a document in this conversation. Choose a compatible model or remove its message.');
          if (!attachment.fileData) throw new Error(`Document ${attachment.name} is unavailable. Reattach it before sending.`);
          if (++documentCount > MAX_CHAT_DOCUMENTS) throw new Error('A chat request accepts at most eight documents, including history.');
          parts.push({ type: 'file', file: { filename: attachment.name, file_data: attachment.fileData } });
        }
      }
      if (message.content) {
        parts.push({ type: 'text', text: message.content });
      }
      return [{ role: message.role, content: parts }];
    }
    return [{ role: message.role, content: message.content }];
  });
}
