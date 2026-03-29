export interface ChatUploadedFile {
  id: string;
  name: string;
  type: string;
  size: number;
  textContent?: string;
  preview?: string;
}

export interface MessageAttachment {
  id: string;
  name: string;
  type: string;
  size?: number;
  preview?: string;
  textContent?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  attachments?: MessageAttachment[];
  thinkingContent?: string;
  generatedImages?: string[];
  ttftMs?: number;
  tps?: number;
  tokens?: Array<{
    token: string;
    probability: number;
    logprob: number;
    topLogprobs?: Array<{ token: string; logprob: number }>;
  }>;
}

export interface Conversation {
  id: string;
  name: string;
  modelId: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
  summary?: string;
  originNodeId?: string;
}

export interface ChatModelInfo {
  id: string;
  name: string;
  base_model: string;
  storage_size_megabytes: number;
  capabilities: string[];
  family: string;
  quantization: string;
}

export function getFileCategory(type: string, name: string): 'image' | 'text' | 'pdf' | 'audio' | 'other' {
  if (type.startsWith('image/')) return 'image';
  if (type.startsWith('text/') || name.endsWith('.txt') || name.endsWith('.md') || name.endsWith('.json') || name.endsWith('.csv')) return 'text';
  if (type === 'application/pdf' || name.endsWith('.pdf')) return 'pdf';
  if (type.startsWith('audio/')) return 'audio';
  return 'other';
}

export function getFileIcon(type: string, name: string): string {
  switch (getFileCategory(type, name)) {
    case 'image': return '🖼';
    case 'text': return '📄';
    case 'pdf': return '📑';
    case 'audio': return '🎵';
    default: return '📎';
  }
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

export function truncateFileName(name: string, maxLen = 20): string {
  if (name.length <= maxLen) return name;
  const dotIdx = name.lastIndexOf('.');
  const ext = dotIdx > 0 ? name.slice(dotIdx) : '';
  const available = maxLen - ext.length - 3;
  return name.slice(0, Math.max(1, available)) + '...' + ext;
}
