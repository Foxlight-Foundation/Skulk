import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { ChatForm } from './ChatForm';
import type { ChatSpeechModelOption } from '../../types/chat';

const sttModel: ChatSpeechModelOption = {
  modelId: 'mlx-community/whisper-large-v3-mlx',
  label: 'whisper-large-v3-mlx',
  defaultResponseFormat: 'wav',
  responseFormats: ['wav'],
};

const realtimeSttModel: ChatSpeechModelOption = {
  modelId: 'mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit',
  label: 'Voxtral-Mini-4B-Realtime-2602-4bit',
  defaultResponseFormat: 'wav',
  responseFormats: ['wav'],
  supportsRealtime: true,
};

const ttsModel: ChatSpeechModelOption = {
  modelId: 'mlx-community/Kokoro-82M-bf16',
  label: 'Kokoro-82M-bf16',
  defaultResponseFormat: 'mp3',
  responseFormats: ['mp3', 'wav'],
};

const meta: Meta<typeof ChatForm> = {
  title: 'Chat/ChatForm',
  component: ChatForm,
  parameters: { layout: 'centered' },
  decorators: [(Story) => (
    <div style={{ width: 'min(600px, calc(100vw - 48px))', padding: 24, background: '#000' }}>
      <Story />
    </div>
  )],
};

export default meta;
type Story = StoryObj<typeof ChatForm>;

export const Default: Story = {
  args: {
    onSend: (msg, files) => console.log('Send:', msg, files),
    placeholder: 'Ask anything…',
  },
};

export const WithModelHeader: Story = {
  args: {
    onSend: () => {},
    modelLabel: 'Qwen3-30B-A3B-4bit',
    onOpenModelPicker: () => alert('Open picker'),
    showThinkingToggle: true,
    thinkingEnabled: false,
    onToggleThinking: () => {},
    ttftMs: 245,
    tps: 42.3,
  },
};

export const NoSpeechModels: Story = {
  args: {
    onSend: () => {},
    modelLabel: 'Qwen3-30B-A3B-4bit',
    transcriptionModels: [],
    speechModels: [],
  },
};

export const SttOnly: Story = {
  args: {
    onSend: () => {},
    modelLabel: 'Qwen3-30B-A3B-4bit',
    transcriptionModels: [sttModel],
    selectedTranscriptionModelId: sttModel.modelId,
    onTranscribeAudio: async () => 'recorded transcript',
  },
};

export const RealtimeStt: Story = {
  args: {
    onSend: () => {},
    modelLabel: 'Qwen3-30B-A3B-4bit',
    transcriptionModels: [realtimeSttModel],
    selectedTranscriptionModelId: realtimeSttModel.modelId,
    realtimeTranscriptionAvailable: true,
    onTranscribeAudio: async () => 'batch fallback transcript',
  },
};

export const TtsOnly: Story = {
  args: {
    onSend: () => {},
    canSendMessages: false,
    speechModels: [ttsModel],
    selectedSpeechModelId: ttsModel.modelId,
    selectedVoice: 'af_heart',
    autoSpeakAssistant: true,
    onSpeakText: (text) => console.log('Speak:', text),
    onAutoSpeakAssistantChange: (enabled) => console.log('Auto speak:', enabled),
  },
};

export const PlaybackActive: Story = {
  args: {
    onSend: () => {},
    modelLabel: 'Qwen3-30B-A3B-4bit',
    speechModels: [ttsModel],
    selectedSpeechModelId: ttsModel.modelId,
    isSpeaking: true,
    onStopSpeaking: () => console.log('Stop speech'),
  },
};

export const TranscriptionError: Story = {
  args: {
    onSend: () => {},
    modelLabel: 'Qwen3-30B-A3B-4bit',
    transcriptionModels: [sttModel],
    selectedTranscriptionModelId: sttModel.modelId,
    voiceError: 'Transcription failed.',
    onTranscribeAudio: async () => {
      throw new Error('Transcription failed.');
    },
  },
};

export const Loading: Story = {
  args: {
    onSend: () => {},
    isLoading: true,
    onCancel: () => alert('Cancelled'),
    modelLabel: 'Llama-3.1-8B-4bit',
    tps: 38.1,
  },
};

export const Interactive: Story = {
  render: () => {
    const [loading, setLoading] = useState(false);
    const [thinking, setThinking] = useState(false);
    return (
      <ChatForm
        onSend={(msg) => {
          console.log('Sent:', msg);
          setLoading(true);
          setTimeout(() => setLoading(false), 2000);
        }}
        onCancel={() => setLoading(false)}
        isLoading={loading}
        modelLabel="Qwen3-30B-A3B"
        showThinkingToggle
        thinkingEnabled={thinking}
        onToggleThinking={() => setThinking(!thinking)}
      />
    );
  },
};
