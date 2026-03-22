/**
 * MainPage  —  "/"
 *
 * The primary interface: topology visualization + chat.
 * Composes ChatSidebar, TopologyPane, ChatMessages, ChatForm,
 * ChatModelSelector, and ModelPickerModal.
 */
import React, { useState, useCallback, useMemo } from 'react';
import styled from 'styled-components';
import { ChatSidebar } from '../components/chat/ChatSidebar';
import { ChatMessages } from '../components/chat/ChatMessages';
import { ChatForm } from '../components/chat/ChatForm';
import { ChatModelSelector, pickAutoModel } from '../components/chat/ChatModelSelector';
import type { ChatModelInfo } from '../components/chat/ChatModelSelector';
import { TopologyPane } from '../components/topology/TopologyPane';
import { SystemWarningsBanner } from '../components/topology/SystemWarningsBanner';
import { ModelPickerModal } from '../components/models/ModelPickerModal';
import { useChatStore } from '../stores/chatStore';
import { useTopologyStore } from '../stores/topologyStore';
import { useModelsStore } from '../stores/modelsStore';
import { useUIStore } from '../stores/uiStore';
import type { MessageContent, TextContent, ImageContent } from '../api/types';
import type { ChatUploadedFile } from '../types/files';

// ─── Styled components ────────────────────────────────────────────────────────

const PageLayout = styled.div`
  display: flex;
  height: 100%;
  width: 100%;
  overflow: hidden;
`;

const SidebarColumn = styled.div<{ $visible: boolean }>`
  width: 240px;
  flex-shrink: 0;
  display: ${({ $visible }) => ($visible ? 'flex' : 'none')};
  flex-direction: column;

  @media (max-width: 768px) {
    display: none;
  }
`;

const MainColumn = styled.div`
  flex: 1;
  display: flex;
  overflow: hidden;
  min-width: 0;
`;

const ChatColumn = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-width: 0;
`;

const MessagesArea = styled.div`
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
`;

const EmptyChatArea = styled.div`
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
`;

const FormArea = styled.div`
  padding: 12px 16px 16px;
  flex-shrink: 0;
`;

const TopologyColumn = styled.div<{ $minimized: boolean }>`
  width: ${({ $minimized }) => ($minimized ? '180px' : '380px')};
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  transition: width 0.2s ease;

  @media (max-width: 900px) {
    display: none;
  }
`;

const TopologyOnlyLayout = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const MobileTopBar = styled.div`
  display: none;
  @media (max-width: 768px) {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-bottom: 1px solid ${({ theme }) => theme.colors.border};
    flex-shrink: 0;
  }
`;

const MobileMenuBtn = styled.button`
  background: none;
  border: none;
  color: ${({ theme }) => theme.colors.lightGray};
  cursor: pointer;
  padding: 4px;
  display: flex;
  align-items: center;
  &:hover { color: ${({ theme }) => theme.colors.foreground}; }
`;

// ─── Helpers ──────────────────────────────────────────────────────────────────

function buildMessageContent(
  text: string,
  files: ChatUploadedFile[],
): MessageContent {
  if (files.length === 0) return text;

  const parts: Array<TextContent | ImageContent> = [];
  if (text.trim()) parts.push({ type: 'text', text });

  for (const file of files) {
    if (file.preview) {
      parts.push({ type: 'image_url', image_url: { url: file.preview } });
    } else if (file.textContent) {
      parts.push({ type: 'text', text: `[File: ${file.name}]\n${file.textContent}` });
    }
  }

  return parts.length > 0 ? parts : text;
}

// ─── Component ─────────────────────────────────────────────────────────────────

const MainPage: React.FC = () => {
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  // Chat store
  const selectedModelId = useChatStore((s) => s.selectedModelId);
  const setSelectedModel = useChatStore((s) => s.setSelectedModel);
  const createConversation = useChatStore((s) => s.createConversation);
  const activeConversationId = useChatStore((s) => s.activeConversationId);
  const hasStartedChat = useChatStore((s) => s.hasStartedChat);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const generateImageMessage = useChatStore((s) => s.generateImageMessage);
  const editImageMessage = useChatStore((s) => s.editImageMessage);
  const editingImageBase64 = useChatStore((s) => s.editingImageBase64);

  // UI store
  const chatSidebarVisible = useUIStore((s) => s.chatSidebarVisible);
  const topologyMinimized = useUIStore((s) => s.topologyMinimized);
  const topologyOnlyMode = useUIStore((s) => s.topologyOnlyMode);

  // Models / topology
  const models = useModelsStore((s) => s.models);
  const nodeMemoryUsages = useTopologyStore((s) => s.nodeMemoryUsages);
  const topology = useTopologyStore((s) => s.topology);

  // Compute total cluster RAM (GB)
  const totalMemoryGB = useMemo(() => {
    let bytes = 0;
    for (const mem of Object.values(nodeMemoryUsages)) bytes += mem.ramTotal?.inBytes ?? 0;
    return bytes / (1024 ** 3);
  }, [nodeMemoryUsages]);

  // Cluster label
  const clusterLabel = useMemo(() => {
    const n = Object.keys(topology?.nodes ?? {}).length;
    if (n === 0) return 'your device';
    return n === 1 ? '1-device cluster' : `${n}-device cluster`;
  }, [topology]);

  // Model task / capability maps
  const modelTasks = useMemo(() => {
    const map: Record<string, string[]> = {};
    for (const m of models) if (m.tasks) map[m.id] = m.tasks;
    return map;
  }, [models]);

  const modelCapabilities = useMemo(() => {
    const map: Record<string, string[]> = {};
    for (const m of models) if (m.capabilities) map[m.id] = m.capabilities;
    return map;
  }, [models]);

  // ChatModelInfo list for selector
  const chatModelInfos = useMemo(
    (): ChatModelInfo[] =>
      models
        .filter((m) => m.base_model)
        .map((m) => ({
          id: m.id,
          name: m.name,
          base_model: m.base_model!,
          storage_size_megabytes: m.sizeBytes ? m.sizeBytes / (1024 * 1024) : 0,
          capabilities: m.capabilities,
          family: m.family,
          quantization: m.quantization,
        })),
    [models],
  );

  const supportsOnlyImageEditing = selectedModelId
    ? (modelTasks[selectedModelId] ?? []).includes('ImageToImage') &&
      !(modelTasks[selectedModelId] ?? []).includes('TextToImage')
    : false;

  const isTextToImageModel = selectedModelId
    ? (modelTasks[selectedModelId] ?? []).includes('TextToImage')
    : false;

  const handleAutoSend = useCallback(
    async (content: string, files?: ChatUploadedFile[]) => {
      // Auto-pick model if none selected
      let modelId = selectedModelId;
      if (!modelId) {
        const picked = pickAutoModel(chatModelInfos, totalMemoryGB || 8);
        if (picked) {
          setSelectedModel(picked.id);
          modelId = picked.id;
        } else {
          setShowModelPicker(true);
          return;
        }
      }

      // Ensure conversation exists
      if (!activeConversationId) createConversation();

      // Image edit takes priority
      const isEditMode =
        editingImageBase64 !== null ||
        (!!files?.length && supportsOnlyImageEditing);

      if (isEditMode && editingImageBase64) {
        await editImageMessage(content);
        return;
      }

      // Text-to-image generation
      if (isTextToImageModel && !editingImageBase64) {
        await generateImageMessage(content);
        return;
      }

      // Standard chat with optional file attachments
      const msgContent = buildMessageContent(content, files ?? []);
      await sendMessage(msgContent);
    },
    [
      selectedModelId,
      chatModelInfos,
      totalMemoryGB,
      setSelectedModel,
      activeConversationId,
      createConversation,
      editingImageBase64,
      supportsOnlyImageEditing,
      isTextToImageModel,
      editImageMessage,
      generateImageMessage,
      sendMessage,
    ],
  );

  const handleModelSelect = useCallback(
    (modelId: string) => {
      setSelectedModel(modelId);
      if (!activeConversationId) createConversation();
    },
    [setSelectedModel, activeConversationId, createConversation],
  );

  const handleNewChat = useCallback(() => {
    createConversation();
  }, [createConversation]);

  const chatStarted = hasStartedChat();

  // Topology-only mode
  if (topologyOnlyMode) {
    return (
      <PageLayout>
        <TopologyOnlyLayout>
          <SystemWarningsBanner />
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <TopologyPane />
          </div>
        </TopologyOnlyLayout>
      </PageLayout>
    );
  }

  return (
    <>
      <PageLayout>
        {/* Desktop sidebar */}
        <SidebarColumn $visible={chatSidebarVisible}>
          <ChatSidebar onNewChat={handleNewChat} />
        </SidebarColumn>

        <MainColumn>
          <ChatColumn>
            {/* Mobile menu button */}
            <MobileTopBar>
              <MobileMenuBtn
                type="button"
                onClick={() => setMobileSidebarOpen(true)}
                aria-label="Open sidebar"
              >
                <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </MobileMenuBtn>
            </MobileTopBar>

            <SystemWarningsBanner />

            <MessagesArea>
              {chatStarted ? (
                <ChatMessages />
              ) : (
                <EmptyChatArea>
                  <ChatModelSelector
                    models={chatModelInfos}
                    clusterLabel={clusterLabel}
                    totalMemoryGB={totalMemoryGB || 8}
                    onSelect={handleModelSelect}
                    onAddModel={() => setShowModelPicker(true)}
                  />
                </EmptyChatArea>
              )}
            </MessagesArea>

            <FormArea>
              <ChatForm
                showModelSelector
                modelTasks={modelTasks}
                modelCapabilities={modelCapabilities}
                showHelperText={!chatStarted}
                autofocus
                onAutoSend={handleAutoSend}
                onOpenModelPicker={() => setShowModelPicker(true)}
              />
            </FormArea>
          </ChatColumn>

          {/* Topology right column */}
          <TopologyColumn $minimized={topologyMinimized}>
            <TopologyPane />
          </TopologyColumn>
        </MainColumn>
      </PageLayout>

      {/* Mobile sidebar drawer */}
      <ChatSidebar
        isMobileDrawer
        isOpen={mobileSidebarOpen}
        onClose={() => setMobileSidebarOpen(false)}
        onNewChat={handleNewChat}
        onSelectConversation={() => setMobileSidebarOpen(false)}
      />

      {/* Model picker */}
      {showModelPicker && <ModelPickerModal onClose={() => setShowModelPicker(false)} />}
    </>
  );
};

export default MainPage;
