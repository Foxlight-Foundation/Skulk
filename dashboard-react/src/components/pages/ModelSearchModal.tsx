import { useCallback, useEffect, useRef, useState } from 'react';
import styled from 'styled-components';
import { ModelBrowser } from '../models/ModelBrowser';
import type { ModelInfo, HuggingFaceModel } from '../../types/models';
import { addToast } from '../../hooks/useToast';

interface ModelSearchModalProps {
  open: boolean;
  onClose: () => void;
  existingModelIds: Set<string>;
  onDownloadStarted: () => void;
}

export function ModelSearchModal({
  open,
  onClose,
  existingModelIds,
  onDownloadStarted,
}: ModelSearchModalProps) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [favorites, setFavorites] = useState<Set<string>>(new Set());
  const [hfResults, setHfResults] = useState<HuggingFaceModel[]>([]);
  const [hfTrending, setHfTrending] = useState<HuggingFaceModel[]>([]);
  const [hfSearching, setHfSearching] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  // Fetch models on open
  useEffect(() => {
    if (!open) return;
    (async () => {
      try {
        const res = await fetch('/models');
        if (!res.ok) return;
        const data = await res.json();
        setModels(data.data ?? []);
      } catch { /* ignore */ }
    })();
    // Fetch trending
    (async () => {
      try {
        const res = await fetch('/models/search?query=&limit=20');
        if (!res.ok) return;
        setHfTrending(await res.json());
      } catch { /* ignore */ }
    })();
  }, [open]);

  const handleHfSearch = useCallback((query: string) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (!query.trim()) {
      setHfResults([]);
      setHfSearching(false);
      return;
    }
    setHfSearching(true);
    debounceRef.current = setTimeout(async () => {
      try {
        const res = await fetch(`/models/search?query=${encodeURIComponent(query)}&limit=20`);
        if (res.ok) setHfResults(await res.json());
      } catch { /* ignore */ }
      finally { setHfSearching(false); }
    }, 500);
  }, []);

  const handleSelect = useCallback(async (modelId: string) => {
    try {
      const res = await fetch(`/store/models/${encodeURIComponent(modelId)}/download`, {
        method: 'POST',
      });
      if (res.ok) {
        addToast({ type: 'success', message: `Downloading ${modelId} to store` });
        onDownloadStarted();
      } else {
        addToast({ type: 'error', message: `Failed to start download for ${modelId}` });
      }
    } catch {
      addToast({ type: 'error', message: `Failed to start download` });
    }
  }, [onDownloadStarted]);

  const handleAddModel = useCallback(async (modelId: string) => {
    try {
      const res = await fetch('/models/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId }),
      });
      if (res.ok) {
        addToast({ type: 'success', message: `Added ${modelId}` });
        // Refresh model list
        const listRes = await fetch('/models');
        if (listRes.ok) {
          const data = await listRes.json();
          setModels(data.data ?? []);
        }
      }
    } catch { /* ignore */ }
  }, []);

  const toggleFavorite = useCallback((groupId: string) => {
    setFavorites(prev => {
      const next = new Set(prev);
      if (next.has(groupId)) next.delete(groupId);
      else next.add(groupId);
      return next;
    });
  }, []);

  if (!open) return null;

  return (
    <>
      <Backdrop onClick={onClose} />
      <ModalContainer>
        <ModalHeader>
          <ModalTitle>Find Models</ModalTitle>
          <CloseButton onClick={onClose} aria-label="Close">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M6 18L18 6M6 6l12 12" />
            </svg>
          </CloseButton>
        </ModalHeader>
        <ModalBody>
          <ModelBrowser
            models={models}
            selectedModelId={null}
            favorites={favorites}
            existingModelIds={existingModelIds}
            canModelFit={() => true}
            getModelFitStatus={() => 'fits_now'}
            onSelect={handleSelect}
            onToggleFavorite={toggleFavorite}
            onAddModel={handleAddModel}
            hfSearchResults={hfResults}
            hfTrendingModels={hfTrending}
            hfIsSearching={hfSearching}
            onHfSearch={handleHfSearch}
            mode="store-download"
          />
        </ModalBody>
      </ModalContainer>
    </>
  );
}

const Backdrop = styled.div`
  position: fixed;
  inset: 0;
  z-index: 50;
  background: rgba(0, 0, 0, 0.7);
`;

const ModalContainer = styled.div`
  position: fixed;
  inset: 40px;
  z-index: 51;
  display: flex;
  flex-direction: column;
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg};
  overflow: hidden;
`;

const ModalHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
`;

const ModalTitle = styled.h2`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.lg};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.gold};
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 1px;
`;

const CloseButton = styled.button`
  all: unset;
  cursor: pointer;
  color: ${({ theme }) => theme.colors.textMuted};
  transition: color 0.15s;
  display: flex;
  &:hover { color: ${({ theme }) => theme.colors.text}; }
`;

const ModalBody = styled.div`
  flex: 1;
  overflow: hidden;
`;
