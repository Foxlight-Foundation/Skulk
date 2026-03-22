/**
 * ModelPickerModal
 *
 * Full-screen model selection dialog with:
 * - Family sidebar navigation
 * - Search filter (name + HuggingFace search)
 * - Capability / download status filters
 * - Model groups with ModelCard per entry
 * - Favorites and Recents sections
 *
 * Ported from ModelPickerModal.svelte.
 */
import React, {
  useState,
  useEffect,
  useCallback,
  useMemo,
  useRef,
} from 'react';
import { createPortal } from 'react-dom';
import styled from 'styled-components';
import { useTranslate } from '@tolgee/react';
import { FamilySidebar } from './FamilySidebar';
import { ModelPickerGroup } from './ModelPickerGroup';
import { ModelFilterPopover } from './ModelFilterPopover';
import { HuggingFaceResultItem } from './HuggingFaceResultItem';
import { ModelCard } from './ModelCard';
import { useModelsStore } from '../../stores/modelsStore';
import { useFavoritesStore } from '../../stores/favoritesStore';
import { useRecentsStore } from '../../stores/recentsStore';
import { useChatStore } from '../../stores/chatStore';
import { startDownload } from '../../api/client';
import { searchHuggingFace } from '../../api/client';
import type { HuggingFaceModel } from '../../api/types';
import {
  type ModelFilters,
  DEFAULT_FILTERS,
} from './ModelFilterPopover';

// ─── Styled components ────────────────────────────────────────────────────────

const Backdrop = styled.div`
  position: fixed;
  inset: 0;
  z-index: 1000;
  background: oklch(0 0 0 / 0.75);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
`;

const Modal = styled.div`
  background: ${({ theme }) => theme.colors.darkGray};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radius.lg};
  display: flex;
  flex-direction: column;
  width: 100%;
  max-width: 860px;
  height: 80vh;
  max-height: 700px;
  overflow: hidden;
  box-shadow: 0 24px 80px oklch(0 0 0 / 0.6);
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 16px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  flex-shrink: 0;
`;

const Title = styled.h2`
  font-size: 12px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.yellow};
  margin: 0;
`;

const SearchInput = styled.input`
  flex: 1;
  background: ${({ theme }) => theme.colors.mediumGray};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radius.md};
  color: ${({ theme }) => theme.colors.foreground};
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  padding: 6px 12px;
  &::placeholder { color: ${({ theme }) => theme.colors.mutedForeground}; }
  &:focus { outline: none; border-color: ${({ theme }) => theme.colors.yellowDarker}; }
`;

const FilterToggle = styled.button<{ $active: boolean }>`
  font-size: 11px;
  padding: 5px 10px;
  border-radius: ${({ theme }) => theme.radius.sm};
  border: 1px solid ${({ theme, $active }) =>
    $active ? theme.colors.yellow : theme.colors.border};
  background: ${({ theme, $active }) =>
    $active ? 'oklch(0.85 0.18 85 / 0.1)' : 'transparent'};
  color: ${({ theme, $active }) =>
    $active ? theme.colors.yellow : theme.colors.lightGray};
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.fast};
  &:hover { border-color: ${({ theme }) => theme.colors.yellowDarker}; color: ${({ theme }) => theme.colors.yellow}; }
`;

const CloseBtn = styled.button`
  background: none;
  border: none;
  color: ${({ theme }) => theme.colors.lightGray};
  cursor: pointer;
  font-size: 18px;
  padding: 2px 6px;
  border-radius: ${({ theme }) => theme.radius.sm};
  transition: color ${({ theme }) => theme.transitions.fast};
  &:hover { color: ${({ theme }) => theme.colors.foreground}; }
`;

const Body = styled.div`
  display: flex;
  flex: 1;
  overflow: hidden;
  position: relative;
`;

const Content = styled.div`
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
`;

const FilterPopoverWrapper = styled.div`
  position: absolute;
  top: 8px;
  right: 8px;
  z-index: 10;
`;

const EmptyState = styled.div`
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 8px;
  color: ${({ theme }) => theme.colors.mutedForeground};
  font-size: 12px;
  letter-spacing: 0.08em;
  padding: 40px;
  text-align: center;
`;

const SectionTitle = styled.div`
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.mutedForeground};
  padding: 10px 12px 4px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
`;

const HfSection = styled.div`
  display: flex;
  flex-direction: column;
`;

// ─── Component ─────────────────────────────────────────────────────────────────

export interface ModelPickerModalProps {
  onClose: () => void;
}

export const ModelPickerModal: React.FC<ModelPickerModalProps> = ({ onClose }) => {
  const { t } = useTranslate();
  const [search, setSearch] = useState('');
  const [filters, setFilters] = useState<ModelFilters>(DEFAULT_FILTERS);
  const [showFilters, setShowFilters] = useState(false);
  const [activeFamily, setActiveFamily] = useState<string | null>(null);
  const [hfResults, setHfResults] = useState<HuggingFaceModel[]>([]);
  const [hfLoading, setHfLoading] = useState(false);
  const hfTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const models = useModelsStore((s) => s.models);
  const fetchModels = useModelsStore((s) => s.fetchModels);
  const favoriteIds = useFavoritesStore((s) => s.favoriteIds);
  const recentIds = useRecentsStore((s) => s.getRecentModelIds());
  const setSelectedModel = useChatStore((s) => s.setSelectedModel);
  const recordLaunch = useRecentsStore((s) => s.recordRecentLaunch);

  useEffect(() => {
    void fetchModels();
    searchRef.current?.focus();
  }, [fetchModels]);

  // Escape to close
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  // HuggingFace search debounce
  useEffect(() => {
    if (hfTimerRef.current) clearTimeout(hfTimerRef.current);
    if (search.trim().length < 3) { setHfResults([]); return; }

    hfTimerRef.current = setTimeout(async () => {
      setHfLoading(true);
      try {
        const results = await searchHuggingFace(search.trim());
        setHfResults(results.slice(0, 20));
      } catch {
        setHfResults([]);
      } finally {
        setHfLoading(false);
      }
    }, 600);

    return () => { if (hfTimerRef.current) clearTimeout(hfTimerRef.current); };
  }, [search]);

  const handleLaunch = useCallback(
    (id: string) => {
      setSelectedModel(id);
      recordLaunch(id);
      onClose();
    },
    [setSelectedModel, recordLaunch, onClose],
  );

  const handleDownload = useCallback(async (id: string) => {
    try {
      await startDownload(id);
    } catch { /* ignore */ }
  }, []);

  // Filtered model list
  const filtered = useMemo(() => {
    let list = models;
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter((m) => m.id.toLowerCase().includes(q));
    }
    if (filters.capabilities.size > 0) {
      list = list.filter((m) => {
        const caps = m.capabilities ?? [];
        return [...filters.capabilities].some((c) => caps.includes(c));
      });
    }
    if (filters.downloadStatus === 'downloaded') {
      list = list.filter((m) => m.isDownloaded);
    } else if (filters.downloadStatus === 'not_downloaded') {
      list = list.filter((m) => !m.isDownloaded);
    }
    return list;
  }, [models, search, filters]);

  // Group by family
  const families = useMemo(
    () =>
      [...new Set(filtered.map((m) => m.family ?? 'Other').filter(Boolean))].sort(),
    [filtered],
  );

  const byFamily = useMemo(() => {
    const map = new Map<string, typeof filtered>();
    for (const m of filtered) {
      const fam = m.family ?? 'Other';
      if (!map.has(fam)) map.set(fam, []);
      map.get(fam)!.push(m);
    }
    return map;
  }, [filtered]);

  const favModels = useMemo(
    () => filtered.filter((m) => favoriteIds.includes(m.id)),
    [filtered, favoriteIds],
  );

  const recentModels = useMemo(
    () =>
      recentIds
        .map((id) => filtered.find((m) => m.id === id))
        .filter(Boolean) as typeof filtered,
    [filtered, recentIds],
  );

  const activeModels = activeFamily ? (byFamily.get(activeFamily) ?? []) : filtered;
  const hasFilters =
    filters.capabilities.size > 0 || filters.downloadStatus !== 'all';

  return createPortal(
    <Backdrop onClick={onClose}>
      <Modal onClick={(e) => e.stopPropagation()} role="dialog" aria-modal aria-label={t('models.title')}>
        <Header>
          <Title>{t('models.title')}</Title>
          <SearchInput
            ref={searchRef}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={t('models.search_placeholder')}
            aria-label={t('models.search_placeholder')}
          />
          <FilterToggle
            $active={hasFilters || showFilters}
            onClick={() => setShowFilters((p) => !p)}
            aria-label={t('models.filters')}
          >
            {t('models.filters')}
            {hasFilters && ' ●'}
          </FilterToggle>
          <CloseBtn onClick={onClose} aria-label={t('common.close')}>✕</CloseBtn>
        </Header>

        <Body>
          {/* Family sidebar — hidden when searching */}
          {!search.trim() && (
            <FamilySidebar
              families={families}
              activeFamily={activeFamily}
              onSelect={(f) =>
                setActiveFamily((prev) => (prev === f ? null : f))
              }
            />
          )}

          <Content>
            {showFilters && (
              <FilterPopoverWrapper>
                <ModelFilterPopover filters={filters} onChange={setFilters} />
              </FilterPopoverWrapper>
            )}

            {filtered.length === 0 && hfResults.length === 0 ? (
              <EmptyState>
                <div>{t('models.no_results')}</div>
                <div style={{ opacity: 0.6 }}>{t('models.no_results_hint')}</div>
              </EmptyState>
            ) : (
              <>
                {/* Favorites section */}
                {!search && favModels.length > 0 && (
                  <>
                    <SectionTitle>★ {t('models.favorites')}</SectionTitle>
                    {favModels.map((m) => (
                      <div key={m.id} style={{ padding: '4px 10px' }}>
                        <ModelCard
                          model={m}
                          onLaunch={handleLaunch}
                          onDownload={handleDownload}
                        />
                      </div>
                    ))}
                  </>
                )}

                {/* Recents section */}
                {!search && recentModels.length > 0 && (
                  <>
                    <SectionTitle>⏱ {t('models.recents')}</SectionTitle>
                    {recentModels.map((m) => (
                      <div key={m.id} style={{ padding: '4px 10px' }}>
                        <ModelCard
                          model={m}
                          onLaunch={handleLaunch}
                          onDownload={handleDownload}
                        />
                      </div>
                    ))}
                  </>
                )}

                {/* Model groups */}
                {activeFamily ? (
                  <ModelPickerGroup
                    key={activeFamily}
                    family={activeFamily}
                    models={activeModels}
                    defaultOpen
                    onLaunch={handleLaunch}
                    onDownload={handleDownload}
                  />
                ) : (
                  families.map((fam) => (
                    <ModelPickerGroup
                      key={fam}
                      family={fam}
                      models={byFamily.get(fam) ?? []}
                      defaultOpen={families.length === 1}
                      onLaunch={handleLaunch}
                      onDownload={handleDownload}
                    />
                  ))
                )}

                {/* HuggingFace results */}
                {search.trim().length >= 3 && (
                  <HfSection>
                    <SectionTitle>
                      🤗 {t('models.huggingface_search')}
                      {hfLoading && ' …'}
                    </SectionTitle>
                    {hfResults.map((m) => (
                      <HuggingFaceResultItem
                        key={m.id}
                        model={m}
                        onAdd={(hfModel) => handleDownload(hfModel.id)}
                      />
                    ))}
                  </HfSection>
                )}
              </>
            )}
          </Content>
        </Body>
      </Modal>
    </Backdrop>,
    document.body,
  );
};
