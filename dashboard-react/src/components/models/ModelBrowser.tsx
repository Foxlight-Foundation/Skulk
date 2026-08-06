import { useState } from 'react';
import styled, { css } from 'styled-components';
import { Button } from '../common/Button';
import type {
  ModelInfo,
  ModelFitStatus,
  DownloadAvailability,
  InstanceStatus,
  PickerMode,
  HuggingFaceModel,
  ModelGroup,
} from '../../types/models';
import { useModelPicker } from '../../hooks/useModelPicker';
import { SearchBar } from '../common/SearchBar';
import { ModelFilterPopover } from './ModelFilterPopover';
import { ModelPickerGroup } from './ModelPickerGroup';
import { HuggingFaceResultItem } from './HuggingFaceResultItem';
import { useSkulkTranslation } from '../../i18n/tolgee';

/** Data and actions used to browse Skulk-supported models or search Hugging Face. */
export interface ModelBrowserProps {
  models: ModelInfo[];
  selectedModelId: string | null;
  favorites: Set<string>;
  recentModelIds?: string[];
  existingModelIds?: Set<string>;
  canModelFit: (modelId: string) => boolean;
  getModelFitStatus: (modelId: string) => ModelFitStatus;
  onSelect: (modelId: string, ggufFile?: string | null) => void;
  onToggleFavorite: (groupId: string) => void;
  onShowInfo?: (group: ModelGroup) => void;
  onAddModel?: (modelId: string, ggufFile?: string | null) => Promise<boolean>;
  downloadStatusMap?: Map<string, DownloadAvailability>;
  instanceStatuses?: Record<string, InstanceStatus>;
  mode?: PickerMode;

  /** HuggingFace integration — optional, browser works without it */
  hfSearchResults?: HuggingFaceModel[];
  hfTrendingModels?: HuggingFaceModel[];
  hfIsSearching?: boolean;
  onHfSearch?: (query: string, mlxOnly?: boolean) => void;
  mlxOnly?: boolean;
  onToggleMlxOnly?: () => void;
}

/* ---------- layout ---------- */

const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background: ${({ theme }) => theme.colors.bg};
  color: ${({ theme }) => theme.colors.text};
  overflow: hidden;
`;

const SourceSwitcher = styled.div`
  display: flex;
  gap: 4px;
  padding: 12px 16px 0;
  flex-shrink: 0;

  @media (max-width: 640px) {
    & > button {
      flex: 1;
    }
  }
`;

const SourceButton = styled.button<{ $active: boolean }>`
  appearance: none;
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  background: transparent;
  color: ${({ theme }) => theme.colors.textSecondary};
  cursor: pointer;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  font-weight: 600;
  padding: 8px 12px;
  transition: background 0.15s, border-color 0.15s, color 0.15s;

  &:hover {
    background: ${({ theme }) => theme.colors.surfaceHover};
    color: ${({ theme }) => theme.colors.text};
  }

  ${({ $active }) => $active && css`
    background: ${({ theme }) => theme.colors.goldBg};
    border-color: ${({ theme }) => theme.colors.goldDim};
    color: ${({ theme }) => theme.colors.gold};
  `}
`;

const Main = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  overflow: hidden;
`;

const Toolbar = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  position: relative;
  flex-shrink: 0;

  & > *:first-child {
    flex: 1;
    min-width: 0;
  }

  @media (max-width: 640px) {
    flex-wrap: wrap;

    & > *:first-child {
      flex-basis: 100%;
    }
  }
`;

const FilterBtn = styled(Button)<{ $active: boolean }>`
  ${({ $active, theme }) =>
    $active &&
    `
      color: ${theme.colors.gold};
      border-color: ${theme.colors.goldDim};
    `}
`;

const CatalogScopeSelect = styled.select`
  min-width: 150px;
  max-width: 210px;
  height: 34px;
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surface};
  color: ${({ theme }) => theme.colors.text};
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
  padding: 0 28px 0 10px;
`;

const ListArea = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 0 12px 12px;
`;

const SectionHeader = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.textMuted};
  padding: 12px 12px 6px;
`;

const EmptyMsg = styled.div`
  padding: 24px;
  text-align: center;
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: ${({ theme }) => theme.fontSizes.tableBody};
`;

const FilterIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
  </svg>
);

const FAMILY_TOKEN_LABELS: Readonly<Record<string, string>> = {
  audiodit: 'AudioDiT',
  bert: 'BERT',
  glm: 'GLM',
  gpt: 'GPT',
  longcat: 'LongCat',
  oss: 'OSS',
  qwen: 'Qwen',
  qwen3: 'Qwen3',
  stt: 'STT',
  tts: 'TTS',
  vl: 'VL',
};

function familyLabel(family: string): string {
  return family
    .split(/[_-]/)
    .filter(Boolean)
    .map((token) => FAMILY_TOKEN_LABELS[token.toLowerCase()]
      ?? `${token.charAt(0).toUpperCase()}${token.slice(1)}`)
    .join(' ');
}

/* ---------- component ---------- */

/** Render model discovery with separate source selection and catalog filtering. */
export function ModelBrowser({
  models,
  selectedModelId,
  favorites,
  recentModelIds,
  existingModelIds = new Set(),
  canModelFit,
  getModelFitStatus,
  onSelect,
  onToggleFavorite,
  onShowInfo,
  onAddModel,
  downloadStatusMap,
  instanceStatuses,
  mode = 'launch',
  hfSearchResults,
  hfTrendingModels,
  hfIsSearching,
  onHfSearch,
  mlxOnly = false,
  onToggleMlxOnly,
}: ModelBrowserProps) {
  const { t } = useSkulkTranslation();
  const [source, setSource] = useState<'catalog' | 'huggingface'>('catalog');
  const picker = useModelPicker({
    models,
    favorites,
    recentModelIds,
    canModelFit,
    getModelFitStatus,
    downloadStatusMap,
    instanceStatuses,
  });

  const isHf = source === 'huggingface';

  const hasActiveFilters =
    picker.filters.capabilities.length > 0 ||
    picker.filters.sizeRange !== null ||
    picker.filters.downloadedOnly ||
    picker.filters.readyOnly;

  const hfModels = (hfSearchResults && hfSearchResults.length > 0)
    ? hfSearchResults
    : (picker.searchQuery.trim() ? [] : hfTrendingModels ?? []);

  return (
    <Container>
      <SourceSwitcher role="group" aria-label={t('modelBrowser.source', 'Model source')}>
        <SourceButton
          type="button"
          aria-pressed={!isHf}
          $active={!isHf}
          onClick={() => {
            setSource('catalog');
            picker.setSelectedFamily(null);
          }}
        >
          {t('modelBrowser.supportedCatalog', 'Supported models')}
        </SourceButton>
        <SourceButton
          type="button"
          aria-pressed={isHf}
          $active={isHf}
          onClick={() => {
            setSource('huggingface');
            if (picker.searchQuery.trim() && onHfSearch) {
              onHfSearch(picker.searchQuery);
            }
          }}
        >
          {t('modelBrowser.huggingFaceSearch', 'Search Hugging Face')}
        </SourceButton>
      </SourceSwitcher>

      {/* Main content */}
      <Main>
        {/* Toolbar */}
        <Toolbar>
          <SearchBar
            value={picker.searchQuery}
            onChange={(q) => {
              picker.setSearchQuery(q);
              if (isHf && onHfSearch) onHfSearch(q);
            }}
            placeholder={isHf
              ? (mlxOnly
                  ? t('modelBrowser.searchMlxCommunity', 'Search mlx-community...')
                  : t('modelBrowser.searchHuggingFace', 'Search all of Hugging Face...'))
              : t('modelBrowser.searchModels', 'Search models...')}
            autoFocus
            ariaLabel={t('modelBrowser.searchAriaLabel', 'Search models')}
          />
          {isHf && onToggleMlxOnly && (
            <FilterBtn
              variant="outline"
              size="sm"
              $active={mlxOnly}
              onClick={() => {
                const nextMlxOnly = !mlxOnly;
                onToggleMlxOnly();
                if (picker.searchQuery.trim() && onHfSearch) {
                  onHfSearch(picker.searchQuery, nextMlxOnly);
                }
              }}
            >
              {t('modelBrowser.mlxOnly', 'MLX only')}
            </FilterBtn>
          )}
          {!isHf && (
            <>
              <CatalogScopeSelect
                value={picker.selectedFamily ?? 'all'}
                aria-label={t('modelBrowser.catalogScope', 'Filter supported models by family')}
                onChange={(event) => {
                  const value = event.target.value;
                  picker.setSelectedFamily(value === 'all' ? null : value);
                }}
              >
                <option value="all">{t('modelBrowser.allSupported', 'All supported models')}</option>
                {picker.hasFavorites && (
                  <option value="favorites">{t('familySidebar.favorites', 'Favorites')}</option>
                )}
                {picker.hasRecents && (
                  <option value="recents">{t('familySidebar.recent', 'Recent')}</option>
                )}
                <optgroup label={t('modelInfo.family', 'Family')}>
                  {picker.uniqueFamilies.map((family) => (
                    <option key={family} value={family}>{familyLabel(family)}</option>
                  ))}
                </optgroup>
              </CatalogScopeSelect>
              <FilterBtn
                variant="outline"
                size="sm"
                $active={hasActiveFilters}
                onClick={() => picker.setShowFilters(!picker.showFilters)}
              >
                <FilterIcon />
                {t('modelBrowser.filters', 'Filters')}
              </FilterBtn>
            </>
          )}
          {picker.showFilters && !isHf && (
            <ModelFilterPopover
              filters={picker.filters}
              onChange={picker.setFilters}
              onClear={picker.clearFilters}
              onClose={() => picker.setShowFilters(false)}
            />
          )}
        </Toolbar>

        {/* List */}
        <ListArea>
          {isHf ? (
            /* HuggingFace results */
            <>
              {hfIsSearching && <EmptyMsg>{t('modelBrowser.searching', 'Searching...')}</EmptyMsg>}
              {!hfIsSearching && hfModels.length === 0 && picker.searchQuery && (
                <EmptyMsg>{t('modelBrowser.noResultsFound', 'No results found')}</EmptyMsg>
              )}
              {!picker.searchQuery.trim() && hfModels.length > 0 && (
                <SectionHeader>{t('modelBrowser.trending', 'Trending')}</SectionHeader>
              )}
              {hfModels.map((m) => (
                <HuggingFaceResultItem
                  key={m.id}
                  model={m}
                  isAdded={models.some((mod) => mod.id === m.id)}
                  isInStore={existingModelIds.has(m.id) && !m.matched_file}
                  isAdding={false}
                  onAdd={async () => {
                    const added = await onAddModel?.(m.id, m.matched_file);
                    if (added !== false) onSelect(m.id, m.matched_file);
                  }}
                  onSelect={async () => {
                    if (m.matched_file) {
                      const updated = await onAddModel?.(m.id, m.matched_file);
                      if (updated === false) return;
                    }
                    onSelect(m.id, m.matched_file);
                  }}
                />
              ))}
            </>
          ) : (
            /* Local model groups */
            <>
              {mode === 'store-download' && picker.filteredGroups.length > 0 && (
                <>
                  <SectionHeader>{t('modelBrowser.supportedCatalog', 'Supported models')}</SectionHeader>
                  {picker.filteredGroups.map((g) => (
                    <ModelPickerGroup
                      key={g.id}
                      group={g}
                      isExpanded={picker.expandedGroups.has(g.id)}
                      isFavorite={favorites.has(g.id)}
                      selectedModelId={selectedModelId}
                      canModelFit={canModelFit}
                      getModelFitStatus={getModelFitStatus}
                      onToggleExpand={() => picker.toggleExpanded(g.id)}
                      onSelectModel={onSelect}
                      onToggleFavorite={onToggleFavorite}
                      onShowInfo={onShowInfo}
                      downloadStatusMap={downloadStatusMap}
                      instanceStatuses={instanceStatuses}
                      mode={mode}
                    />
                  ))}
                </>
              )}
              {mode !== 'store-download' && picker.recommendedGroups.length > 0 && (
                <>
                  <SectionHeader>{t('modelBrowser.recommended', 'Recommended')}</SectionHeader>
                  {picker.recommendedGroups.map((g) => (
                    <ModelPickerGroup
                      key={g.id}
                      group={g}
                      isExpanded={picker.expandedGroups.has(g.id)}
                      isFavorite={favorites.has(g.id)}
                      selectedModelId={selectedModelId}
                      canModelFit={canModelFit}
                      getModelFitStatus={getModelFitStatus}
                      onToggleExpand={() => picker.toggleExpanded(g.id)}
                      onSelectModel={onSelect}
                      onToggleFavorite={onToggleFavorite}
                      onShowInfo={onShowInfo}
                      downloadStatusMap={downloadStatusMap}
                      instanceStatuses={instanceStatuses}
                      mode={mode}
                    />
                  ))}
                </>
              )}
              {mode !== 'store-download' && picker.otherGroups.length > 0 && (
                <>
                  {picker.recommendedGroups.length > 0 && (
                    <SectionHeader>{t('modelBrowser.other', 'Other')}</SectionHeader>
                  )}
                  {picker.otherGroups.map((g) => (
                    <ModelPickerGroup
                      key={g.id}
                      group={g}
                      isExpanded={picker.expandedGroups.has(g.id)}
                      isFavorite={favorites.has(g.id)}
                      selectedModelId={selectedModelId}
                      canModelFit={canModelFit}
                      getModelFitStatus={getModelFitStatus}
                      onToggleExpand={() => picker.toggleExpanded(g.id)}
                      onSelectModel={onSelect}
                      onToggleFavorite={onToggleFavorite}
                      onShowInfo={onShowInfo}
                      downloadStatusMap={downloadStatusMap}
                      instanceStatuses={instanceStatuses}
                      mode={mode}
                    />
                  ))}
                </>
              )}
              {picker.filteredGroups.length === 0 && (
                <EmptyMsg>
                  {picker.searchQuery
                    ? t('modelBrowser.noModelsMatch', 'No models match your search')
                    : t('modelBrowser.noModelsAvailable', 'No models available')}
                </EmptyMsg>
              )}
            </>
          )}
        </ListArea>
      </Main>
    </Container>
  );
}
