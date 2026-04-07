import styled from 'styled-components';
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
import { FamilySidebar } from './FamilySidebar';
import { ModelFilterPopover } from './ModelFilterPopover';
import { ModelPickerGroup } from './ModelPickerGroup';
import { HuggingFaceResultItem } from './HuggingFaceResultItem';

export interface ModelBrowserProps {
  models: ModelInfo[];
  selectedModelId: string | null;
  favorites: Set<string>;
  recentModelIds?: string[];
  existingModelIds?: Set<string>;
  canModelFit: (modelId: string) => boolean;
  getModelFitStatus: (modelId: string) => ModelFitStatus;
  onSelect: (modelId: string) => void;
  onToggleFavorite: (groupId: string) => void;
  onShowInfo?: (group: ModelGroup) => void;
  onAddModel?: (modelId: string) => Promise<void>;
  downloadStatusMap?: Map<string, DownloadAvailability>;
  instanceStatuses?: Record<string, InstanceStatus>;
  mode?: PickerMode;

  /** HuggingFace integration — optional, browser works without it */
  hfSearchResults?: HuggingFaceModel[];
  hfTrendingModels?: HuggingFaceModel[];
  hfIsSearching?: boolean;
  onHfSearch?: (query: string) => void;
}

/* ---------- layout ---------- */

const Container = styled.div`
  display: flex;
  height: 100%;
  background: ${({ theme }) => theme.colors.bg};
  color: ${({ theme }) => theme.colors.text};
  overflow: hidden;
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
`;

const FilterBtn = styled(Button)<{ $active: boolean }>`
  ${({ $active, theme }) =>
    $active &&
    `
      color: ${theme.colors.gold};
      border-color: ${theme.colors.goldDim};
    `}
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

/* ---------- component ---------- */

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
}: ModelBrowserProps) {
  const picker = useModelPicker({
    models,
    favorites,
    recentModelIds,
    canModelFit,
    getModelFitStatus,
    downloadStatusMap,
    instanceStatuses,
  });

  const isHf = picker.selectedFamily === 'huggingface';

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
      {/* Sidebar */}
      <FamilySidebar
        families={picker.uniqueFamilies}
        selectedFamily={picker.selectedFamily}
        hasFavorites={picker.hasFavorites}
        hasRecents={picker.hasRecents}
        onSelect={picker.setSelectedFamily}
      />

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
            placeholder={isHf ? 'Search mlx-community models…' : 'Search models…'}
            autoFocus
          />
          {!isHf && (
            <FilterBtn
              variant="outline"
              size="sm"
              $active={hasActiveFilters}
              onClick={() => picker.setShowFilters(!picker.showFilters)}
            >
              <FilterIcon />
              Filters
            </FilterBtn>
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
              {hfIsSearching && <EmptyMsg>Searching…</EmptyMsg>}
              {!hfIsSearching && hfModels.length === 0 && picker.searchQuery && (
                <EmptyMsg>No results found</EmptyMsg>
              )}
              {!picker.searchQuery.trim() && hfModels.length > 0 && (
                <SectionHeader>Trending</SectionHeader>
              )}
              {hfModels.map((m) => (
                <HuggingFaceResultItem
                  key={m.id}
                  model={m}
                  isAdded={models.some((mod) => mod.id === m.id)}
                  isInStore={existingModelIds.has(m.id)}
                  isAdding={false}
                  onAdd={async () => {
                    await onAddModel?.(m.id);
                    onSelect(m.id);
                  }}
                  onSelect={() => onSelect(m.id)}
                />
              ))}
            </>
          ) : (
            /* Local model groups */
            <>
              {picker.recommendedGroups.length > 0 && (
                <>
                  <SectionHeader>Recommended</SectionHeader>
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
              {picker.otherGroups.length > 0 && (
                <>
                  {picker.recommendedGroups.length > 0 && (
                    <SectionHeader>Other</SectionHeader>
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
                  {picker.searchQuery ? 'No models match your search' : 'No models available'}
                </EmptyMsg>
              )}
            </>
          )}
        </ListArea>
      </Main>
    </Container>
  );
}
