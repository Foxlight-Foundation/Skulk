import { useState } from 'react';
import styled, { css } from 'styled-components';
import { Button } from '../common/Button';
import { Spinner } from '../common/Spinner';
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
import { InfoTooltip } from '../common/InfoTooltip';
import type { BurstInfo } from './burst';

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

  /** Burst verdict for one catalog variant id; enables fleet-first ordering. */
  getBurstInfo?: (variantId: string) => BurstInfo | null;
  /** Burst verdict for one Hugging Face result (estimate-based). */
  getHfBurstInfo?: (model: HuggingFaceModel) => BurstInfo | null;
  /** Fleet total memory, shown in burst tooltips. */
  fleetMemoryBytes?: number;
  /** Fetch another page of Hugging Face results; absent when exhausted. */
  onHfLoadMore?: () => void;
  /** Active Hugging Face task filter (pipeline tag); null browses all. */
  hfTask?: string | null;
  /** Change the Hugging Face task filter. */
  onHfTaskChange?: (task: string | null) => void;
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
  margin: 14px 16px 0;
  padding: 4px;
  border: 1px solid ${({ theme }) => theme.colors.borderLight};
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surfaceSunken};
  flex-shrink: 0;
`;

const SourceButton = styled.button<{ $active: boolean }>`
  appearance: none;
  flex: 1;
  border: 1px solid transparent;
  border-radius: ${({ theme }) => theme.radii.sm};
  background: transparent;
  color: ${({ theme }) => theme.colors.textSecondary};
  cursor: pointer;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.nav};
  font-weight: 600;
  padding: 7px 12px;
  transition: background 0.15s, border-color 0.15s, color 0.15s;

  &:hover {
    color: ${({ theme }) => theme.colors.text};
  }

  ${({ $active }) => $active && css`
    background: ${({ theme }) => theme.colors.surface};
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
  padding: 12px 16px 8px;
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

/* Horizontally scrollable rail of family scope chips. */
const FamilyRail = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 16px 10px;
  overflow-x: auto;
  flex-shrink: 0;
  scrollbar-width: none;

  &::-webkit-scrollbar {
    display: none;
  }
`;

const FamilyChip = styled.button<{ $active: boolean }>`
  appearance: none;
  flex-shrink: 0;
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: 999px;
  background: transparent;
  color: ${({ theme }) => theme.colors.textSecondary};
  cursor: pointer;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-weight: 500;
  padding: 4px 12px;
  transition: background 0.15s, border-color 0.15s, color 0.15s;
  white-space: nowrap;

  &:hover {
    color: ${({ theme }) => theme.colors.text};
    background: ${({ theme }) => theme.colors.surfaceHover};
  }

  ${({ $active }) => $active && css`
    background: ${({ theme }) => theme.colors.goldBg};
    border-color: ${({ theme }) => theme.colors.goldDim};
    color: ${({ theme }) => theme.colors.gold};

    &:hover {
      background: ${({ theme }) => theme.colors.goldBg};
      color: ${({ theme }) => theme.colors.gold};
    }
  `}
`;

const ListArea = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 0 16px 16px;
`;

/* Card wrapping result rows so the list reads as one surface, matching the
 * store table's bordered-card dialect. */
const ListCard = styled.div`
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surface};
  overflow: hidden;
`;

const SectionHeader = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.textMuted};
  padding: 4px 2px 8px;
`;

const MoreRow = styled.div`
  display: flex;
  justify-content: center;
  padding: 12px 0 4px;
`;

const EmptyMsg = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 40px 24px;
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
  getBurstInfo,
  getHfBurstInfo,
  fleetMemoryBytes,
  onHfLoadMore,
  hfTask = null,
  onHfTaskChange,
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

  // Hugging Face task filters offered on the search tab: the slice of the
  // Hub typology Skulk users act on, labeled in the catalog's vocabulary.
  const hfTaskOptions = [
    { tag: 'text-generation', label: t('modelBrowser.taskText', 'Text') },
    { tag: 'image-text-to-text', label: t('modelBrowser.taskVision', 'Vision') },
    { tag: 'automatic-speech-recognition', label: t('modelBrowser.taskStt', 'STT') },
    { tag: 'text-to-speech', label: t('modelBrowser.taskTts', 'TTS') },
    { tag: 'sentence-similarity', label: t('modelBrowser.taskEmbedding', 'Embedding') },
    { tag: 'text-to-image', label: t('modelBrowser.taskImageGen', 'Image gen') },
  ];

  const hfModelsRaw = (hfSearchResults && hfSearchResults.length > 0)
    ? hfSearchResults
    : (picker.searchQuery.trim() ? [] : hfTrendingModels ?? []);

  // Fleet-first ordering: stable partition so favorites/recents/popularity
  // order survives inside each half. A group is locally placeable when ANY
  // of its variants is; rows we can make no claim about stay with the
  // placeable half rather than being demoted on a guess.
  const groupIsBurst = (g: ModelGroup) =>
    getBurstInfo !== undefined && g.variants.every((v) => getBurstInfo(v.id) !== null);
  const localGroups = picker.filteredGroups.filter((g) => !groupIsBurst(g));
  const burstGroups = picker.filteredGroups.filter(groupIsBurst);

  const hfIsBurst = (m: HuggingFaceModel) =>
    getHfBurstInfo !== undefined && getHfBurstInfo(m) !== null;
  const hfLocal = hfModelsRaw.filter((m) => !hfIsBurst(m));
  const hfBurst = hfModelsRaw.filter(hfIsBurst);
  const hfModels = hfModelsRaw;

  const burstSectionHeader = (
    <SectionHeader style={{ paddingTop: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
      {t('modelBrowser.needsBurst', 'Needs burst capacity')}
      <InfoTooltip
        size={13}
        placement="right"
        delay={100}
        content={
          <div style={{ maxWidth: 280 }}>
            {t(
              'modelBrowser.needsBurstExplainer',
              'These models will not fit on your local hardware, either by size or because no local node runs their artifact format. Serving one means adding remote resources to the fabric, such as a cloud GPU node. Automated burst provisioning arrives in a follow-up release.',
            )}
          </div>
        }
      />
    </SectionHeader>
  );

  const renderHfItem = (m: HuggingFaceModel) => (
    <HuggingFaceResultItem
      key={m.id}
      model={m}
      isAdded={models.some((mod) => mod.id === m.id)}
      isInStore={existingModelIds.has(m.id) && !m.matched_file}
      isAdding={false}
      burst={getHfBurstInfo?.(m) ?? null}
      fleetMemoryBytes={fleetMemoryBytes}
      onSelectQuant={async (ggufFile) => {
        const added = await onAddModel?.(m.id, ggufFile);
        if (added !== false) onSelect(m.id, ggufFile);
      }}
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
  );

  const renderGroup = (g: ModelGroup) => (
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
      getBurstInfo={getBurstInfo}
      fleetMemoryBytes={fleetMemoryBytes}
    />
  );

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
            <FilterBtn
              variant="outline"
              size="sm"
              $active={hasActiveFilters}
              onClick={() => picker.setShowFilters(!picker.showFilters)}
            >
              <FilterIcon />
              {t('modelBrowser.filters', 'Filters')}
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

        {/* Hugging Face task scope */}
        {isHf && onHfTaskChange && (
          <FamilyRail role="group" aria-label={t('modelBrowser.taskScope', 'Filter Hugging Face results by task')}>
            <FamilyChip
              type="button"
              $active={hfTask === null}
              aria-pressed={hfTask === null}
              onClick={() => onHfTaskChange(null)}
            >
              {t('modelBrowser.allFamilies', 'All')}
            </FamilyChip>
            {hfTaskOptions.map(({ tag, label }) => (
              <FamilyChip
                key={tag}
                type="button"
                $active={hfTask === tag}
                aria-pressed={hfTask === tag}
                onClick={() => onHfTaskChange(tag)}
              >
                {label}
              </FamilyChip>
            ))}
          </FamilyRail>
        )}

        {/* Catalog family scope */}
        {!isHf && (
          <FamilyRail role="group" aria-label={t('modelBrowser.catalogScope', 'Filter supported models by family')}>
            <FamilyChip
              type="button"
              $active={picker.selectedFamily === null}
              aria-pressed={picker.selectedFamily === null}
              onClick={() => picker.setSelectedFamily(null)}
            >
              {t('modelBrowser.allFamilies', 'All')}
            </FamilyChip>
            {picker.hasFavorites && (
              <FamilyChip
                type="button"
                $active={picker.selectedFamily === 'favorites'}
                aria-pressed={picker.selectedFamily === 'favorites'}
                onClick={() => picker.setSelectedFamily('favorites')}
              >
                {t('familySidebar.favorites', 'Favorites')}
              </FamilyChip>
            )}
            {picker.hasRecents && (
              <FamilyChip
                type="button"
                $active={picker.selectedFamily === 'recents'}
                aria-pressed={picker.selectedFamily === 'recents'}
                onClick={() => picker.setSelectedFamily('recents')}
              >
                {t('familySidebar.recent', 'Recent')}
              </FamilyChip>
            )}
            {picker.uniqueFamilies.map((family) => (
              <FamilyChip
                key={family}
                type="button"
                $active={picker.selectedFamily === family}
                aria-pressed={picker.selectedFamily === family}
                onClick={() => picker.setSelectedFamily(family)}
              >
                {familyLabel(family)}
              </FamilyChip>
            ))}
          </FamilyRail>
        )}

        {/* List */}
        <ListArea>
          {isHf ? (
            /* HuggingFace results */
            <>
              {hfIsSearching && (
                <EmptyMsg>
                  <Spinner size={20} />
                  {t('modelBrowser.searching', 'Searching...')}
                </EmptyMsg>
              )}
              {!hfIsSearching && hfModels.length === 0 && picker.searchQuery && (
                <EmptyMsg>{t('modelBrowser.noResultsFound', 'No results found')}</EmptyMsg>
              )}
              {!hfIsSearching && hfModels.length > 0 && (
                <>
                  {!picker.searchQuery.trim() && (
                    <SectionHeader>{t('modelBrowser.trending', 'Trending')}</SectionHeader>
                  )}
                  <ListCard>
                    {hfLocal.map((m) => renderHfItem(m))}
                  </ListCard>
                  {onHfLoadMore && (
                    <MoreRow>
                      <Button variant="outline" size="sm" onClick={onHfLoadMore}>
                        {t('modelBrowser.showMore', 'Show more')}
                      </Button>
                    </MoreRow>
                  )}
                  {hfBurst.length > 0 && (
                    <>
                      {burstSectionHeader}
                      <ListCard>
                        {hfBurst.map((m) => renderHfItem(m))}
                      </ListCard>
                    </>
                  )}
                </>
              )}
            </>
          ) : (
            /* Curated catalog groups */
            <>
              {mode === 'store-download' && localGroups.length > 0 && (
                <ListCard>
                  {localGroups.map(renderGroup)}
                </ListCard>
              )}
              {mode === 'store-download' && burstGroups.length > 0 && (
                <>
                  {burstSectionHeader}
                  <ListCard>
                    {burstGroups.map(renderGroup)}
                  </ListCard>
                </>
              )}
              {mode !== 'store-download' && picker.recommendedGroups.length > 0 && (
                <>
                  <SectionHeader>{t('modelBrowser.recommended', 'Recommended')}</SectionHeader>
                  <ListCard>
                    {picker.recommendedGroups.map(renderGroup)}
                  </ListCard>
                </>
              )}
              {mode !== 'store-download' && picker.otherGroups.length > 0 && (
                <>
                  {picker.recommendedGroups.length > 0 && (
                    <SectionHeader style={{ paddingTop: 12 }}>{t('modelBrowser.other', 'Other')}</SectionHeader>
                  )}
                  <ListCard>
                    {picker.otherGroups.map(renderGroup)}
                  </ListCard>
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
