import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled, { css } from 'styled-components';
import { FiX, FiInfo } from 'react-icons/fi';
import { MdPlayArrow } from 'react-icons/md';
import type { TopologyData } from '../../types/topology';
import type { PlacementPreview } from '../../types/models';
import { ModelCard } from '../models/ModelCard';
import { Button } from '../common/Button';

/* ── Types ────────────────────────────────────────────── */

export interface PlacementManagerProps {
  modelId: string;
  modelSizeMb?: number;
  topology: TopologyData;
  open: boolean;
  onClose: () => void;
  onLaunch: (params: { modelId: string; sharding: string; instanceMeta: string; minNodes: number }) => void;
  /** Embedding models: hide sharding/networking selectors and node slider */
  isEmbedding?: boolean;
}

interface ComboStatus {
  available: boolean;
  error?: string;
  preview?: PlacementPreview;
}

interface NodeCountOptions {
  pipeline_ring: ComboStatus;
  pipeline_jaccl: ComboStatus;
  tensor_ring: ComboStatus;
  tensor_jaccl: ComboStatus;
}

/* ── Helpers ──────────────────────────────────────────── */

function modelLabel(modelId: string): string {
  const parts = modelId.split('/');
  return parts[parts.length - 1];
}

function comboKey(sharding: string, meta: string): keyof NodeCountOptions {
  const s = sharding === 'Tensor' ? 'tensor' : 'pipeline';
  const m = meta === 'MlxJaccl' ? 'jaccl' : 'ring';
  return `${s}_${m}` as keyof NodeCountOptions;
}

function extractNodeCount(preview: PlacementPreview): number {
  if (!preview.instance || typeof preview.instance !== 'object') return 1;
  const inner = (preview.instance as Record<string, unknown>).MlxRingInstance
    ?? (preview.instance as Record<string, unknown>).MlxJacclInstance;
  if (!inner || typeof inner !== 'object') return 1;
  const sa = (inner as Record<string, unknown>).shardAssignments;
  if (!sa || typeof sa !== 'object') return 1;
  const ntr = (sa as Record<string, unknown>).nodeToRunner;
  if (!ntr || typeof ntr !== 'object') return 1;
  return Object.keys(ntr as Record<string, unknown>).length;
}

/* ── Styles ───────────────────────────────────────────── */

const Overlay = styled.div`
  position: fixed;
  inset: 0;
  z-index: 100;
  background: ${({ theme }) => theme.colors.overlay};
  display: flex;
  align-items: center;
  justify-content: center;
`;

const Modal = styled.div`
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg};
  width: 560px;
  max-height: 85vh;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
`;

const Title = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.md};
  font-family: ${({ theme }) => theme.fonts.body};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text};
`;

const ModelName = styled.span`
  color: ${({ theme }) => theme.colors.gold};
  font-weight: 500;
`;

const CloseBtn = styled.button`
  all: unset;
  cursor: pointer;
  color: ${({ theme }) => theme.colors.textMuted};
  &:hover { color: ${({ theme }) => theme.colors.text}; }
`;

const Body = styled.div`
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

const Section = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const SectionLabel = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.textSecondary};
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const SliderRow = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
`;

const Slider = styled.input`
  flex: 1;
  accent-color: ${({ theme }) => theme.colors.gold};
`;

const SliderValue = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.gold};
  min-width: 60px;
  text-align: right;
`;

const OptionRow = styled.div`
  display: flex;
  gap: 12px;
`;

const OptionBtn = styled.button<{ $selected: boolean; $disabled: boolean }>`
  all: unset;
  cursor: ${({ $disabled }) => $disabled ? 'not-allowed' : 'pointer'};
  flex: 1;
  padding: 10px 14px;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid ${({ $selected, $disabled, theme }) =>
    $disabled ? 'rgba(255,255,255,0.08)' : $selected ? theme.colors.goldDim : theme.colors.border};
  background: ${({ $selected, $disabled, theme}) => $disabled ? 'rgba(255,255,255,0.02)' : $selected ? theme.colors.goldBg : 'transparent'};
  opacity: ${({ $disabled }) => $disabled ? 0.5 : 1};
  transition: all 0.15s;
  text-align: center;

  ${({ $disabled }) => !$disabled && css`
    &:hover {
      border-color: ${({ theme }) => theme.colors.goldDim};
    }
  `}
`;

const OptionLabel = styled.div<{ $selected: boolean }>`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  font-weight: 500;
  color: ${({ $selected, theme}) => $selected ? theme.colors.gold : 'rgba(255,255,255,0.7)'};
`;

const OptionSub = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
  margin-top: 2px;
`;

const Callout = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 6px;
  padding: 8px 10px;
  border-radius: ${({ theme }) => theme.radii.sm};
  background: rgba(245, 158, 11, 0.08);
  border: 1px solid rgba(245, 158, 11, 0.2);
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: rgba(245, 158, 11, 0.9);
`;

const ErrorCallout = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 10px 12px;
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.errorBg};
  border: 1px solid rgba(239, 68, 68, 0.25);
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  color: rgba(248, 113, 113, 1);
`;

const Footer = styled.div`
  display: flex;
  justify-content: flex-end;
  padding: 16px 20px;
  border-top: 1px solid ${({ theme }) => theme.colors.border};
`;

const LaunchBtn = styled(Button)`
  gap: 6px;
`;

const Loading = styled.div`
  padding: 40px;
  text-align: center;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const CardWrapper = styled.div`
  pointer-events: none;
`;

/* ── Component ────────────────────────────────────────── */

export function PlacementManager({ modelId, modelSizeMb, topology, open, onClose, onLaunch, isEmbedding }: PlacementManagerProps) {
  const [previews, setPreviews] = useState<PlacementPreview[]>([]);
  const [loading, setLoading] = useState(false);
  const [minNodes, setMinNodes] = useState(1);
  const [sharding, setSharding] = useState<'Pipeline' | 'Tensor'>('Pipeline');
  const [instanceMeta, setInstanceMeta] = useState<'MlxRing' | 'MlxJaccl'>('MlxRing');

  const totalNodes = Object.keys(topology?.nodes ?? {}).length;

  // Fetch previews when modal opens
  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setPreviews([]);
    setMinNodes(1);
    setSharding('Pipeline');
    setInstanceMeta('MlxRing');

    const controller = new AbortController();
    (async () => {
      try {
        const res = await fetch(`/instance/previews?model_id=${encodeURIComponent(modelId)}`, {
          signal: controller.signal,
        });
        if (res.ok && !controller.signal.aborted) {
          const data = await res.json();
          setPreviews(data.previews ?? data ?? []);
        }
      } catch {
        if (controller.signal.aborted) return;
      }
      finally { if (!controller.signal.aborted) setLoading(false); }
    })();
    return () => controller.abort();
  }, [open, modelId]);

  // Group previews by node count
  const optionsByNodeCount = useMemo(() => {
    const map: Record<number, NodeCountOptions> = {};
    const empty: ComboStatus = { available: false, error: 'No preview available' };

    for (let n = 1; n <= totalNodes; n++) {
      map[n] = { pipeline_ring: { ...empty }, pipeline_jaccl: { ...empty }, tensor_ring: { ...empty }, tensor_jaccl: { ...empty } };
    }

    for (const p of previews) {
      const key = comboKey(p.sharding, p.instance_meta);

      if (p.instance && !p.error) {
        // Successful placement — map to the actual node count
        const count = extractNodeCount(p);
        if (map[count]) {
          map[count][key] = { available: true, preview: p };
        }
      } else if (p.error) {
        // Error preview — no instance, so apply error to all node counts
        // that don't already have a successful placement for this combo
        for (let n = 1; n <= totalNodes; n++) {
          if (map[n] && !map[n][key].available) {
            map[n][key] = { available: false, error: p.error };
          }
        }
      }
    }

    return map;
  }, [previews, totalNodes]);

  // Default to first valid node count + combo after previews load
  const defaultsSet = useRef(false);
  useEffect(() => {
    if (defaultsSet.current || Object.keys(optionsByNodeCount).length === 0) return;
    const combos: (keyof NodeCountOptions)[] = ['pipeline_ring', 'tensor_ring', 'pipeline_jaccl', 'tensor_jaccl'];
    for (let n = 1; n <= totalNodes; n++) {
      const opts = optionsByNodeCount[n];
      if (!opts) continue;
      for (const k of combos) {
        if (opts[k]?.available) {
          setMinNodes(n);
          setSharding(k.startsWith('tensor') ? 'Tensor' : 'Pipeline');
          setInstanceMeta(k.endsWith('jaccl') ? 'MlxJaccl' : 'MlxRing');
          defaultsSet.current = true;
          return;
        }
      }
    }
  }, [optionsByNodeCount, totalNodes]);

  // Reset defaults flag when modal reopens
  useEffect(() => {
    if (open) defaultsSet.current = false;
  }, [open]);

  const currentOptions = optionsByNodeCount[minNodes];
  const currentKey = comboKey(sharding, instanceMeta);
  const currentCombo = currentOptions?.[currentKey];
  const currentPreview = currentCombo?.available ? currentCombo.preview ?? null : null;

  // Auto-select first available combo when node count changes
  useEffect(() => {
    if (!currentOptions) return;
    if (currentOptions[currentKey]?.available) return;

    const priorities: (keyof NodeCountOptions)[] = ['pipeline_ring', 'tensor_ring', 'pipeline_jaccl', 'tensor_jaccl'];
    for (const k of priorities) {
      if (currentOptions[k]?.available) {
        setSharding(k.startsWith('tensor') ? 'Tensor' : 'Pipeline');
        setInstanceMeta(k.endsWith('jaccl') ? 'MlxJaccl' : 'MlxRing');
        return;
      }
    }
  }, [minNodes, currentOptions, currentKey]);

  const handleLaunch = useCallback(() => {
    if (isEmbedding) {
      onLaunch({ modelId, sharding: 'Pipeline', instanceMeta: 'MlxRing', minNodes: 1 });
    } else {
      onLaunch({ modelId, sharding, instanceMeta, minNodes });
    }
    onClose();
  }, [modelId, sharding, instanceMeta, minNodes, isEmbedding, onLaunch, onClose]);

  if (!open) return null;

  const embeddingComboAvailable = isEmbedding
    ? (optionsByNodeCount[1]?.pipeline_ring?.available ?? false)
    : false;
  const canLaunch = isEmbedding ? embeddingComboAvailable : (currentCombo?.available ?? false);
  const pipelineRing = currentOptions?.pipeline_ring;
  const pipelineJaccl = currentOptions?.pipeline_jaccl;
  const tensorRing = currentOptions?.tensor_ring;
  const tensorJaccl = currentOptions?.tensor_jaccl;

  const noComboAvailable = currentOptions
    && !pipelineRing?.available && !pipelineJaccl?.available
    && !tensorRing?.available && !tensorJaccl?.available;

  // Check if ANY node count has a valid placement at all
  const anyPlacementPossible = Object.values(optionsByNodeCount).some((opts) =>
    opts.pipeline_ring.available || opts.pipeline_jaccl.available
    || opts.tensor_ring.available || opts.tensor_jaccl.available,
  );

  // Find the most relevant error to show when nothing works
  const placementError = noComboAvailable
    ? (pipelineRing?.error ?? tensorRing?.error ?? pipelineJaccl?.error ?? tensorJaccl?.error ?? 'No valid placement found')
    : null;

  const shardingError = !noComboAvailable && sharding === 'Tensor'
    ? tensorRing?.error ?? tensorJaccl?.error
    : !noComboAvailable ? (pipelineRing?.error ?? pipelineJaccl?.error) : undefined;
  const networkError = !noComboAvailable && instanceMeta === 'MlxJaccl'
    ? (sharding === 'Pipeline' ? pipelineJaccl?.error : tensorJaccl?.error)
    : undefined;

  return (
    <Overlay onClick={onClose}>
      <Modal onClick={(e) => e.stopPropagation()}>
        <Header>
          <Title>Place <ModelName>{modelLabel(modelId)}</ModelName></Title>
          <CloseBtn onClick={onClose} aria-label="Close placement manager"><FiX size={18} /></CloseBtn>
        </Header>

        <Body>
          {loading ? (
            <Loading>Analyzing placement options...</Loading>
          ) : isEmbedding ? (
            /* ── Simplified embedding placement ─────────────── */
            <>
              <Section>
                <SectionLabel>Cluster Preview</SectionLabel>
                <CardWrapper>
                  <ModelCard
                    model={{ id: modelId, name: modelLabel(modelId), storage_size_megabytes: modelSizeMb }}
                    nodes={topology?.nodes ?? {}}
                    sharding="Pipeline"
                    runtime="MlxRing"
                    apiPreview={currentPreview}
                    hideActions
                  />
                </CardWrapper>
              </Section>

              {!anyPlacementPossible && placementError && (
                <ErrorCallout>
                  <FiInfo size={16} style={{ flexShrink: 0, marginTop: 1 }} />
                  {placementError}
                </ErrorCallout>
              )}

              {anyPlacementPossible && (
                <Callout>
                  <FiInfo size={14} style={{ flexShrink: 0, marginTop: 1 }} />
                  Embedding models run on a single node. The cluster will automatically select the best node.
                </Callout>
              )}
            </>
          ) : (
            <>
              {/* Cluster visualization via ModelCard */}
              <Section>
                <SectionLabel>Cluster Preview</SectionLabel>
                <CardWrapper>
                  <ModelCard
                    model={{ id: modelId, name: modelLabel(modelId), storage_size_megabytes: modelSizeMb }}
                    nodes={topology?.nodes ?? {}}
                    sharding={sharding}
                    runtime={instanceMeta}
                    apiPreview={currentPreview}
                    hideActions
                  />
                </CardWrapper>
              </Section>

              {/* Placement error — shown when nothing works at any node count */}
              {!anyPlacementPossible && placementError && (
                <ErrorCallout>
                  <FiInfo size={16} style={{ flexShrink: 0, marginTop: 1 }} />
                  {placementError}
                </ErrorCallout>
              )}

              {/* Node count slider — only when placement is possible */}
              {anyPlacementPossible && (
                <Section>
                  <SectionLabel>Nodes</SectionLabel>
                  <SliderRow>
                    <Slider
                      type="range"
                      min={1}
                      max={Math.max(totalNodes, 1)}
                      value={minNodes}
                      onChange={(e) => setMinNodes(Number(e.target.value))}
                    />
                    <SliderValue>{minNodes} of {totalNodes}</SliderValue>
                  </SliderRow>
                </Section>
              )}

              {/* Per-node-count error */}
              {anyPlacementPossible && noComboAvailable && placementError && (
                <ErrorCallout>
                  <FiInfo size={16} style={{ flexShrink: 0, marginTop: 1 }} />
                  {placementError}
                </ErrorCallout>
              )}

              {anyPlacementPossible && (
                <>
                  {/* Sharding */}
                  <Section>
                    <SectionLabel>Sharding</SectionLabel>
                    <OptionRow>
                      <OptionBtn
                        $selected={sharding === 'Pipeline'}
                        $disabled={!pipelineRing?.available && !pipelineJaccl?.available}
                        onClick={() => {
                          if (pipelineRing?.available || pipelineJaccl?.available) setSharding('Pipeline');
                        }}
                      >
                        <OptionLabel $selected={sharding === 'Pipeline'}>Pipeline</OptionLabel>
                        <OptionSub>Layers split across nodes</OptionSub>
                      </OptionBtn>
                      <OptionBtn
                        $selected={sharding === 'Tensor'}
                        $disabled={!tensorRing?.available && !tensorJaccl?.available}
                        onClick={() => {
                          if (tensorRing?.available || tensorJaccl?.available) setSharding('Tensor');
                        }}
                      >
                        <OptionLabel $selected={sharding === 'Tensor'}>Tensor</OptionLabel>
                        <OptionSub>Weights split across nodes</OptionSub>
                      </OptionBtn>
                    </OptionRow>
                    {shardingError && !canLaunch && (
                      <Callout><FiInfo size={14} style={{ flexShrink: 0, marginTop: 1 }} /> {shardingError}</Callout>
                    )}
                  </Section>

                  {/* Networking */}
                  <Section>
                    <SectionLabel>Networking</SectionLabel>
                    <OptionRow>
                      <OptionBtn
                        $selected={instanceMeta === 'MlxRing'}
                        $disabled={!(sharding === 'Pipeline' ? pipelineRing : tensorRing)?.available}
                        onClick={() => {
                          const combo = sharding === 'Pipeline' ? pipelineRing : tensorRing;
                          if (combo?.available) setInstanceMeta('MlxRing');
                        }}
                      >
                        <OptionLabel $selected={instanceMeta === 'MlxRing'}>MLX Ring</OptionLabel>
                        <OptionSub>Works over any network</OptionSub>
                      </OptionBtn>
                      <OptionBtn
                        $selected={instanceMeta === 'MlxJaccl'}
                        $disabled={!(sharding === 'Pipeline' ? pipelineJaccl : tensorJaccl)?.available}
                        onClick={() => {
                          const combo = sharding === 'Pipeline' ? pipelineJaccl : tensorJaccl;
                          if (combo?.available) setInstanceMeta('MlxJaccl');
                        }}
                      >
                        <OptionLabel $selected={instanceMeta === 'MlxJaccl'}>MLX Jaccl</OptionLabel>
                        <OptionSub>RDMA / Thunderbolt 5</OptionSub>
                      </OptionBtn>
                    </OptionRow>
                    {networkError && (
                      <Callout><FiInfo size={14} style={{ flexShrink: 0, marginTop: 1 }} /> {networkError}</Callout>
                    )}
                  </Section>
                </>
              )}
            </>
          )}
        </Body>

        {anyPlacementPossible && (
          <Footer>
            <LaunchBtn
              variant="primary"
              size="md"
              disabled={!canLaunch || loading}
              onClick={handleLaunch}
            >
              <MdPlayArrow size={18} /> Launch Model
            </LaunchBtn>
          </Footer>
        )}
      </Modal>
    </Overlay>
  );
}
