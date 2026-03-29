import { useCallback, useEffect, useMemo, useState } from 'react';
import styled, { css } from 'styled-components';
import { FiX, FiInfo } from 'react-icons/fi';
import { MdPlayArrow } from 'react-icons/md';
import type { TopologyData } from '../../types/topology';
import { Button } from '../common/Button';

/* ── Types ────────────────────────────────────────────── */

export interface PlacementManagerProps {
  modelId: string;
  topology: TopologyData;
  open: boolean;
  onClose: () => void;
  onLaunch: (params: { modelId: string; sharding: string; instanceMeta: string; minNodes: number }) => void;
}

interface RawPreview {
  model_id: string;
  sharding: string;
  instance_meta: string;
  instance: unknown | null;
  memory_delta_by_node: Record<string, number> | null;
  error: string | null;
}

interface ComboStatus {
  available: boolean;
  error?: string;
  memoryDelta?: Record<string, number>;
  nodeIds?: string[];
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

function formatBytes(bytes: number): string {
  const gb = Math.abs(bytes) / 1e9;
  const sign = bytes >= 0 ? '+' : '-';
  return `${sign}${gb.toFixed(1)}GB`;
}

function comboKey(sharding: string, meta: string): keyof NodeCountOptions {
  const s = sharding === 'Tensor' ? 'tensor' : 'pipeline';
  const m = meta === 'MlxJaccl' ? 'jaccl' : 'ring';
  return `${s}_${m}` as keyof NodeCountOptions;
}

function extractNodeIds(instance: unknown): string[] {
  if (!instance || typeof instance !== 'object') return [];
  const inner = (instance as Record<string, unknown>).MlxRingInstance
    ?? (instance as Record<string, unknown>).MlxJacclInstance;
  if (!inner || typeof inner !== 'object') return [];
  const sa = (inner as Record<string, unknown>).shardAssignments;
  if (!sa || typeof sa !== 'object') return [];
  const ntr = (sa as Record<string, unknown>).nodeToRunner;
  if (!ntr || typeof ntr !== 'object') return [];
  return Object.keys(ntr as Record<string, unknown>);
}

/* ── Styles ───────────────────────────────────────────── */

const Overlay = styled.div`
  position: fixed;
  inset: 0;
  z-index: 100;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
`;

const Modal = styled.div`
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg};
  width: 520px;
  max-height: 80vh;
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

const NodeGrid = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
`;

const NodeCard = styled.div<{ $active: boolean }>`
  padding: 8px 12px;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid ${({ $active, theme }) => $active ? 'rgba(74, 222, 128, 0.4)' : theme.colors.border};
  background: ${({ $active }) => $active ? 'rgba(74, 222, 128, 0.06)' : 'transparent'};
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 100px;
  transition: all 0.15s;
`;

const NodeName = styled.div<{ $active: boolean }>`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  font-weight: 500;
  color: ${({ $active }) => $active ? '#4ade80' : 'rgba(255,255,255,0.7)'};
`;

const NodeMem = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const MemDelta = styled.span<{ $positive: boolean }>`
  color: ${({ $positive }) => $positive ? '#f59e0b' : '#4ade80'};
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
  background: ${({ $selected, $disabled }) =>
    $disabled ? 'rgba(255,255,255,0.02)' : $selected ? 'rgba(255, 215, 0, 0.08)' : 'transparent'};
  opacity: ${({ $disabled }) => $disabled ? 0.5 : 1};
  transition: all 0.15s;
  text-align: center;

  ${({ $disabled }) => !$disabled && css`
    &:hover {
      border-color: rgba(255, 215, 0, 0.4);
    }
  `}
`;

const OptionLabel = styled.div<{ $selected: boolean }>`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  font-weight: 500;
  color: ${({ $selected }) => $selected ? '#FFD700' : 'rgba(255,255,255,0.7)'};
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

/* ── Component ────────────────────────────────────────── */

export function PlacementManager({ modelId, topology, open, onClose, onLaunch }: PlacementManagerProps) {
  const [previews, setPreviews] = useState<RawPreview[]>([]);
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

    (async () => {
      try {
        const res = await fetch(`/instance/previews?model_id=${encodeURIComponent(modelId)}`);
        if (res.ok) {
          const data = await res.json();
          setPreviews(data.previews ?? data ?? []);
        }
      } catch { /* ignore */ }
      finally { setLoading(false); }
    })();
  }, [open, modelId]);

  // Group previews by node count
  const optionsByNodeCount = useMemo(() => {
    const map: Record<number, NodeCountOptions> = {};

    for (let n = 1; n <= totalNodes; n++) {
      map[n] = {
        pipeline_ring: { available: false, error: 'No preview available' },
        pipeline_jaccl: { available: false, error: 'No preview available' },
        tensor_ring: { available: false, error: 'No preview available' },
        tensor_jaccl: { available: false, error: 'No preview available' },
      };
    }

    for (const p of previews) {
      const nodeIds = p.instance ? extractNodeIds(p.instance) : [];
      const count = nodeIds.length || 1;
      const key = comboKey(p.sharding, p.instance_meta);

      if (!map[count]) {
        map[count] = {
          pipeline_ring: { available: false, error: 'No preview available' },
          pipeline_jaccl: { available: false, error: 'No preview available' },
          tensor_ring: { available: false, error: 'No preview available' },
          tensor_jaccl: { available: false, error: 'No preview available' },
        };
      }

      map[count][key] = p.error
        ? { available: false, error: p.error }
        : { available: true, memoryDelta: p.memory_delta_by_node ?? undefined, nodeIds };
    }

    return map;
  }, [previews, totalNodes]);

  // Current options for selected node count
  const currentOptions = optionsByNodeCount[minNodes];
  const currentCombo = currentOptions?.[comboKey(sharding, instanceMeta)];

  // Auto-select first available combo when node count changes
  useEffect(() => {
    if (!currentOptions) return;
    const key = comboKey(sharding, instanceMeta);
    if (currentOptions[key]?.available) return; // current selection still valid

    // Try to find a valid combo
    const priorities: (keyof NodeCountOptions)[] = ['pipeline_ring', 'tensor_ring', 'pipeline_jaccl', 'tensor_jaccl'];
    for (const k of priorities) {
      if (currentOptions[k]?.available) {
        setSharding(k.startsWith('tensor') ? 'Tensor' : 'Pipeline');
        setInstanceMeta(k.endsWith('jaccl') ? 'MlxJaccl' : 'MlxRing');
        return;
      }
    }
  }, [minNodes, currentOptions, sharding, instanceMeta]);

  // Nodes that would be used
  const activeNodeIds = useMemo(() => {
    return currentCombo?.nodeIds ?? [];
  }, [currentCombo]);

  const handleLaunch = useCallback(() => {
    onLaunch({ modelId, sharding, instanceMeta, minNodes });
    onClose();
  }, [modelId, sharding, instanceMeta, minNodes, onLaunch, onClose]);

  if (!open) return null;

  const canLaunch = currentCombo?.available ?? false;
  const pipelineRing = currentOptions?.pipeline_ring;
  const pipelineJaccl = currentOptions?.pipeline_jaccl;
  const tensorRing = currentOptions?.tensor_ring;
  const tensorJaccl = currentOptions?.tensor_jaccl;

  // Determine sharding/networking errors for callouts
  const shardingError = sharding === 'Tensor'
    ? tensorRing?.error ?? tensorJaccl?.error
    : pipelineRing?.error ?? pipelineJaccl?.error;
  const networkError = instanceMeta === 'MlxJaccl'
    ? (sharding === 'Pipeline' ? pipelineJaccl?.error : tensorJaccl?.error)
    : undefined;

  return (
    <Overlay onClick={onClose}>
      <Modal onClick={(e) => e.stopPropagation()}>
        <Header>
          <Title>Place <ModelName>{modelLabel(modelId)}</ModelName></Title>
          <CloseBtn onClick={onClose}><FiX size={18} /></CloseBtn>
        </Header>

        <Body>
          {loading ? (
            <Loading>Analyzing placement options...</Loading>
          ) : (
            <>
              {/* Node count slider */}
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

              {/* Node visualization */}
              <Section>
                <NodeGrid>
                  {Object.entries(topology?.nodes ?? {}).map(([nodeId, node]) => {
                    const active = activeNodeIds.includes(nodeId);
                    const mem = node.macmon_info?.memory;
                    const delta = currentCombo?.memoryDelta?.[nodeId];
                    return (
                      <NodeCard key={nodeId} $active={active}>
                        <NodeName $active={active}>
                          {node.friendly_name ?? nodeId.slice(0, 8)}
                        </NodeName>
                        <NodeMem>
                          {mem ? `${(mem.ram_usage / 1e9).toFixed(1)}/${(mem.ram_total / 1e9).toFixed(0)}GB` : '—'}
                          {active && delta != null && (
                            <> <MemDelta $positive={delta > 0}>{formatBytes(delta)}</MemDelta></>
                          )}
                        </NodeMem>
                      </NodeCard>
                    );
                  })}
                </NodeGrid>
              </Section>

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
        </Body>

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
      </Modal>
    </Overlay>
  );
}
