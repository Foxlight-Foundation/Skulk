import styled, { keyframes, css, useTheme } from 'styled-components';
import { FiExternalLink, FiCheckCircle, FiXCircle, FiLoader, FiClock } from 'react-icons/fi';
import { BsChatDotsFill } from 'react-icons/bs';
import { InfoTooltip } from '../common/InfoTooltip';
import type { Theme } from '../../theme';
import { useSkulkTranslation, type SkulkTranslate } from '../../i18n/tolgee';

/* ── Types ────────────────────────────────────────────── */

export type InstanceStatus =
  | 'loading'
  | 'warming_up'
  | 'ready'
  | 'running'
  | 'failed'
  | 'shutting_down';

/** A single node's runner phase, collapsed from the runner lifecycle
 *  (idle -> connecting -> connected -> loading -> loaded -> warming up -> ready)
 *  into the categories the per-node status line renders. */
export type NodeRunnerState = 'ready' | 'loading' | 'pending' | 'failed' | 'stopping';

export interface InstanceNodeStatus {
  /** Friendly node name (e.g. "kite2"). */
  name: string;
  state: NodeRunnerState;
}

export interface RunningInstanceCardProps {
  instanceId: string;
  modelId: string;
  sharding: 'Pipeline' | 'Tensor';
  instanceType: 'MlxRing' | 'MlxJaccl';
  /** Serving engine: MLX (in-process), in-process llama.cpp, or the served
   *  llama-server. Drives the type label so a GGUF/served instance is not
   *  mislabelled as an MLX ring. */
  engine: 'mlx' | 'llama_cpp' | 'served';
  /** Per-node placement status: one entry per node the instance is placed on
   *  (all pipeline / tensor ranks), each with its runner's current phase, so the
   *  card shows which node is the laggard rather than a single aggregate. */
  nodeStatuses: InstanceNodeStatus[];
  status: InstanceStatus;
  statusMessage?: string;
  /** 0–100, shown during loading */
  loadProgress?: number;
  onDelete?: () => void;
  onChat?: () => void;
  isEmbedding?: boolean;
  /** Speculative-decoding status from the model card's runtime section:
   *  shown as a badge when the card declares an MTP sidecar or assistant
   *  drafter and the placement allows it (#254). */
  speculation?: { kind: 'sidecar' | 'assistant'; depth: number };
  className?: string;
}

/* ── Status helpers ───────────────────────────────────── */

/** Build status display config from the active theme. Called per render so the
 *  status colors track theme switches. */
function buildStatusConfig(
  theme: Theme,
  t: SkulkTranslate,
): Record<InstanceStatus, { label: string; color: string; glow: string; defaultMessage: string }> {
  return {
    loading:       { label: t('instance.status.loading', 'Loading'),       color: theme.colors.gold,    glow: theme.colors.goldDim,    defaultMessage: t('instance.status.loadingMessage', 'Downloading model...') },
    warming_up:    { label: t('instance.status.warmingUp', 'Warming Up'),    color: theme.colors.gold,    glow: theme.colors.goldDim,    defaultMessage: t('instance.status.warmingUpMessage', 'Preparing for inference...') },
    ready:         { label: t('instance.status.ready', 'Ready'),         color: theme.colors.healthy, glow: theme.colors.accentBg,   defaultMessage: t('instance.status.readyMessage', 'Ready to chat!') },
    running:       { label: t('instance.status.running', 'Running'),       color: theme.colors.healthy, glow: theme.colors.accentBg,   defaultMessage: t('instance.status.runningMessage', 'Processing inference...') },
    failed:        { label: t('instance.status.failed', 'Failed'),        color: theme.colors.error,   glow: theme.colors.errorBg,    defaultMessage: t('instance.status.failedMessage', 'Instance failed') },
    shutting_down: { label: t('instance.status.shuttingDown', 'Shutting Down'), color: theme.colors.warning, glow: theme.colors.warningBg,  defaultMessage: t('instance.status.shuttingDownMessage', 'Shutting down...') },
  };
}

function formatInstanceId(id: string): string {
  return id.slice(0, 8).toUpperCase();
}

/** The engine/topology label under the model name. MLX shows its sharding +
 *  ring/jaccl transport; the llama.cpp engines are single-node, so they show the
 *  engine name instead of an MLX-specific ring label. */
function formatEngineLabel(
  engine: 'mlx' | 'llama_cpp' | 'served',
  sharding: 'Pipeline' | 'Tensor',
  instanceType: 'MlxRing' | 'MlxJaccl',
  t: SkulkTranslate,
): string {
  if (engine === 'served') return t('placement.served', 'Served (llama.cpp)');
  if (engine === 'llama_cpp') return t('placement.llamaCpp', 'llama.cpp');
  const shard = sharding === 'Pipeline' ? t('common.pipeline', 'Pipeline') : t('common.tensor', 'Tensor');
  const transport = instanceType === 'MlxRing' ? t('placement.mlxRing', 'MLX Ring') : t('placement.mlxJaccl', 'MLX Jaccl');
  return `${shard} · ${transport}`;
}

function hfUrl(modelId: string): string | null {
  return modelId.includes('/') ? `https://huggingface.co/${modelId}` : null;
}

/** Icon + colour + spin for a single node's runner phase. */
function nodeStateVisual(
  state: NodeRunnerState,
  theme: Theme,
): { Icon: typeof FiCheckCircle; color: string; spin: boolean } {
  switch (state) {
    case 'ready': return { Icon: FiCheckCircle, color: theme.colors.healthy, spin: false };
    case 'failed': return { Icon: FiXCircle, color: theme.colors.error, spin: false };
    case 'stopping': return { Icon: FiClock, color: theme.colors.warning, spin: false };
    case 'pending': return { Icon: FiClock, color: theme.colors.textMuted, spin: false };
    case 'loading':
    default: return { Icon: FiLoader, color: theme.colors.gold, spin: true };
  }
}

function nodeStateLabel(state: NodeRunnerState, t: SkulkTranslate): string {
  switch (state) {
    case 'ready': return t('instance.node.ready', 'ready');
    case 'failed': return t('instance.node.failed', 'failed');
    case 'stopping': return t('instance.node.stopping', 'stopping');
    case 'pending': return t('instance.node.pending', 'pending');
    case 'loading':
    default: return t('instance.node.loading', 'loading');
  }
}

/* ── Animations ───────────────────────────────────────── */

const pulse = keyframes`
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
`;

const progressStripe = keyframes`
  0% { background-position: 0 0; }
  100% { background-position: 20px 0; }
`;

const spin = keyframes`
  to { transform: rotate(360deg); }
`;

/* ── Styled components ────────────────────────────────── */

const Card = styled.div<{ $color: string; $glow: string }>`
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ $color }) => $color};
  border-radius: ${({ theme }) => theme.radii.md};
  box-shadow: 0 0 6px ${({ $glow }) => $glow};
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 280px;
  max-width: 380px;
  font-family: ${({ theme }) => theme.fonts.body};
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
`;

const IdGroup = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const StatusDot = styled.span<{ $color: string }>`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${({ $color }) => $color};
  flex-shrink: 0;
  animation: ${pulse} 1.5s ease-in-out infinite;
`;

const InstanceIdText = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const DeleteBtn = styled.button`
  all: unset;
  cursor: pointer;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.error};
  border: 1px solid ${({ theme }) => theme.colors.error};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 2px 8px;
  transition: all 0.15s;

  &:hover {
    background: ${({ theme }) => theme.colors.errorBg};
  }
`;

const ModelIdText = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  color: ${({ theme }) => theme.colors.text};
  font-weight: 500;
  word-break: break-all;
`;

const MetaRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const StatusBadge = styled.span<{ $color: string }>`
  font-size: 10px;
  font-weight: 600;
  color: ${({ $color }) => $color};
  background: ${({ $color }) => $color}1a;
  border: 1px solid ${({ $color }) => $color}40;
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 1px 6px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
`;

const HfLink = styled.a`
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
  text-decoration: none;
  transition: color 0.15s;

  &:hover {
    color: ${({ theme }) => theme.colors.text};
  }
`;

const NodeRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 3px 10px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const NodeChip = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 4px;
  white-space: nowrap;

  svg { flex-shrink: 0; }
  svg.spin { animation: ${spin} 0.9s linear infinite; }
`;

const StatusLabel = styled.div<{ $color: string }>`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-weight: 700;
  color: ${({ $color }) => $color};
`;

const StatusMessage = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  font-style: italic;
`;

const Footer = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const ChatBtn = styled.button`
  all: unset;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: ${({ theme }) => theme.fontSizes.xs};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.healthy};
  border: 1px solid rgba(74, 222, 128, 0.3);
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 3px 10px;
  transition: all 0.15s;

  &:hover {
    background: rgba(74, 222, 128, 0.12);
    border-color: rgba(74, 222, 128, 0.5);
  }
`;

const ProgressTrack = styled.div`
  width: 100%;
  height: 4px;
  background: ${({ theme }) => theme.colors.borderLight};
  border-radius: 2px;
  overflow: hidden;
`;

const ProgressFill = styled.div<{ $pct: number; $color: string }>`
  height: 100%;
  width: ${({ $pct }) => $pct}%;
  background: ${({ $color }) => $color};
  border-radius: 2px;
  transition: width 0.3s ease-out;
  ${({ $pct }) =>
    $pct < 100 &&
    css`
      background-image: linear-gradient(
        45deg,
        ${({ theme }) => theme.colors.border} 25%,
        transparent 25%,
        transparent 50%,
        ${({ theme }) => theme.colors.border} 50%,
        ${({ theme }) => theme.colors.border} 75%,
        transparent 75%
      );
      background-size: 20px 20px;
      animation: ${progressStripe} 0.6s linear infinite;
    `}
`;

/* ── Component ────────────────────────────────────────── */

export function RunningInstanceCard({
  instanceId,
  modelId,
  sharding,
  instanceType,
  engine,
  nodeStatuses,
  status,
  statusMessage,
  loadProgress,
  onDelete,
  onChat,
  isEmbedding,
  speculation,
  className,
}: RunningInstanceCardProps) {
  const { t } = useSkulkTranslation();
  const theme = useTheme() as Theme;
  const baseCfg = buildStatusConfig(theme, t)[status];
  const cfg = isEmbedding && status === 'ready'
    ? { ...baseCfg, defaultMessage: t('instance.status.readyForEmbedding', 'Ready for embedding') }
    : baseCfg;
  const link = hfUrl(modelId);
  const showProgress = (status === 'loading' || status === 'warming_up') && loadProgress != null;
  const canChat = (status === 'ready' || status === 'running') && !isEmbedding;

  return (
    <Card $color={cfg.color} $glow={cfg.glow} className={className}>
      <Header>
        <IdGroup>
          <StatusDot $color={cfg.color} />
          <InstanceIdText>{formatInstanceId(instanceId)}</InstanceIdText>
        </IdGroup>
        {onDelete && <DeleteBtn onClick={onDelete}>{t('common.delete', 'Delete')}</DeleteBtn>}
      </Header>

      <ModelIdText>{modelId}</ModelIdText>

      <MetaRow>
        <span>
          {formatEngineLabel(engine, sharding, instanceType, t)}
        </span>
        <StatusBadge $color={cfg.color}>{cfg.label}</StatusBadge>
        {speculation && (
          <InfoTooltip
            content={t(
              'instance.speculationTooltip',
              'Speculative decoding active: {kind}, draft depth {depth}',
              {
                kind: speculation.kind === 'assistant'
                  ? t('instance.speculation.assistantDrafter', 'assistant drafter')
                  : t('instance.speculation.mtpSidecar', 'MTP sidecar'),
                depth: speculation.depth,
              },
            )}
          >
            <StatusBadge $color={theme.colors.accent}>
              {t('instance.speculationDepthBadge', 'MTP D{depth}', { depth: speculation.depth })}
            </StatusBadge>
          </InfoTooltip>
        )}
      </MetaRow>

      {link && (
        <HfLink href={link} target="_blank" rel="noopener noreferrer">
          {t('common.huggingFace', 'Hugging Face')} <FiExternalLink size={11} />
        </HfLink>
      )}

      <NodeRow>
        {nodeStatuses.map((n) => {
          const v = nodeStateVisual(n.state, theme);
          return (
            <NodeChip key={n.name} title={`${n.name}: ${nodeStateLabel(n.state, t)}`}>
              <v.Icon size={12} color={v.color} className={v.spin ? 'spin' : undefined} />
              {n.name}
            </NodeChip>
          );
        })}
      </NodeRow>

      {showProgress && (
        <ProgressTrack>
          <ProgressFill $pct={loadProgress!} $color={cfg.color} />
        </ProgressTrack>
      )}

      <Footer>
        <div>
          <StatusLabel $color={cfg.color}>{cfg.label.toUpperCase()}</StatusLabel>
          <StatusMessage>{statusMessage ?? cfg.defaultMessage}</StatusMessage>
        </div>
        {canChat && onChat && (
          <ChatBtn onClick={onChat}>
            <BsChatDotsFill size={14} /> {t('header.nav.chat', 'Chat')}
          </ChatBtn>
        )}
      </Footer>
    </Card>
  );
}
