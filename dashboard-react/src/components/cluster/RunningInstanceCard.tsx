import styled, { keyframes, css, useTheme } from 'styled-components';
import { FiExternalLink } from 'react-icons/fi';
import { BsChatDotsFill } from 'react-icons/bs';
import type { Theme } from '../../theme';

/* ── Types ────────────────────────────────────────────── */

export type InstanceStatus =
  | 'loading'
  | 'warming_up'
  | 'ready'
  | 'running'
  | 'failed'
  | 'shutting_down';

export interface RunningInstanceCardProps {
  instanceId: string;
  modelId: string;
  sharding: 'Pipeline' | 'Tensor';
  instanceType: 'MlxRing' | 'MlxJaccl';
  nodeName: string;
  status: InstanceStatus;
  statusMessage?: string;
  /** 0–100, shown during loading */
  loadProgress?: number;
  onDelete?: () => void;
  onChat?: () => void;
  isEmbedding?: boolean;
  className?: string;
}

/* ── Status helpers ───────────────────────────────────── */

/** Build status display config from the active theme. Called per render so the
 *  status colors track theme switches. */
function buildStatusConfig(
  theme: Theme,
): Record<InstanceStatus, { label: string; color: string; glow: string; defaultMessage: string }> {
  return {
    loading:       { label: 'Loading',       color: theme.colors.gold,    glow: theme.colors.goldDim,    defaultMessage: 'Downloading model...' },
    warming_up:    { label: 'Warming Up',    color: theme.colors.gold,    glow: theme.colors.goldDim,    defaultMessage: 'Preparing for inference...' },
    ready:         { label: 'Ready',         color: theme.colors.healthy, glow: theme.colors.accentBg,   defaultMessage: 'Ready to chat!' },
    running:       { label: 'Running',       color: theme.colors.healthy, glow: theme.colors.accentBg,   defaultMessage: 'Processing inference...' },
    failed:        { label: 'Failed',        color: theme.colors.error,   glow: theme.colors.errorBg,    defaultMessage: 'Instance failed' },
    shutting_down: { label: 'Shutting Down', color: theme.colors.warning, glow: theme.colors.warningBg,  defaultMessage: 'Shutting down...' },
  };
}

function formatInstanceId(id: string): string {
  return id.slice(0, 8).toUpperCase();
}

function formatInstanceType(type: 'MlxRing' | 'MlxJaccl'): string {
  return type === 'MlxRing' ? 'MLX Ring' : 'MLX Jaccl';
}

function hfUrl(modelId: string): string | null {
  return modelId.includes('/') ? `https://huggingface.co/${modelId}` : null;
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

/* ── Styled components ────────────────────────────────── */

const Card = styled.div<{ $color: string; $glow: string }>`
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ $color }) => $color};
  border-radius: ${({ theme }) => theme.radii.md};
  box-shadow: 0 0 12px ${({ $glow }) => $glow}, inset 0 0 12px ${({ $glow }) => $glow};
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

const NodeName = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textSecondary};
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
  nodeName,
  status,
  statusMessage,
  loadProgress,
  onDelete,
  onChat,
  isEmbedding,
  className,
}: RunningInstanceCardProps) {
  const theme = useTheme() as Theme;
  const baseCfg = buildStatusConfig(theme)[status];
  const cfg = isEmbedding && status === 'ready'
    ? { ...baseCfg, defaultMessage: 'Ready for embedding' }
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
        {onDelete && <DeleteBtn onClick={onDelete}>Delete</DeleteBtn>}
      </Header>

      <ModelIdText>{modelId}</ModelIdText>

      <MetaRow>
        <span>{sharding} &middot; {formatInstanceType(instanceType)}</span>
        <StatusBadge $color={cfg.color}>{cfg.label}</StatusBadge>
      </MetaRow>

      {link && (
        <HfLink href={link} target="_blank" rel="noopener noreferrer">
          Hugging Face <FiExternalLink size={11} />
        </HfLink>
      )}

      <NodeName>{nodeName}</NodeName>

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
            <BsChatDotsFill size={14} /> Chat
          </ChatBtn>
        )}
      </Footer>
    </Card>
  );
}
