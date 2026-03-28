import styled from 'styled-components';
import { RunningInstanceCard, type InstanceStatus } from '../cluster/RunningInstanceCard';

/* ── Types ────────────────────────────────────────────── */

export interface InstanceCardData {
  instanceId: string;
  modelId: string;
  sharding: 'Pipeline' | 'Tensor';
  instanceType: 'MlxRing' | 'MlxJaccl';
  nodeName: string;
  status: InstanceStatus;
  statusMessage?: string;
  loadProgress?: number;
}

export interface InstancePanelProps {
  instances: InstanceCardData[];
  onDelete?: (instanceId: string) => void;
  onChat?: (modelId: string) => void;
  className?: string;
}

/* ── Styles ───────────────────────────────────────────── */

const Panel = styled.aside`
  width: 340px;
  flex-shrink: 0;
  border-left: 1px solid ${({ theme }) => theme.colors.border};
  background: transparent;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const PanelHeader = styled.div`
  padding: 12px 16px;
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  font-weight: 600;
  color: ${({ theme }) => theme.colors.textSecondary};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const Count = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
  font-weight: 400;
`;

const CardList = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

/* ── Component ────────────────────────────────────────── */

export function InstancePanel({ instances, onDelete, onChat, className }: InstancePanelProps) {
  return (
    <Panel className={className}>
      <PanelHeader>
        Active Instances
        <Count>{instances.length}</Count>
      </PanelHeader>
      <CardList>
        {instances.map((inst) => (
          <RunningInstanceCard
            key={inst.instanceId}
            {...inst}
            onDelete={onDelete ? () => onDelete(inst.instanceId) : undefined}
            onChat={onChat ? () => onChat(inst.modelId) : undefined}
          />
        ))}
      </CardList>
    </Panel>
  );
}
