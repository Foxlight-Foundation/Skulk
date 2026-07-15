import styled from 'styled-components';
import { CenteredSpinner } from '../common/Spinner';
import { useGetClusterPerformanceEnvelopesQuery } from '../../store/endpoints/observability';
import { useSkulkTranslation, type SkulkTranslate } from '../../i18n/tolgee';
import type {
  ConcurrencyBucketSummary,
  PerformanceEnvelopeSummary,
} from '../../types/performanceEnvelope';

/**
 * "Performance" tab for the observability panel: the observe-only
 * throughput-and-latency-versus-concurrency envelopes the fabric has measured,
 * per (hardware class × model × engine × quant), gathered across the cluster.
 *
 * Data only (adaptive concurrency, Phase 0) — nothing here changes serving
 * behavior; it shows how each instance performs as concurrency rises and marks
 * the measured throughput knee.
 */
const Wrap = styled.div`
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  gap: 12px;
  padding: 8px 4px;
`;

const Empty = styled.div`
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: 13px;
  padding: 16px;
  text-align: center;
`;

const NodeGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const NodeHeading = styled.div`
  font-size: 12px;
  font-weight: 700;
  color: ${({ theme }) => theme.colors.textMuted};
  text-transform: uppercase;
  letter-spacing: 0.04em;
`;

const Card = styled.div`
  border: 1px solid rgba(128, 128, 128, 0.25);
  border-radius: 6px;
  padding: 8px 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

const CardHeader = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 6px;
  font-size: 12px;
`;

const Model = styled.span`
  font-weight: 600;
`;

const Tag = styled.span`
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  padding: 1px 6px;
  border-radius: 4px;
  color: #fff;
  background: #557;
`;

const Muted = styled.span`
  color: ${({ theme }) => theme.colors.textMuted};
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
`;

const Th = styled.th`
  text-align: right;
  padding: 2px 6px;
  color: ${({ theme }) => theme.colors.textMuted};
  font-weight: 600;
  border-bottom: 1px solid rgba(128, 128, 128, 0.25);
  white-space: nowrap;
  &:first-child {
    text-align: left;
  }
`;

const Td = styled.td<{ $knee?: boolean }>`
  text-align: right;
  padding: 2px 6px;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
  font-weight: ${({ $knee }) => ($knee ? 700 : 400)};
  &:first-child {
    text-align: left;
  }
`;

const Failure = styled.div`
  color: ${({ theme }) => theme.colors.textMuted};
  font-size: 12px;
  font-style: italic;
`;

function fmt(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined ? '—' : value.toFixed(digits);
}

function EnvelopeCard({
  envelope,
  t,
}: {
  envelope: PerformanceEnvelopeSummary;
  t: SkulkTranslate;
}) {
  return (
    <Card>
      <CardHeader>
        <Model>{envelope.modelId}</Model>
        <Tag>{envelope.backend || t('performance.unknownBackend', 'unknown')}</Tag>
        <Muted>{envelope.hardwareClass}</Muted>
        {envelope.quantization ? <Muted>· {envelope.quantization}</Muted> : null}
        <Muted>
          · {envelope.observationCount} {t('performance.observationsUnit', 'obs')}
        </Muted>
      </CardHeader>
      <Table>
        <thead>
          <tr>
            <Th>{t('performance.col.concurrency', 'Concurrency')}</Th>
            <Th>{t('performance.col.requests', 'Reqs')}</Th>
            <Th>{t('performance.col.decodeTps', 'Decode tok/s')}</Th>
            <Th>{t('performance.col.aggregateTps', 'Aggregate tok/s')}</Th>
            <Th>{t('performance.col.ttftP50', 'TTFT p50 (s)')}</Th>
            <Th>{t('performance.col.ttftP90', 'TTFT p90 (s)')}</Th>
          </tr>
        </thead>
        <tbody>
          {envelope.buckets.map((bucket: ConcurrencyBucketSummary) => {
            const isKnee = bucket.concurrency === envelope.kneeConcurrency;
            return (
              <tr key={bucket.concurrency}>
                <Td $knee={isKnee}>
                  {bucket.concurrency}
                  {isKnee ? ` ${t('performance.kneeMark', '(knee)')}` : ''}
                </Td>
                <Td $knee={isKnee}>{bucket.requestCount}</Td>
                <Td $knee={isKnee}>{fmt(bucket.decodeTpsMean)}</Td>
                <Td $knee={isKnee}>{fmt(bucket.aggregateDecodeTps)}</Td>
                <Td $knee={isKnee}>{fmt(bucket.ttftSecondsP50, 2)}</Td>
                <Td $knee={isKnee}>{fmt(bucket.ttftSecondsP90, 2)}</Td>
              </tr>
            );
          })}
        </tbody>
      </Table>
    </Card>
  );
}

export function PerformanceTab() {
  const { t } = useSkulkTranslation();
  // Poll while the tab is mounted; the panel unmounts inactive tabs, so this
  // stops when the operator switches away.
  const { data, isLoading, isError } = useGetClusterPerformanceEnvelopesQuery(undefined, {
    pollingInterval: 5000,
  });

  if (isLoading) {
    return <CenteredSpinner />;
  }

  // A failed fetch must not masquerade as "no data yet"; surface it distinctly.
  if (isError) {
    return (
      <Wrap>
        <Empty>
          {t('performance.loadFailed', 'Could not load performance envelopes.')}
        </Empty>
      </Wrap>
    );
  }

  const nodesWithData = (data?.nodes ?? []).filter(
    (node) => !node.ok || (node.report?.envelopes.length ?? 0) > 0,
  );

  if (nodesWithData.length === 0) {
    return (
      <Wrap>
        <Empty>
          {t(
            'performance.empty',
            'No performance envelopes yet. Serve some requests and the fabric will start measuring its throughput-vs-concurrency curve per hardware, model, and engine.',
          )}
        </Empty>
      </Wrap>
    );
  }

  return (
    <Wrap>
      {nodesWithData.map((node) => (
        <NodeGroup key={node.nodeId}>
          <NodeHeading>{node.nodeId}</NodeHeading>
          {!node.ok ? (
            <Failure>
              {t('performance.nodeUnreachable', 'Unreachable')}: {node.error ?? 'unknown'}
            </Failure>
          ) : (
            (node.report?.envelopes ?? []).map((envelope) => (
              <EnvelopeCard
                key={`${envelope.hardwareClass}:${envelope.modelId}:${envelope.backend}:${envelope.quantization}`}
                envelope={envelope}
                t={t}
              />
            ))
          )}
        </NodeGroup>
      ))}
    </Wrap>
  );
}
