import { useEffect, useState, type ReactNode } from 'react';
import styled from 'styled-components';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { operatorSession } from '../../auth/operatorSession';

/** Read-only workload assessment from core; never a provider offer. */
export interface InferenceAssessment {
  status: 'local' | 'external_required' | 'blocked' | 'insufficient_evidence';
  readiness: 'not_observed' | 'loaded_not_workload_verified';
  workload: { contextTokens: number; concurrentRequests: number };
  reasons: string[];
  assumptions: string[];
  uncertainty: string[];
  externalConfigurations: {
    backend: string;
    replicaCount: number;
    requestsPerReplica: number;
    contextPoolTokens: number;
    gpusPerNode: number;
    minimumMemoryBytesPerNode: number;
    estimatedMemoryBytesPerReplica: number;
    memoryPool: 'ram' | 'vram';
  }[];
}

/** Assessment follows the selected artifact, launch context and exclusions. */
export interface InferenceAssessmentPanelProps {
  modelId: string;
  contextTokens: number | null;
  excludedNodeIds: string[];
  /**
   * What to offer when the verdict is that external capacity is required,
   * given the assessed context; for example a remote offer from an installed
   * burst provider. The assessment itself never acquires anything.
   */
  externalAction?: (contextTokens: number) => ReactNode;
}

const Panel = styled.section`
  padding: 12px;
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surface};
  color: ${({ theme }) => theme.colors.text};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  h3 { margin: 0 0 10px; font-size: inherit; }
  p { margin: 8px 0; }
  details { margin-top: 8px; color: ${({ theme }) => theme.colors.textSecondary}; }
  summary { cursor: pointer; }
  ul { padding-left: 20px; }
`;

const Fields = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  label { display: flex; align-items: center; gap: 6px; }
  input {
    width: 90px;
    padding: 4px 6px;
    border: 1px solid ${({ theme }) => theme.colors.border};
    border-radius: 4px;
    background: ${({ theme }) => theme.colors.surface};
    color: inherit;
    font: inherit;
  }
`;

/** Display the core estimate and let operators assess concurrency without effects. */
export function InferenceAssessmentPanel({ modelId, contextTokens, excludedNodeIds, externalAction }: InferenceAssessmentPanelProps) {
  const { t } = useSkulkTranslation();
  const [planningContext, setPlanningContext] = useState('8192');
  const [concurrency, setConcurrency] = useState('1');
  const [assessment, setAssessment] = useState<InferenceAssessment | null>(null);
  const [failed, setFailed] = useState(false);
  const excludedKey = JSON.stringify([...excludedNodeIds].sort());
  const context = contextTokens ?? Number(planningContext);
  const concurrent = Number(concurrency);
  const valid = Number.isInteger(context) && context >= 256 && context <= 1_048_576
    && Number.isInteger(concurrent) && concurrent >= 1 && concurrent <= 16;

  useEffect(() => {
    setAssessment(null);
    setFailed(false);
    if (!valid) return;
    const controller = new AbortController();
    const read = async () => {
      const params = new URLSearchParams({ model_id: modelId, context_tokens: String(context), concurrent_requests: String(concurrent) });
      for (const node of JSON.parse(excludedKey) as string[]) params.append('excluded_node_ids', node);
      try {
        const response = await operatorSession.fetch(`/models/assessment?${params}`, { signal: controller.signal });
        if (!response.ok) throw new Error('Assessment unavailable');
        const result: InferenceAssessment = await response.json();
        if (!['local', 'external_required', 'blocked', 'insufficient_evidence'].includes(result.status)
          || result.workload?.contextTokens !== context || result.workload?.concurrentRequests !== concurrent) throw new Error('Invalid assessment');
        if (!controller.signal.aborted) { setAssessment(result); setFailed(false); }
      } catch {
        if (!controller.signal.aborted) { setAssessment(null); setFailed(true); }
      }
    };
    const timeout = setTimeout(() => void read(), 250);
    const interval = setInterval(() => void read(), 15000);
    return () => { controller.abort(); clearTimeout(timeout); clearInterval(interval); };
  }, [modelId, context, concurrent, excludedKey, valid]);

  const verdicts = {
    local: t('assessment.local', 'Estimated local fit'),
    external_required: t('assessment.externalRequired', 'Estimated external capacity required'),
    blocked: t('assessment.blocked', 'Planning blocked'),
    insufficient_evidence: t('assessment.insufficientEvidence', 'More evidence needed'),
  };

  return <Panel aria-label={t('assessment.title', 'Capacity estimate')}>
    <h3>{t('assessment.title', 'Capacity estimate')}</h3>
    <Fields>
      {contextTokens === null ? <label>
        {t('assessment.planningContext', 'Planning context')}
        <input type="number" min={256} max={1_048_576} aria-label={t('assessment.planningContext', 'Planning context')} value={planningContext} onChange={e => setPlanningContext(e.target.value)} />
      </label> : <span>{t('assessment.context', '{tokens} tokens per request', { tokens: context })}</span>}
      <label>
        {t('assessment.concurrentRequests', 'Simultaneous requests')}
        <input type="number" min={1} max={16} aria-label={t('assessment.concurrentRequests', 'Simultaneous requests')} value={concurrency} onChange={e => setConcurrency(e.target.value)} />
      </label>
    </Fields>
    <div role="status" aria-live="polite">
      {!valid ? <p>{t('assessment.invalidWorkload', 'Enter 256–1,048,576 context tokens and 1–16 simultaneous requests.')}</p>
        : failed ? <p>{t('assessment.unavailable', 'Assessment unavailable. Local fit is unknown.')}</p>
          : !assessment ? <p>{t('assessment.loading', 'Assessing this workload…')}</p>
            : <>
              <p><strong>{verdicts[assessment.status]}</strong></p>
              {assessment.reasons.slice(0, 1).map((reason, index) => <p key={index}>{reason}</p>)}
              {assessment.externalConfigurations.slice(0, 3).map((candidate, index) => <p key={index}>
                {t('assessment.configuration', '{count} remote replica(s), {backend}, at least {memory} GiB {pool} per node ({usable} GiB estimated workload).', {
                  count: candidate.replicaCount,
                  backend: candidate.backend,
                  memory: (Math.ceil(candidate.minimumMemoryBytesPerNode / 1024 ** 3 * 10) / 10).toFixed(1),
                  usable: (candidate.estimatedMemoryBytesPerReplica / 1024 ** 3).toFixed(1),
                  pool: candidate.memoryPool.toUpperCase(),
                })}
              </p>)}
              <p>{assessment.readiness === 'loaded_not_workload_verified'
                ? t('assessment.loaded', 'An idle loaded instance can be reused; this workload has not been verified.')
                : t('assessment.estimated', 'Memory feasibility is estimated; successful loading and serving are unverified.')}</p>
              <details><summary>{t('assessment.evidence', 'Assumptions and uncertainty')}</summary>
                <ul>{[...assessment.reasons.slice(1), ...assessment.assumptions, ...assessment.uncertainty].map((text, index) => <li key={index}>{text}</li>)}</ul>
              </details>
            </>}
    </div>
    <p>{t('assessment.noProvisioning', 'Recommendation only. No external capacity is acquired.')}</p>
    {assessment?.status === 'external_required' && externalAction ? externalAction(context) : null}
  </Panel>;
}
