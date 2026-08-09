import { useEffect, useState } from 'react';
import { useTheme } from 'styled-components';
import { FiExternalLink } from 'react-icons/fi';
import type { HuggingFaceModel } from '../../types/models';
import type { Theme } from '../../theme';
import { formatBytes } from '../../utils/format';
import { useSkulkTranslation } from '../../i18n/tolgee';

/**
 * Info-popover dossier for one Hugging Face result: the model card's own
 * prose answer to "what is this thing", fetched lazily when the popover
 * opens, above the structured facts (lineage, architecture, languages,
 * license, papers, stats). Mounted only while the tooltip is open, so the
 * card-summary request fires once per opened row and is server-cached.
 */
export interface HfModelDossierProps {
  model: HuggingFaceModel;
  author: string;
}

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`;
  return String(n);
}

function relationLabel(
  t: ReturnType<typeof useSkulkTranslation>['t'],
  relation: string,
): string | null {
  switch (relation) {
    case 'quantized': return t('hfDossier.relationQuantized', 'Quantized from');
    case 'finetune': return t('hfDossier.relationFinetune', 'Fine-tuned from');
    case 'merge': return t('hfDossier.relationMerge', 'Merged from');
    case 'adapter': return t('hfDossier.relationAdapter', 'Adapter for');
    default: return null;
  }
}

/** Render the dossier content for one result row's info popover. */
export function HfModelDossier({ model, author }: HfModelDossierProps) {
  const { t } = useSkulkTranslation();
  const theme = useTheme() as Theme;
  const [summary, setSummary] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    (async () => {
      try {
        const res = await fetch(
          `/models/card-summary?model_id=${encodeURIComponent(model.id)}`,
          { signal: controller.signal },
        );
        if (!res.ok) { setSummary(''); return; }
        const data = (await res.json()) as { summary?: string };
        setSummary(data.summary ?? '');
      } catch {
        if (!controller.signal.aborted) setSummary('');
      }
    })();
    return () => controller.abort();
  }, [model.id]);

  const hfUrl = `https://huggingface.co/${model.id}`;
  const relation = model.base_model_relation
    ? relationLabel(t, model.base_model_relation)
    : null;
  const sizeTags = model.tags.filter((tag) =>
    /^\d+[BMK]$|param|safetensor|gguf|mlx|fp16|bf16|\dbit|int[48]/i.test(tag),
  );

  const label = (text: string) => (
    <span style={{ color: theme.colors.textMuted }}>{text}</span>
  );

  return (
    <div style={{ minWidth: 260, maxWidth: 340 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
        <span style={{ color: theme.colors.gold, fontWeight: 600, overflowWrap: 'anywhere' }}>{model.id}</span>
        <a
          href={hfUrl}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: theme.colors.textMuted, display: 'flex', flexShrink: 0 }}
          title={t('common.openOnHuggingFace', 'Open on HuggingFace')}
        >
          <FiExternalLink size={14} />
        </a>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 12px' }}>
        {relation && model.base_model_repo && (
          <>
            {label(relation)}
            <a
              href={`https://huggingface.co/${model.base_model_repo}`}
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: theme.colors.gold, overflowWrap: 'anywhere' }}
            >
              {model.base_model_repo}
            </a>
          </>
        )}
        {model.architecture && (
          <>
            {label(t('hfDossier.architecture', 'Architecture'))}
            <span>{model.architecture}</span>
          </>
        )}
        {(model.languages?.length ?? 0) > 0 && (
          <>
            {label(t('hfDossier.languages', 'Languages'))}
            <span>{model.languages?.join(', ')}</span>
          </>
        )}
        {model.license && (
          <>
            {label(t('huggingFaceResult.license', 'License'))}
            <span>{model.license}</span>
          </>
        )}
        {(model.arxiv_ids?.length ?? 0) > 0 && (
          <>
            {label(t('hfDossier.papers', 'Papers'))}
            <span>
              {model.arxiv_ids?.map((id, i) => (
                <span key={id}>
                  {i > 0 && ', '}
                  <a
                    href={`https://arxiv.org/abs/${id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: theme.colors.gold }}
                  >
                    arXiv:{id}
                  </a>
                </span>
              ))}
            </span>
          </>
        )}
        {model.total_file_size ? (
          <>
            {label(t('hfDossier.downloadSize', 'Download size'))}
            <span>{formatBytes(model.total_file_size)}</span>
          </>
        ) : null}
        {label(t('huggingFaceResult.author', 'Author'))}
        <span>{author}</span>
        {label(t('huggingFaceResult.downloads', 'Downloads'))}
        <span>{formatCount(model.downloads)}</span>
        {label(t('huggingFaceResult.likes', 'Likes'))}
        <span>{formatCount(model.likes)}</span>
        {model.last_modified && (
          <>
            {label(t('huggingFaceResult.updated', 'Updated'))}
            <span>{new Date(model.last_modified).toLocaleDateString()}</span>
          </>
        )}
        {model.matched_file && (
          <>
            {label(t('huggingFaceResult.matchedFile', 'Matched file'))}
            <span style={{ overflowWrap: 'anywhere' }}>{model.matched_file}</span>
          </>
        )}
      </div>

      {/* The model card's own description: the "what is this thing" text.
        * Placed below the fixed-shape facts grid so its async arrival never
        * reflows the facts under the pointer. */}
      {summary !== '' && (
        <div style={{ marginTop: 8, borderTop: `1px solid ${theme.colors.borderLight}`, paddingTop: 6 }}>
          <div style={{ color: theme.colors.textMuted, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
            {t('hfDossier.abstract', 'Abstract')}
          </div>
          {summary === null ? (
            <div style={{ color: theme.colors.textMuted }}>
              {t('hfDossier.loadingCard', 'Reading model card…')}
            </div>
          ) : (
            <div style={{ color: theme.colors.textSecondary, lineHeight: 1.45 }}>
              {summary}
            </div>
          )}
        </div>
      )}

      {sizeTags.length > 0 && (
        <div style={{ marginTop: 8, borderTop: `1px solid ${theme.colors.borderLight}`, paddingTop: 6 }}>
          <div style={{ color: theme.colors.textMuted, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
            {t('huggingFaceResult.tags', 'Tags')}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {sizeTags.map((tag) => (
              <span key={tag} style={{ padding: '1px 6px', borderRadius: 3, background: theme.colors.borderLight, color: theme.colors.textSecondary }}>{tag}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
