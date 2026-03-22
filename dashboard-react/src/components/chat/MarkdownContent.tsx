/**
 * MarkdownContent
 *
 * Renders Markdown with:
 * - KaTeX for inline ($...$) and display ($$...$$) math
 * - highlight.js for syntax-highlighted code blocks with copy buttons
 * - LaTeX environment extraction (align, equation, etc.)
 * - Protection of code blocks from markdown re-parsing
 *
 * Ported from MarkdownContent.svelte. The preprocessing logic is verbatim;
 * the lifecycle converts $effect → useEffect, bind:this → ref + dangerouslySetInnerHTML.
 */
import React, { useRef, useEffect, useMemo } from 'react';
import styled, { createGlobalStyle } from 'styled-components';
import { marked } from 'marked';
import hljs from 'highlight.js';
import katex from 'katex';
import 'katex/dist/katex.min.css';

// ─── Highlight.js dark theme (minimal, custom) ─────────────────────────────────

const HljsStyles = createGlobalStyle`
  .hljs { color: #e2e8f0; }
  .hljs-comment, .hljs-quote { color: #718096; font-style: italic; }
  .hljs-keyword, .hljs-selector-tag { color: #f6ad55; font-weight: bold; }
  .hljs-string, .hljs-doctag { color: #68d391; }
  .hljs-number, .hljs-literal { color: #63b3ed; }
  .hljs-built_in, .hljs-type { color: #76e4f7; }
  .hljs-attr, .hljs-attribute { color: #fbb6ce; }
  .hljs-variable, .hljs-template-variable { color: #faf089; }
  .hljs-function, .hljs-title { color: #b794f4; }
  .hljs-tag { color: #fc8181; }
  .hljs-name { color: #68d391; }
  .hljs-selector-id, .hljs-selector-class { color: #fbb6ce; }
  .hljs-addition { background: rgba(104,211,145,0.15); }
  .hljs-deletion { background: rgba(252,129,129,0.15); }
`;

// ─── Styled components ────────────────────────────────────────────────────────

const Container = styled.div`
  font-size: 13px;
  line-height: 1.7;
  color: ${({ theme }) => theme.colors.foreground};
  overflow-wrap: break-word;
  word-break: break-word;

  p { margin: 0 0 10px; }
  p:last-child { margin-bottom: 0; }

  h1, h2, h3, h4, h5, h6 {
    color: ${({ theme }) => theme.colors.yellow};
    margin: 16px 0 8px;
    font-weight: 700;
    letter-spacing: 0.05em;
  }
  h1 { font-size: 1.4em; }
  h2 { font-size: 1.2em; }
  h3 { font-size: 1.05em; }

  ul, ol {
    margin: 8px 0;
    padding-left: 20px;
  }
  li { margin: 3px 0; }

  blockquote {
    margin: 10px 0;
    padding: 8px 12px;
    border-left: 3px solid ${({ theme }) => theme.colors.yellow};
    background: ${({ theme }) => theme.colors.mediumGray};
    border-radius: 0 ${({ theme }) => theme.radius.sm} ${({ theme }) => theme.radius.sm} 0;
    font-style: italic;
    color: ${({ theme }) => theme.colors.lightGray};
  }

  hr {
    border: none;
    border-top: 1px solid ${({ theme }) => theme.colors.border};
    margin: 16px 0;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 12px;
  }
  th, td {
    padding: 6px 10px;
    border: 1px solid ${({ theme }) => theme.colors.border};
    text-align: left;
  }
  th {
    background: ${({ theme }) => theme.colors.mediumGray};
    color: ${({ theme }) => theme.colors.yellow};
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  tr:nth-child(even) { background: oklch(0.16 0 0 / 0.4); }

  a {
    color: ${({ theme }) => theme.colors.yellow};
    text-decoration: underline;
    text-underline-offset: 2px;
    &:hover { color: ${({ theme }) => theme.colors.yellowGlow}; }
  }

  code.inline-code {
    background: ${({ theme }) => theme.colors.mediumGray};
    border: 1px solid ${({ theme }) => theme.colors.border};
    border-radius: ${({ theme }) => theme.radius.sm};
    padding: 1px 5px;
    font-family: ${({ theme }) => theme.fonts.mono};
    font-size: 0.88em;
    color: oklch(0.85 0.1 75);
  }

  .code-block-wrapper {
    margin: 10px 0;
    border-radius: ${({ theme }) => theme.radius.md};
    border: 1px solid ${({ theme }) => theme.colors.border};
    overflow: hidden;
  }
  .code-block-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 10px;
    background: ${({ theme }) => theme.colors.mediumGray};
    border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  }
  .code-language {
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: ${({ theme }) => theme.colors.lightGray};
  }
  .copy-code-btn {
    background: none;
    border: none;
    cursor: pointer;
    color: ${({ theme }) => theme.colors.lightGray};
    padding: 2px;
    display: flex;
    align-items: center;
    border-radius: 2px;
    transition: color 120ms ease;
    &:hover { color: ${({ theme }) => theme.colors.yellow}; }
  }
  pre {
    margin: 0;
    padding: 12px;
    overflow-x: auto;
    background: ${({ theme }) => theme.colors.darkGray};
    font-family: ${({ theme }) => theme.fonts.mono};
    font-size: 12px;
    line-height: 1.6;
  }

  /* KaTeX display */
  .katex-display {
    margin: 12px 0;
    overflow-x: auto;
    overflow-y: hidden;
  }

  /* Placeholder for unsupported LaTeX */
  .latex-diagram-placeholder {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: ${({ theme }) => theme.colors.mediumGray};
    border-radius: ${({ theme }) => theme.radius.sm};
    font-size: 11px;
    color: ${({ theme }) => theme.colors.lightGray};
  }
`;

// ─── Markdown configuration ────────────────────────────────────────────────────

marked.setOptions({ gfm: true, breaks: true });

const renderer = new marked.Renderer();

renderer.code = function ({ text, lang }: { text: string; lang?: string }) {
  const language = lang && hljs.getLanguage(lang) ? lang : 'plaintext';
  const highlighted = hljs.highlight(text, { language }).value;
  const encoded = encodeURIComponent(text);
  return `
    <div class="code-block-wrapper">
      <div class="code-block-header">
        <span class="code-language">${language}</span>
        <button type="button" class="copy-code-btn" data-code="${encoded}" title="Copy">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect width="14" height="14" x="8" y="8" rx="2"/>
            <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>
          </svg>
        </button>
      </div>
      <pre><code class="hljs language-${language}">${highlighted}</code></pre>
    </div>`;
};

renderer.codespan = function ({ text }: { text: string }) {
  return `<code class="inline-code">${text}</code>`;
};

marked.use({ renderer });

// ─── LaTeX preprocessing ───────────────────────────────────────────────────────

const MATH_PFX = 'MATHPLACEHOLDER';
const CODE_PFX = 'CODEPLACEHOLDER';
const HTML_PFX = 'HTMLPLACEHOLDER';

const MATH_ENVS = [
  'align', 'align\\*', 'equation', 'equation\\*', 'gather', 'gather\\*',
  'multline', 'multline\\*', 'eqnarray', 'eqnarray\\*',
  'array', 'matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'cases',
];

function preprocessLaTeX(
  text: string,
  mathExpressions: Map<string, { content: string; displayMode: boolean }>,
  htmlSnippets: Map<string, string>,
): string {
  let counter = { math: 0, html: 0 };
  mathExpressions.clear();
  htmlSnippets.clear();

  // Protect code blocks
  const codeBlocks: string[] = [];
  let processed = text.replace(/```[\s\S]*?```|`[^`]+`/g, (m) => {
    codeBlocks.push(m);
    return `${CODE_PFX}${codeBlocks.length - 1}END`;
  });

  // Strip LaTeX document commands
  processed = processed.replace(/\\documentclass(\[[^\]]*\])?\{[^}]*\}/g, '');
  processed = processed.replace(/\\usepackage(\[[^\]]*\])?\{[^}]*\}/g, '');
  processed = processed.replace(/\\begin\{document\}|\\end\{document\}|\\maketitle/g, '');
  processed = processed.replace(/\\title\{[^}]*\}|\\author\{[^}]*\}|\\date\{[^}]*\}/g, '');
  processed = processed.replace(/\$\\require\{[^}]*\}\$|\\require\{[^}]*\}/g, '');

  // Unsupported environments → placeholders
  processed = processed.replace(
    /\\begin\{tikzpicture\}[\s\S]*?\\end\{tikzpicture\}/g,
    () => {
      const ph = `${HTML_PFX}${counter.html}END`;
      htmlSnippets.set(ph, '<div class="latex-diagram-placeholder"><span>📐</span> Diagram</div>');
      counter.html++;
      return ph;
    },
  );
  processed = processed.replace(
    /\\begin\{figure\}[\s\S]*?\\end\{figure\}/g,
    () => {
      const ph = `${HTML_PFX}${counter.html}END`;
      htmlSnippets.set(ph, '<div class="latex-diagram-placeholder"><span>🖼️</span> Figure</div>');
      counter.html++;
      return ph;
    },
  );
  processed = processed
    .replace(/\\begin\{center\}|\\end\{center\}/g, '')
    .replace(/\\begin\{flushleft\}|\\end\{flushleft\}/g, '')
    .replace(/\\begin\{flushright\}|\\end\{flushright\}/g, '')
    .replace(/\\label\{[^}]*\}/g, '')
    .replace(/\\caption\{[^}]*\}/g, '');

  processed = processed.replace(/\\\$/g, 'ESCAPEDDOLLARPLACEHOLDER');

  // Math environments → placeholders
  for (const env of MATH_ENVS) {
    const cleanEnv = env.replace('\\*', '*');

    const wrappedRe = new RegExp(
      `\\$\\\\begin\\{${env}\\}(\\{[^}]*\\})?([\\s\\S]*?)\\\\end\\{${env}\\}\\$`,
      'g',
    );
    processed = processed.replace(wrappedRe, (_, args, content) => {
      const mc = `\\begin{${cleanEnv}}${args ?? ''}${content}\\end{${cleanEnv}}`;
      const ph = `${MATH_PFX}DISPLAY${counter.math}END`;
      mathExpressions.set(ph, { content: mc, displayMode: true });
      counter.math++;
      return ph;
    });

    const bareRe = new RegExp(
      `\\\\begin\\{${env}\\}(\\{[^}]*\\})?([\\s\\S]*?)\\\\end\\{${env}\\}`,
      'g',
    );
    processed = processed.replace(bareRe, (_, args, content) => {
      const mc = `\\begin{${cleanEnv}}${args ?? ''}${content}\\end{${cleanEnv}}`;
      const ph = `${MATH_PFX}DISPLAY${counter.math}END`;
      mathExpressions.set(ph, { content: mc, displayMode: true });
      counter.math++;
      return ph;
    });
  }

  // Display math: $$...$$
  processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (_, content) => {
    const ph = `${MATH_PFX}DISPLAY${counter.math}END`;
    mathExpressions.set(ph, { content: content.trim(), displayMode: true });
    counter.math++;
    return ph;
  });

  // Inline math: $...$
  processed = processed.replace(/\$([^$\n]+?)\$/g, (_, content) => {
    const ph = `${MATH_PFX}INLINE${counter.math}END`;
    mathExpressions.set(ph, { content: content.trim(), displayMode: false });
    counter.math++;
    return ph;
  });

  // Restore code blocks
  codeBlocks.forEach((block, i) => {
    processed = processed.replace(`${CODE_PFX}${i}END`, block);
  });

  return processed;
}

function renderMathPlaceholders(
  html: string,
  mathExpressions: Map<string, { content: string; displayMode: boolean }>,
  htmlSnippets: Map<string, string>,
): string {
  let result = html;

  mathExpressions.forEach(({ content, displayMode }, ph) => {
    const escaped = ph.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp(escaped, 'g');
    let rendered: string;
    try {
      rendered = katex.renderToString(content, {
        displayMode,
        throwOnError: false,
        trust: false,
      });
    } catch {
      rendered = `<code class="inline-code">${content}</code>`;
    }
    result = result.replace(re, rendered);
  });

  htmlSnippets.forEach((snippet, ph) => {
    const escaped = ph.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    result = result.replace(new RegExp(escaped, 'g'), snippet);
  });

  result = result.replace(/ESCAPEDDOLLARPLACEHOLDER/g, '$');
  return result;
}

function renderMarkdown(content: string): string {
  const mathExpressions = new Map<string, { content: string; displayMode: boolean }>();
  const htmlSnippets = new Map<string, string>();

  const preprocessed = preprocessLaTeX(content, mathExpressions, htmlSnippets);
  const html = marked.parse(preprocessed) as string;
  return renderMathPlaceholders(html, mathExpressions, htmlSnippets);
}

// ─── Component ─────────────────────────────────────────────────────────────────

export interface MarkdownContentProps {
  content: string;
  className?: string;
}

export const MarkdownContent: React.FC<MarkdownContentProps> = ({
  content,
  className,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  const processedHtml = useMemo(() => renderMarkdown(content), [content]);

  // Delegate copy-button clicks via event delegation
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleClick = (e: MouseEvent) => {
      const btn = (e.target as Element).closest<HTMLButtonElement>('.copy-code-btn');
      if (!btn) return;
      const code = decodeURIComponent(btn.dataset['code'] ?? '');
      void navigator.clipboard.writeText(code);
    };

    container.addEventListener('click', handleClick);
    return () => container.removeEventListener('click', handleClick);
  }, [processedHtml]);

  return (
    <>
      <HljsStyles />
      <Container
        ref={containerRef}
        className={className}
        dangerouslySetInnerHTML={{ __html: processedHtml }}
      />
    </>
  );
};
