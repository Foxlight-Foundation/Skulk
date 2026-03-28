import { useCallback, useEffect, useMemo, useRef } from 'react';
import { Marked } from 'marked';
import hljs from 'highlight.js';
import katex from 'katex';
import styled from 'styled-components';

export interface MarkdownContentProps {
  content: string;
  className?: string;
}

/* ================================================================
   Markdown + LaTeX processing
   ================================================================ */

interface ProcessedMarkdown {
  html: string;
}

function processMarkdown(content: string): ProcessedMarkdown {
  const mathExpressions = new Map<string, { content: string; displayMode: boolean }>();
  let mathCounter = 0;

  // --- LaTeX preprocessing ---
  function preprocessLaTeX(text: string): string {
    // Protect code blocks
    const codeBlocks: string[] = [];
    let processed = text.replace(/```[\s\S]*?```/g, (match) => {
      codeBlocks.push(match);
      return `%%CODE_PLACEHOLDER_${codeBlocks.length - 1}%%`;
    });

    // Remove LaTeX document commands
    processed = processed.replace(/\\(documentclass|usepackage|begin\{document\}|end\{document\}|maketitle|require)\{[^}]*\}/g, '');

    // Handle display math environments
    const displayEnvs = ['align', 'align\\*', 'equation', 'equation\\*', 'gather', 'gather\\*', 'multline', 'multline\\*', 'split', 'cases'];
    for (const env of displayEnvs) {
      const re = new RegExp(`\\\\begin\\{${env}\\}([\\s\\S]*?)\\\\end\\{${env}\\}`, 'g');
      processed = processed.replace(re, (_, inner) => {
        const key = `%%MATH_${mathCounter++}%%`;
        mathExpressions.set(key, { content: `\\begin{${env.replace('\\*', '*')}}${inner}\\end{${env.replace('\\*', '*')}}`, displayMode: true });
        return key;
      });
    }

    // Extract display math ($$...$$)
    processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (_, inner) => {
      const key = `%%MATH_${mathCounter++}%%`;
      mathExpressions.set(key, { content: inner.trim(), displayMode: true });
      return key;
    });

    // Extract inline math ($...$) — skip currency patterns
    processed = processed.replace(/(?<!\$)\$(?!\$)(?!\d)((?:[^$\\]|\\.)+?)\$/g, (_, inner) => {
      const key = `%%MATH_${mathCounter++}%%`;
      mathExpressions.set(key, { content: inner.trim(), displayMode: false });
      return key;
    });

    // Restore code blocks
    for (let i = 0; i < codeBlocks.length; i++) {
      processed = processed.replace(`%%CODE_PLACEHOLDER_${i}%%`, codeBlocks[i]);
    }

    return processed;
  }

  // --- Marked setup ---
  const marked = new Marked();
  marked.use({
    renderer: {
      code({ text, lang }: { text: string; lang?: string | null }) {
        let highlighted: string;
        try {
          highlighted = lang && hljs.getLanguage(lang)
            ? hljs.highlight(text, { language: lang }).value
            : hljs.highlightAuto(text).value;
        } catch {
          highlighted = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }
        const langLabel = lang || 'code';
        return `<div class="mc-code-block"><div class="mc-code-header"><span class="mc-code-lang">${langLabel}</span><button class="mc-copy-btn" data-code="${encodeURIComponent(text)}">Copy</button></div><pre><code class="hljs">${highlighted}</code></pre></div>`;
      },
      codespan({ text }: { text: string }) {
        return `<code class="mc-inline-code">${text}</code>`;
      },
    },
  });

  // Process
  const preprocessed = preprocessLaTeX(content);
  let html = marked.parse(preprocessed) as string;

  // Render math expressions
  for (const [key, { content: mathContent, displayMode }] of mathExpressions) {
    let rendered: string;
    try {
      rendered = katex.renderToString(mathContent, { displayMode, throwOnError: false });
    } catch {
      rendered = `<span class="mc-math-error">${displayMode ? '$$' : '$'}${mathContent}${displayMode ? '$$' : '$'}</span>`;
    }

    if (displayMode) {
      rendered = `<div class="mc-math-display">${rendered}</div>`;
    } else {
      rendered = `<span class="mc-math-inline">${rendered}</span>`;
    }
    html = html.replace(key, rendered);
  }

  return { html };
}

/* ================================================================
   Styles
   ================================================================ */

const Container = styled.div`
  line-height: 1.6;
  color: ${({ theme }) => theme.colors.text};
  font-size: ${({ theme }) => theme.fontSizes.md};

  /* Headings */
  h1, h2 { color: #FFD700; }
  h1 { font-size: 1.5em; margin: 16px 0 8px; }
  h2 { font-size: 1.3em; margin: 14px 0 6px; }
  h3, h4, h5, h6 { margin: 12px 0 4px; }

  /* Paragraphs */
  p { margin: 8px 0; }

  /* Links */
  a { color: #60a5fa; text-decoration: underline; }

  /* Lists */
  ul, ol { padding-left: 20px; margin: 8px 0; }
  li { margin: 4px 0; }

  /* Blockquotes */
  blockquote {
    border-left: 3px solid rgba(255, 215, 0, 0.5);
    background: rgba(255, 215, 0, 0.05);
    padding: 8px 12px;
    margin: 8px 0;
    color: rgba(255, 255, 255, 0.8);
  }

  /* Tables */
  table { width: 100%; border-collapse: collapse; margin: 8px 0; }
  th { background: rgba(255, 215, 0, 0.1); text-align: left; }
  th, td { border: 1px solid ${({ theme }) => theme.colors.border}; padding: 6px 10px; font-size: ${({ theme }) => theme.fontSizes.tableBody}; }

  /* Inline code */
  .mc-inline-code {
    background: rgba(255, 215, 0, 0.1);
    border: 1px solid rgba(255, 215, 0, 0.15);
    border-radius: 3px;
    padding: 1px 5px;
    font-family: ${({ theme }) => theme.fonts.mono};
    font-size: 0.9em;
    color: #FFD700;
  }

  /* Code blocks */
  .mc-code-block {
    background: rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 215, 0, 0.2);
    border-radius: ${({ theme }) => theme.radii.md};
    margin: 12px 0;
    overflow: hidden;
  }

  .mc-code-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 12px;
    background: rgba(255, 215, 0, 0.05);
    border-bottom: 1px solid rgba(255, 215, 0, 0.1);
  }

  .mc-code-lang {
    font-size: ${({ theme }) => theme.fontSizes.label};
    font-family: ${({ theme }) => theme.fonts.mono};
    color: rgba(255, 215, 0, 0.6);
  }

  .mc-copy-btn {
    all: unset;
    cursor: pointer;
    font-size: ${({ theme }) => theme.fontSizes.label};
    font-family: ${({ theme }) => theme.fonts.mono};
    color: rgba(255, 255, 255, 0.4);
    transition: color 0.15s;
    &:hover { color: #FFD700; }
  }

  pre {
    margin: 0;
    padding: 12px;
    overflow-x: auto;
    font-family: ${({ theme }) => theme.fonts.mono};
    font-size: ${({ theme }) => theme.fontSizes.tableBody};
    line-height: 1.5;
  }

  code.hljs { background: transparent; }

  /* Highlight.js dark theme overrides */
  .hljs-keyword, .hljs-selector-tag { color: #c084fc; }
  .hljs-string, .hljs-attr { color: #fbbf24; }
  .hljs-number, .hljs-literal { color: #4ade80; }
  .hljs-built_in, .hljs-function { color: #60a5fa; }
  .hljs-comment { color: #6b7280; font-style: italic; }
  .hljs-title { color: #f472b6; }
  .hljs-type { color: #2dd4bf; }
  .hljs-params { color: #e5e5e5; }

  /* Math */
  .mc-math-display {
    margin: 12px 0;
    padding: 12px;
    border: 1px solid rgba(255, 215, 0, 0.15);
    border-radius: ${({ theme }) => theme.radii.md};
    overflow-x: auto;
  }

  .mc-math-inline {
    display: inline;
  }

  .mc-math-error {
    color: #f87171;
    font-family: ${({ theme }) => theme.fonts.mono};
    font-size: ${({ theme }) => theme.fontSizes.sm};
    background: rgba(239, 68, 68, 0.1);
    padding: 2px 6px;
    border-radius: 3px;
  }

  /* KaTeX color overrides */
  .katex { color: rgba(229, 229, 229, 0.9); }
  .katex .delimsizing, .katex .mord { color: rgba(191, 191, 191, 0.75); }
`;

/* ================================================================
   Component
   ================================================================ */

export function MarkdownContent({ content, className }: MarkdownContentProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const processed = useMemo(() => processMarkdown(content), [content]);

  const handleClick = useCallback((e: MouseEvent) => {
    const target = e.target as HTMLElement;
    if (target.classList.contains('mc-copy-btn')) {
      const code = decodeURIComponent(target.getAttribute('data-code') || '');
      navigator.clipboard.writeText(code).then(() => {
        const original = target.textContent;
        target.textContent = '✓ Copied';
        setTimeout(() => { target.textContent = original; }, 2000);
      });
    }
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener('click', handleClick);
    return () => el.removeEventListener('click', handleClick);
  }, [handleClick]);

  return (
    <Container
      ref={containerRef}
      className={className}
      dangerouslySetInnerHTML={{ __html: processed.html }}
    />
  );
}
