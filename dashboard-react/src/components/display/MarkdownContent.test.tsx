import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import katex from 'katex';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { darkTheme } from '../../theme/theme';
import { MarkdownContent } from './MarkdownContent';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}));

let root: Root | null = null;
let container: HTMLDivElement | null = null;

async function renderMarkdown(content: string): Promise<HTMLDivElement> {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(
      <ThemeProvider theme={darkTheme}>
        <MarkdownContent content={content} />
      </ThemeProvider>,
    );
  });
  return container;
}

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  root = null;
  container = null;
  vi.restoreAllMocks();
});

describe('MarkdownContent', () => {
  it('renders headings, safe links, code blocks with copy button, and KaTeX math', async () => {
    const content = [
      '# Title',
      '',
      'See [Foxlight](https://foxlight.ai) and inline $x^2 + 1$.',
      '',
      '$$\\frac{\\sqrt{a^2 + b^2}}{c}$$',
      '',
      '```python',
      'print("ok")',
      '```',
    ].join('\n');

    const el = await renderMarkdown(content);

    expect(el.querySelector('h1')?.textContent).toBe('Title');
    const link = el.querySelector<HTMLAnchorElement>('a');
    expect(link?.getAttribute('href')).toBe('https://foxlight.ai');
    const copyBtn = el.querySelector<HTMLButtonElement>('.mc-copy-btn');
    expect(copyBtn).not.toBeNull();
    expect(decodeURIComponent(copyBtn?.getAttribute('data-code') ?? '')).toBe('print("ok")');
    expect(el.querySelectorAll('.katex').length).toBeGreaterThanOrEqual(2);
  });

  it('sanitizes executable HTML, event handlers, and unsafe link protocols', async () => {
    const payload = [
      '<script>window.__skulkXss = 1;</script>',
      '<img src="x" onerror="window.__skulkXss = 2">',
      '<svg onload="window.__skulkXss = 3"><foreignObject><script>window.__skulkXss = 4;</script></foreignObject></svg>',
      '<iframe src="https://evil.example"></iframe>',
      '<style>body { display: none; }</style>',
      '<form action="https://evil.example"><input name="x"></form>',
      '[bad-md](javascript:window.__skulkXss=5)',
      '<a href="javascript:window.__skulkXss=6">bad-html</a>',
      '<a href="data:text/html,<script>window.__skulkXss=7</script>">bad-data</a>',
    ].join('\n\n');

    const el = await renderMarkdown(payload);

    expect((window as unknown as Record<string, unknown>).__skulkXss).toBeUndefined();
    expect(el.querySelector('script, iframe, style, form, input, foreignObject')).toBeNull();
    for (const node of el.querySelectorAll('*')) {
      for (const attr of [...node.attributes]) {
        expect(attr.name.toLowerCase().startsWith('on')).toBe(false);
        if (attr.name.toLowerCase() === 'href' || attr.name.toLowerCase() === 'src') {
          expect(attr.value.trim().toLowerCase()).not.toMatch(/^(javascript|data|vbscript):/);
        }
      }
    }
  });

  it('escapes and sanitizes raw math content when KaTeX throws', async () => {
    vi.spyOn(katex, 'renderToString').mockImplementation(() => {
      throw new Error('forced KaTeX failure');
    });

    const el = await renderMarkdown('$$<img src=x onerror="window.__skulkMathXss=1">$$');

    expect((window as unknown as Record<string, unknown>).__skulkMathXss).toBeUndefined();
    expect(el.querySelector('img')).toBeNull();
    const fallback = el.querySelector('.mc-math-error');
    expect(fallback?.textContent).toContain('<img src=x onerror="window.__skulkMathXss=1">');
  });
});
