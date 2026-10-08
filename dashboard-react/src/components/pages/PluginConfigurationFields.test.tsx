import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { darkTheme } from '../../theme/theme';
import { exampleStudioSettingsSchema } from './exampleStudioSettings.fixture';
import { PluginConfigurationFields } from './PluginConfigurationFields';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({ t: (_key: string, fallback: string, params?: Record<string, unknown>) => fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? '')) }),
}));

let root: Root;
let host: HTMLDivElement;
let changes: Record<string, unknown>[];

// Holds the draft as the settings editor does, recording every change.
function Harness({ initial }: { initial: Record<string, unknown> }) {
  const [values, setValues] = useState(initial);
  return <PluginConfigurationFields schema={exampleStudioSettingsSchema} values={values} onChange={(next) => { changes.push(next); setValues(next); }} />;
}
function input(label: string): HTMLInputElement {
  const target = [...host.querySelectorAll('label')].find((item) => item.textContent === label);
  if (!target) throw new Error(`Missing field: ${label}`);
  return host.querySelector<HTMLInputElement>(`[id="${target.htmlFor}"]`)!;
}
// React tracks an input's value itself, so a test sets it through the native setter.
async function type(element: HTMLInputElement, text: string) {
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(element, text);
    element.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

beforeEach(async () => {
  changes = [];
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  await act(async () => { root.render(<ThemeProvider theme={darkTheme}><Harness initial={{}} /></ThemeProvider>); });
});
afterEach(async () => {
  await act(async () => { root.unmount(); });
  host.remove();
});

it('renders every field of the settings, optional ones included, with plain labels and help', () => {
  const labels = [...host.querySelectorAll('label')].map((item) => item.textContent);
  expect(labels).toEqual(['Skulk API URL', 'Default model', 'Render timeout (seconds)', 'Keep renders', 'Refine model', 'Refine timeout (seconds)', 'Page reach', 'Comfy URL']);
  expect(host.textContent).toContain('How long one render may run before it is stopped.');
  // An omitted key shows the default the plugin applies.
  expect(input('Skulk API URL').placeholder).toBe('http://127.0.0.1:52415');
  expect(input('Render timeout (seconds)').placeholder).toBe('86400');
  expect(input('Default model').placeholder).toBe('Optional');
  expect(input('Default model').value).toBe('');
  // The optional value keeps its scalar's constraints.
  expect(input('Comfy URL').maxLength).toBe(2048);
  expect(input('Render timeout (seconds)').min).toBe('60');
  expect(input('Render timeout (seconds)').max).toBe('86400');
  expect(input('Render timeout (seconds)').getAttribute('aria-describedby')).toBeTruthy();
});

it('submits an emptied optional value as null, and an ordinary value as typed', async () => {
  await type(input('Default model'), 'example/video-model');
  expect(changes.at(-1)).toEqual({ default_model: 'example/video-model' });
  await type(input('Default model'), '');
  expect(changes.at(-1)).toEqual({ default_model: null });
  await type(input('Keep renders'), '12');
  expect(changes.at(-1)).toEqual({ default_model: null, keep_renders: 12 });
  // A typed value can return to the plugin's default.
  const reset = [...host.querySelectorAll('button')].find((item) => item.textContent === 'Use default')!;
  await act(async () => { reset.click(); });
  expect(changes.at(-1)).toEqual({ default_model: null });
});
