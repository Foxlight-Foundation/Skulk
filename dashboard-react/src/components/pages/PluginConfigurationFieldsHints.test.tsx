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

// A plugin's settings with one ordinary money field, one duration and the rest advanced.
const cleanupHost = { anyOf: [{ type: 'string' }, { type: 'null' }], default: null, title: 'Cleanup host', 'x-skulk-advanced': true };
const hintedSchema: Record<string, unknown> = {
  type: 'object',
  required: ['spending_limit_usd', 'authority_port'],
  properties: {
    spending_limit_usd: { type: 'number', default: 10, minimum: 1, 'x-skulk-unit': 'usd', description: 'The most all rentals together may cost.' },
    idle_minutes: { type: 'integer', default: 5, minimum: 1, 'x-skulk-unit': 'minutes' },
    authority_port: { type: 'integer', title: 'Authority port', 'x-skulk-advanced': true },
    cleanup_host: cleanupHost,
  },
};

let root: Root;
let host: HTMLDivElement;
let changes: Record<string, unknown>[];
let renders = 0;

function Harness({ schema, initial }: { schema: Record<string, unknown>; initial: Record<string, unknown> }) {
  const [values, setValues] = useState(initial);
  return <PluginConfigurationFields schema={schema} values={values} onChange={(next) => { changes.push(next); setValues(next); }} />;
}
// Each render starts a fresh draft, as reopening the settings does.
async function render(schema: Record<string, unknown>, initial: Record<string, unknown> = {}) {
  renders += 1;
  await act(async () => { root.render(<ThemeProvider theme={darkTheme}><Harness key={renders} schema={schema} initial={initial} /></ThemeProvider>); });
}
function input(label: string): HTMLInputElement {
  const target = [...host.querySelectorAll('label')].find((item) => item.textContent === label);
  if (!target) throw new Error(`Missing field: ${label}`);
  return host.querySelector<HTMLInputElement>(`[id="${target.htmlFor}"]`)!;
}
async function type(element: HTMLInputElement, text: string) {
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(element, text);
    element.dispatchEvent(new Event('input', { bubbles: true }));
  });
}
function advancedSection() { return host.querySelector('details'); }

beforeEach(() => {
  changes = [];
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(async () => { root.unmount(); });
  host.remove();
});

it('puts advanced fields in a closed section after the ordinary ones', async () => {
  await render(hintedSchema, { authority_port: 4100 });
  const labels = [...host.querySelectorAll('label')].map((item) => item.textContent);
  // The declared units are not repeated in labels made from the keys.
  expect(labels).toEqual(['Spending limit', 'Idle', 'Authority port', 'Cleanup host']);
  const section = advancedSection()!;
  expect(section.querySelector('summary')?.textContent).toBe('Advanced');
  expect(section.open).toBe(false);
  expect(section.contains(input('Authority port'))).toBe(true);
  expect(section.contains(input('Spending limit'))).toBe(false);
  await act(async () => { section.querySelector('summary')!.click(); });
  expect(advancedSection()!.open).toBe(true);
});

it('opens the section by itself when a required advanced field has no value', async () => {
  await render(hintedSchema);
  expect(advancedSection()!.open).toBe(true);
});

it('reopens the section, and keeps it open, while a required advanced field is emptied', async () => {
  await render(hintedSchema, { authority_port: 4100 });
  expect(advancedSection()!.open).toBe(false);
  await act(async () => { advancedSection()!.querySelector('summary')!.click(); });
  await type(input('Authority port'), '');
  expect(changes.at(-1)).toEqual({});
  expect(advancedSection()!.open).toBe(true);
  // Closing it while the value is still missing does not hide the field.
  await act(async () => { advancedSection()!.querySelector('summary')!.click(); });
  await act(async () => { await vi.waitFor(() => expect(advancedSection()!.open).toBe(true)); });
});

it('steps dollars by the cent unless a stored amount is finer', async () => {
  await render(hintedSchema, { spending_limit_usd: 1.005, authority_port: 4100 });
  const limit = input('Spending limit');
  expect(limit.value).toBe('1.005');
  // A cent step would make the saved amount invalid and block saving the form.
  expect(limit.step).toBe('any');
  expect(limit.validity.valid).toBe(true);
});

it('keeps every field collapsed when all of them are advanced', async () => {
  await render({ type: 'object', properties: { cleanup_host: cleanupHost } });
  expect(advancedSection()!.open).toBe(false);
  expect(advancedSection()!.contains(input('Cleanup host'))).toBe(true);
});

it('shows dollars with a dollar sign, to the cent', async () => {
  await render(hintedSchema, { spending_limit_usd: 25, authority_port: 4100 });
  const limit = input('Spending limit');
  expect(limit.value).toBe('25.00');
  expect(limit.step).toBe('0.01');
  const unit = host.querySelector(`[id="${limit.id}-unit"]`);
  expect(unit?.textContent).toBe('$');
  expect(limit.getAttribute('aria-describedby')).toContain(`${limit.id}-unit`);
  // What is typed stays as typed, and the value is stored as a plain number of dollars.
  await type(limit, '12.5');
  expect(limit.value).toBe('12.5');
  expect(changes.at(-1)).toEqual({ spending_limit_usd: 12.5, authority_port: 4100 });
  await act(async () => { limit.dispatchEvent(new FocusEvent('focusout', { bubbles: true })); });
  expect(input('Spending limit').value).toBe('12.50');
  // An omitted amount shows the plugin's default the same way.
  await render(hintedSchema, { authority_port: 4100 });
  expect(input('Spending limit').placeholder).toBe('10.00');
});

it('shows a duration\'s unit after its control without converting it', async () => {
  await render(hintedSchema, { idle_minutes: 7, authority_port: 4100 });
  const idle = input('Idle');
  expect(idle.value).toBe('7');
  expect(host.querySelector(`[id="${idle.id}-unit"]`)?.textContent).toBe('minutes');
  await type(idle, '10');
  expect(changes.at(-1)).toEqual({ idle_minutes: 10, authority_port: 4100 });
});

it('renders a schema without hints exactly as before', async () => {
  await render(exampleStudioSettingsSchema);
  expect(advancedSection()).toBeNull();
  expect(host.textContent).not.toContain('$');
  const labels = [...host.querySelectorAll('label')].map((item) => item.textContent);
  expect(labels).toEqual(['Skulk API URL', 'Default model', 'Render timeout (seconds)', 'Keep renders', 'Refine model', 'Refine timeout (seconds)', 'Page reach', 'Comfy URL']);
  expect(input('Render timeout (seconds)').step).toBe('1');
});
