import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { darkTheme, lightTheme } from '../../theme/theme';
import { NetworkMesh } from './NetworkMesh';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
let root: Root;
let host: HTMLDivElement;
let media: MediaQueryList;
const frames = new Map<number, FrameRequestCallback>();
let frameId = 0;

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  media = Object.assign(new EventTarget(), {
    matches: false, media: '(prefers-reduced-motion: reduce)', onchange: null,
    addListener: vi.fn(), removeListener: vi.fn(),
  });
  vi.spyOn(window, 'matchMedia').mockReturnValue(media);
  vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
    frames.set(++frameId, callback);
    return frameId;
  });
  vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(id => { frames.delete(id); });
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  vi.restoreAllMocks();
  frames.clear();
});

function advanceFrame() {
  const [id, callback] = [...frames.entries()][0];
  frames.delete(id);
  callback(performance.now());
}

it.each([darkTheme, lightTheme])('draws drifting particles and stops scheduling on unmount in $name', async theme => {
  await act(async () => root.render(<ThemeProvider theme={theme}><NetworkMesh count={12} speed={1} /></ThemeProvider>));
  const canvas = host.querySelector('canvas')!;
  expect(canvas.getAttribute('aria-hidden')).toBe('true');
  const initial = canvas.toDataURL();
  advanceFrame();
  expect(canvas.toDataURL()).not.toBe(initial);
  expect(frames.size).toBe(1);
  await act(async () => root.unmount());
  expect(frames.size).toBe(0);
});

it('draws a still mesh for reduced motion and reacts to preference changes and resize', async () => {
  Object.defineProperty(media, 'matches', { value: true, writable: true });
  await act(async () => root.render(<ThemeProvider theme={darkTheme}><NetworkMesh /></ThemeProvider>));
  const canvas = host.querySelector('canvas')!;
  const initial = canvas.toDataURL();
  expect(frames.size).toBe(0);
  window.dispatchEvent(new Event('resize'));
  expect(canvas.toDataURL()).toBe(initial);
  expect(canvas.width).toBe(window.innerWidth * devicePixelRatio);
  Object.defineProperty(media, 'matches', { value: false });
  media.dispatchEvent(new Event('change'));
  expect(frames.size).toBe(1);
  const moving = canvas.toDataURL();
  advanceFrame();
  expect(canvas.toDataURL()).not.toBe(moving);
  Object.defineProperty(media, 'matches', { value: true });
  media.dispatchEvent(new Event('change'));
  expect(frames.size).toBe(0);
});
