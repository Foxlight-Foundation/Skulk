import { useCallback, useEffect, useRef, useSyncExternalStore } from 'react';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  /** 0 = persistent (manual dismiss only). */
  duration: number;
  createdAt: number;
}

export interface ToastInput {
  type: ToastType;
  message: string;
  /** If true, toast stays until manually dismissed. */
  persistent?: boolean;
  /** Auto-dismiss duration in ms. Defaults: success/info 4000, error/warning 6000. */
  duration?: number;
}

const DEFAULT_DURATIONS: Record<ToastType, number> = {
  success: 4000,
  info: 4000,
  warning: 6000,
  error: 6000,
};

/* ---- singleton store (module-level, shared across all hook consumers) ---- */

let toastList: Toast[] = [];
const timers = new Map<string, ReturnType<typeof setTimeout>>();
const listeners = new Set<() => void>();

function emit() {
  for (const l of listeners) l();
}

function generateId(): string {
  return `toast-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function addToast(input: ToastInput): string {
  const id = generateId();
  const duration = input.persistent ? 0 : (input.duration ?? DEFAULT_DURATIONS[input.type]);

  toastList = [...toastList, { id, type: input.type, message: input.message, duration, createdAt: Date.now() }];
  emit();

  if (duration > 0) {
    timers.set(id, setTimeout(() => dismissToast(id), duration));
  }

  return id;
}

export function dismissToast(id: string): void {
  const timer = timers.get(id);
  if (timer) { clearTimeout(timer); timers.delete(id); }
  toastList = toastList.filter((t) => t.id !== id);
  emit();
}

/* ---- React hook ---- */

function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

function getSnapshot() {
  return toastList;
}

export function useToast() {
  const toasts = useSyncExternalStore(subscribe, getSnapshot);
  return { toasts, addToast, dismissToast };
}
