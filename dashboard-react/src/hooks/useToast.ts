import { useCallback, useEffect, useRef, useSyncExternalStore } from 'react';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

/** Toast record stored in the module-level toast store. */
export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  /** 0 = persistent (manual dismiss only). */
  duration: number;
  createdAt: number;
}

/** Input accepted when creating a toast. */
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

/** Add a toast to the global toast store and return its generated id. */
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

/** Dismiss a toast immediately and clear any scheduled auto-dismiss timer. */
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

/** Subscribe React components to the shared toast store. */
export function useToast() {
  const toasts = useSyncExternalStore(subscribe, getSnapshot);
  return { toasts, addToast, dismissToast };
}
