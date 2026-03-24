import { useEffect, useRef, useState } from 'react';

export interface Size {
  width: number;
  height: number;
}

export function useResizeObserver<T extends HTMLElement | SVGElement>(): [
  React.RefObject<T | null>,
  Size,
] {
  const ref = useRef<T | null>(null);
  const [size, setSize] = useState<Size>({ width: 0, height: 0 });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        const { width, height } = entry.contentRect;
        setSize((prev) =>
          prev.width === width && prev.height === height
            ? prev
            : { width, height },
        );
      }
    });

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return [ref, size];
}
