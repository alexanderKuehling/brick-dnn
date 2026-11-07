import { useState, type KeyboardEvent } from 'react';

export interface ChartCursorOptions {
  /** horizontal: ←/→ (and ↑/↓) step; vertical: ↑ previous, ↓ next; grid: rows of `columns`. */
  orientation?: 'horizontal' | 'vertical' | 'grid';
  columns?: number;
  /** Enter / Space on the active item. */
  onActivate?: (index: number) => void;
}

/**
 * The active item of a chart for pointer and keyboard (one tab stop per chart, arrow keys
 * move, Home/End jump, Escape clears, Enter activates).
 */
export function useChartCursor(count: number, options: ChartCursorOptions = {}) {
  const { orientation = 'horizontal', columns = 1, onActivate } = options;
  const [active, setActive] = useState<number | null>(null);
  const index = active !== null && active >= 0 && active < count ? active : null;

  const onKeyDown = (event: KeyboardEvent) => {
    if (count === 0) {
      return;
    }
    const last = count - 1;
    const current = index ?? 0;
    const clamp = (value: number) => Math.min(last, Math.max(0, value));
    const step = orientation === 'grid' ? columns : 1;
    const moves: Record<string, number | null> =
      orientation === 'horizontal'
        ? {
            ArrowLeft: clamp(current - 1),
            ArrowDown: clamp(current - 1),
            ArrowRight: clamp(current + 1),
            ArrowUp: clamp(current + 1),
          }
        : {
            ArrowLeft: clamp(current - 1),
            ArrowRight: clamp(current + 1),
            ArrowUp: clamp(current - step),
            ArrowDown: clamp(current + step),
          };
    const keys: Record<string, number | null> = {
      ...moves,
      Home: 0,
      End: last,
      PageUp: clamp(current - step * 10),
      PageDown: clamp(current + step * 10),
      Escape: null,
    };
    if (event.key in keys) {
      event.preventDefault();
      setActive(keys[event.key]);
    } else if ((event.key === 'Enter' || event.key === ' ') && index !== null && onActivate) {
      event.preventDefault();
      onActivate(index);
    }
  };

  return {
    active: index,
    setActive,
    onKeyDown,
    onFocus: () => setActive((value) => value ?? 0),
    onBlur: () => setActive(null),
  };
}
