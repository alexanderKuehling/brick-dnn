import { useEffect, useRef, useState } from 'react';

/**
 * For a dialog that is mounted while it is open (and removed when it closes): when it goes,
 * the keyboard focus goes back to where it was before, usually the button that opened it
 * (WCAG 2.4.3). Mantine does this when `opened` turns false, but not when the dialog is
 * removed at once. The element is remembered while rendering, before the dialog takes the
 * focus.
 */
export function useReturnFocus(): void {
  const [previous] = useState(() =>
    typeof document === 'undefined' ? null : (document.activeElement as HTMLElement | null)
  );
  useEffect(
    () => () => {
      window.setTimeout(() => {
        const current = document.activeElement;
        const lost = !current || current === document.body;
        if (lost && previous?.isConnected) {
          previous.focus();
        }
      }, 0);
    },
    [previous]
  );
}

/**
 * The same for a dialog that stays mounted and is shown with `opened`. Mantine returns the
 * focus itself from the second opening on, but not when the dialog is mounted already open
 * (as the lazily loaded training panel is the first time). This catches that case: once the
 * dialog has closed (and its closing transition is over), a focus left on nothing goes back.
 */
export function useReturnFocusOnClose(opened: boolean): void {
  const trigger = useRef<HTMLElement | null>(null);
  if (opened && trigger.current === null && typeof document !== 'undefined') {
    trigger.current = document.activeElement as HTMLElement | null;
  }
  useEffect(() => {
    if (opened || !trigger.current) {
      return undefined;
    }
    const element = trigger.current;
    trigger.current = null;
    const restore = () => {
      const current = document.activeElement;
      if ((!current || current === document.body) && element.isConnected) {
        element.focus();
      }
    };
    const timers = [0, 400].map((delay) => window.setTimeout(restore, delay));
    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [opened]);
}
