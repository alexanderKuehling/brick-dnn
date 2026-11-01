/**
 * Drag with pointer events, so it works with a mouse as well as on touch screens
 * (HTML5 drag and drop does not fire on touch devices).
 *
 * Mouse: the drag starts after a small movement.
 * Touch: pulling sideways or a long press starts the drag; vertical swipes keep
 * scrolling the list. The element needs `touch-action: pan-y` for this.
 */

export interface Point {
  x: number;
  y: number;
}

export interface PointerDragOptions {
  /** Builds the preview that follows the pointer. It is removed when the drag ends. */
  createGhost: () => HTMLElement;
  /** Called once the gesture turned into a drag. */
  onStart?: () => void;
  /** Called with the release point in client coordinates. Not called when cancelled. */
  onDrop: (point: Point) => void;
}

const MOUSE_THRESHOLD = 4;
const TOUCH_THRESHOLD = 8;
const LONG_PRESS_MS = 300;

export function startPointerDrag(event: React.PointerEvent, options: PointerDragOptions) {
  if (event.pointerType === 'mouse' && event.button !== 0) {
    return;
  }
  const touch = event.pointerType !== 'mouse';
  const { pointerId } = event;
  const origin: Point = { x: event.clientX, y: event.clientY };
  let last = origin;
  let ghost: HTMLElement | null = null;
  let timer: number | undefined;

  const place = () => {
    if (ghost) {
      ghost.style.transform = `translate(${last.x}px, ${last.y}px) translate(-50%, -50%)`;
    }
  };

  const begin = () => {
    window.clearTimeout(timer);
    ghost = options.createGhost();
    ghost.style.position = 'fixed';
    ghost.style.left = '0';
    ghost.style.top = '0';
    ghost.style.pointerEvents = 'none';
    ghost.style.zIndex = '10000';
    document.body.appendChild(ghost);
    document.body.style.userSelect = 'none';
    place();
    options.onStart?.();
  };

  const finish = (drop: boolean) => {
    window.clearTimeout(timer);
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
    window.removeEventListener('pointercancel', onCancel);
    window.removeEventListener('touchmove', onTouchMove);
    window.removeEventListener('keydown', onKey);
    if (ghost) {
      ghost.remove();
      document.body.style.userSelect = '';
      if (drop) {
        options.onDrop(last);
      }
    }
  };

  const onMove = (e: PointerEvent) => {
    if (e.pointerId !== pointerId) {
      return;
    }
    last = { x: e.clientX, y: e.clientY };
    if (!ghost) {
      const dx = last.x - origin.x;
      const dy = last.y - origin.y;
      if (Math.hypot(dx, dy) < (touch ? TOUCH_THRESHOLD : MOUSE_THRESHOLD)) {
        return;
      }
      if (touch && Math.abs(dx) <= Math.abs(dy)) {
        finish(false); // vertical swipe: the user is scrolling the list
        return;
      }
      begin();
    }
    place();
  };

  const onUp = (e: PointerEvent) => {
    if (e.pointerId === pointerId) {
      last = { x: e.clientX, y: e.clientY };
      finish(true);
    }
  };

  const onCancel = (e: PointerEvent) => {
    if (e.pointerId === pointerId) {
      finish(false);
    }
  };

  // Once dragging, stop the browser from scrolling (needed after a long press).
  const onTouchMove = (e: TouchEvent) => {
    if (ghost && e.cancelable) {
      e.preventDefault();
    }
  };

  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      finish(false);
    }
  };

  window.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', onUp);
  window.addEventListener('pointercancel', onCancel);
  window.addEventListener('touchmove', onTouchMove, { passive: false });
  window.addEventListener('keydown', onKey);
  if (touch) {
    timer = window.setTimeout(begin, LONG_PRESS_MS);
  }
}
