/**
 * Paddle.js: Paddle's checkout in an overlay on this page.
 *
 * The script comes from Paddle's servers, so it is only loaded when somebody opens the
 * checkout (never while just looking at the site, and never on a server that sells no plans):
 * until then no request goes to Paddle.
 */
import type { BillingCheckout, BillingEnabled } from '@/api';

export const PADDLE_SCRIPT_URL = 'https://cdn.paddle.com/paddle/v2/paddle.js';

interface PaddleEvent {
  name?: string;
}

/** The part of Paddle.js (v2, Paddle Billing) that is used here. */
interface PaddleJs {
  Environment: { set: (environment: 'sandbox' | 'production') => void };
  Initialize: (options: { token: string; eventCallback?: (event: PaddleEvent) => void }) => void;
  Checkout: {
    open: (options: {
      items: { priceId: string; quantity: number }[];
      customer: { id: string };
      customData: Record<string, string>;
      settings: {
        displayMode: 'overlay';
        theme: 'light' | 'dark';
        allowLogout: boolean;
      };
    }) => void;
    close: () => void;
  };
}

declare global {
  interface Window {
    Paddle?: PaddleJs;
  }
}

export interface CheckoutHandlers {
  /** The payment went through (the overlay shows Paddle's confirmation). */
  onCompleted?: () => void;
  /** The overlay was closed, paid or not. */
  onClosed?: () => void;
}

let ready: Promise<PaddleJs> | null = null;
let handlers: CheckoutHandlers = {};

function inject(): Promise<void> {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = PADDLE_SCRIPT_URL;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => {
      script.remove();
      reject(new Error('The checkout could not be loaded from Paddle. Check your connection.'));
    };
    document.head.appendChild(script);
  });
}

/** Loads and sets up Paddle.js, once per page load. */
export function loadPaddle(info: BillingEnabled): Promise<PaddleJs> {
  if (!ready) {
    ready = inject()
      .then(() => {
        const paddle = window.Paddle;
        if (!paddle) {
          throw new Error('The checkout could not be loaded from Paddle.');
        }
        // The live environment is the default; test tokens only work in the sandbox.
        if (info.environment === 'sandbox') {
          paddle.Environment.set('sandbox');
        }
        paddle.Initialize({
          token: info.client_token,
          eventCallback: (event) => {
            if (event.name === 'checkout.completed') {
              handlers.onCompleted?.();
            } else if (event.name === 'checkout.closed') {
              handlers.onClosed?.();
            }
          },
        });
        return paddle;
      })
      .catch((error) => {
        ready = null; // the next try loads it again
        throw error;
      });
  }
  return ready;
}

/**
 * Opens Paddle's checkout for the plan the server prepared (`api.billing.checkout`): its
 * price, for the account's customer at Paddle. The customer cannot be swapped in the overlay.
 */
export async function openCheckout(
  info: BillingEnabled,
  checkout: BillingCheckout,
  theme: 'light' | 'dark',
  on: CheckoutHandlers = {}
): Promise<void> {
  const paddle = await loadPaddle(info);
  handlers = on;
  paddle.Checkout.open({
    items: [{ priceId: checkout.price_id, quantity: 1 }],
    customer: { id: checkout.customer_id },
    customData: checkout.custom_data,
    settings: { displayMode: 'overlay', theme, allowLogout: false },
  });
}

/** For tests: as if the page was loaded again. */
export function resetPaddle(): void {
  ready = null;
  handlers = {};
}
