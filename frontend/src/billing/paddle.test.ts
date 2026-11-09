import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import type { BillingCheckout, BillingEnabled } from '@/api';
import { loadPaddle, openCheckout, PADDLE_SCRIPT_URL, resetPaddle } from './paddle';
import { BILLING } from './testData';

const SANDBOX: BillingEnabled = {
  ...BILLING,
  environment: 'sandbox',
  client_token: 'test_0123456789abcdef',
};
const CHECKOUT: BillingCheckout = {
  plan: 'plus',
  price_id: 'pri_plus',
  customer_id: 'ctm_ada',
  custom_data: { user_id: '7' },
};

function fakePaddle() {
  return {
    Environment: { set: vi.fn() },
    Initialize: vi.fn(),
    Checkout: { open: vi.fn(), close: vi.fn() },
  };
}

function scripts(): HTMLScriptElement[] {
  return Array.from(document.querySelectorAll<HTMLScriptElement>('script[src*="paddle"]'));
}

/** jsdom does not fetch scripts: answer the one that was added as Paddle's CDN would. */
function answerScript(paddle: ReturnType<typeof fakePaddle> | null): void {
  const script = scripts().at(-1)!;
  if (paddle) {
    window.Paddle = paddle;
    script.onload!(new Event('load'));
  } else {
    script.onerror!(new Event('error'));
  }
}

beforeEach(() => {
  resetPaddle();
  delete window.Paddle;
});

afterEach(() => {
  scripts().forEach((script) => script.remove());
});

test('nothing of Paddle is on the page until the checkout is opened', async () => {
  await import('./paddle');
  expect(scripts()).toHaveLength(0);
  expect(window.Paddle).toBeUndefined();
});

test('the script is loaded from Paddle once and set up with the token', async () => {
  const paddle = fakePaddle();
  const first = loadPaddle(BILLING);
  const second = loadPaddle(BILLING);
  expect(scripts().map((script) => script.src)).toEqual([PADDLE_SCRIPT_URL]);
  answerScript(paddle);
  expect(await first).toBe(paddle);
  expect(await second).toBe(paddle);
  expect(paddle.Initialize).toHaveBeenCalledTimes(1);
  expect(paddle.Initialize.mock.calls[0][0].token).toBe('live_0123456789abcdef');
  // Live is Paddle's default environment.
  expect(paddle.Environment.set).not.toHaveBeenCalled();
  await loadPaddle(BILLING);
  expect(scripts()).toHaveLength(1);
});

test('the sandbox is chosen before Paddle.js is set up', async () => {
  const paddle = fakePaddle();
  const loading = loadPaddle(SANDBOX);
  answerScript(paddle);
  await loading;
  expect(paddle.Environment.set).toHaveBeenCalledWith('sandbox');
  expect(paddle.Environment.set.mock.invocationCallOrder[0]).toBeLessThan(
    paddle.Initialize.mock.invocationCallOrder[0]
  );
  expect(paddle.Initialize.mock.calls[0][0].token).toBe('test_0123456789abcdef');
});

test('the checkout opens for the price and the customer the server named', async () => {
  const paddle = fakePaddle();
  const opening = openCheckout(SANDBOX, CHECKOUT, 'dark');
  answerScript(paddle);
  await opening;
  expect(paddle.Checkout.open).toHaveBeenCalledWith({
    items: [{ priceId: 'pri_plus', quantity: 1 }],
    customer: { id: 'ctm_ada' },
    customData: { user_id: '7' },
    settings: { displayMode: 'overlay', theme: 'dark', allowLogout: false },
  });
});

test('events of the checkout reach the handlers of the checkout that is open', async () => {
  const paddle = fakePaddle();
  const first = { onCompleted: vi.fn(), onClosed: vi.fn() };
  const opening = openCheckout(BILLING, CHECKOUT, 'light', first);
  answerScript(paddle);
  await opening;
  const { eventCallback } = paddle.Initialize.mock.calls[0][0];
  eventCallback({ name: 'checkout.loaded' });
  expect(first.onCompleted).not.toHaveBeenCalled();
  eventCallback({ name: 'checkout.completed' });
  eventCallback({ name: 'checkout.closed' });
  expect(first.onCompleted).toHaveBeenCalledTimes(1);
  expect(first.onClosed).toHaveBeenCalledTimes(1);

  const second = { onCompleted: vi.fn() };
  await openCheckout(BILLING, { ...CHECKOUT, price_id: 'pri_pro' }, 'light', second);
  eventCallback({ name: 'checkout.completed' });
  expect(second.onCompleted).toHaveBeenCalledTimes(1);
  expect(first.onCompleted).toHaveBeenCalledTimes(1);
  expect(paddle.Initialize).toHaveBeenCalledTimes(1);
});

test('a script that cannot be loaded is an error, and the next try loads it again', async () => {
  const failing = openCheckout(BILLING, CHECKOUT, 'light');
  answerScript(null);
  await expect(failing).rejects.toThrow('could not be loaded from Paddle');
  expect(scripts()).toHaveLength(0);

  const paddle = fakePaddle();
  const again = openCheckout(BILLING, CHECKOUT, 'light');
  answerScript(paddle);
  await again;
  expect(paddle.Checkout.open).toHaveBeenCalledTimes(1);
});
