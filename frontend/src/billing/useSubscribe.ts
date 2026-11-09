import { useCallback, useEffect, useRef, useState } from 'react';
import { useComputedColorScheme } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { api, type BillingEnabled, type BillingPlan, type BillingState } from '@/api';
import { openCheckout } from './paddle';

const POLL_MS = 2000;
const GIVE_UP_MS = 90_000;

function pause(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * After a payment: waits until the server knows the subscription (Paddle tells it with a
 * notification, usually within seconds). If that takes long, the server asks Paddle itself.
 * Resolves to the new state, or null if the plan is still not there after a while.
 */
export async function waitForPlan(
  plan: string,
  {
    intervalMs = POLL_MS,
    timeoutMs = GIVE_UP_MS,
    isCancelled = () => false,
  }: { intervalMs?: number; timeoutMs?: number; isCancelled?: () => boolean } = {}
): Promise<BillingState | null> {
  const started = Date.now();
  for (let attempt = 1; Date.now() - started < timeoutMs; attempt += 1) {
    await pause(intervalMs);
    if (isCancelled()) {
      return null;
    }
    try {
      const state =
        attempt >= 4 && attempt % 3 === 1 ? await api.billing.refresh() : await api.billing.state();
      if (state.plan.key === plan && state.subscription?.plan === plan) {
        return state;
      }
    } catch {
      // A failed look is not the end: try again.
    }
  }
  return null;
}

/**
 * Buying a plan: asks the server for the checkout, opens Paddle's overlay (which loads
 * Paddle.js on first use) and, once paid, waits for the plan to arrive.
 */
export function useSubscribe(info: BillingEnabled, onChanged: (state: BillingState) => void) {
  const theme = useComputedColorScheme('light');
  const [busy, setBusy] = useState<string | null>(null);
  const [activating, setActivating] = useState<string | null>(null);
  const mounted = useRef(true);
  useEffect(
    () => () => {
      mounted.current = false;
    },
    []
  );

  const subscribe = useCallback(
    async (plan: BillingPlan) => {
      setBusy(plan.key);
      try {
        const checkout = await api.billing.checkout(plan.key);
        await openCheckout(info, checkout, theme, {
          onCompleted: async () => {
            setActivating(plan.key);
            const state = await waitForPlan(plan.key, { isCancelled: () => !mounted.current });
            if (!mounted.current) {
              return;
            }
            setActivating(null);
            if (state) {
              onChanged(state);
              notifications.show({
                color: 'green',
                title: `You are on the ${plan.name} plan now`,
                message: 'Thank you! Your new limits apply at once.',
              });
            } else {
              notifications.show({
                color: 'yellow',
                title: 'Your payment arrived',
                message:
                  'The plan is not active yet. This can take a few minutes; load this page again then.',
                autoClose: false,
              });
            }
          },
        });
      } catch (err) {
        notifications.show({
          color: 'red',
          title: 'Could not open the checkout',
          message: (err as Error).message,
        });
      } finally {
        if (mounted.current) {
          setBusy(null);
        }
      }
    },
    [info, theme, onChanged]
  );

  return { subscribe, busy, activating };
}
