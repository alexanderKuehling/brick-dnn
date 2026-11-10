import { useEffect, useState } from 'react';
import { IconAlertCircle, IconAlertTriangle, IconExternalLink } from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import { Alert, Badge, Button, Group, Loader, Modal, Stack, Text } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { api, type BillingEnabled, type BillingState, type BillingSubscription } from '@/api';
import { formatDay, formatPrice } from './format';

const STATUS: Record<string, { label: string; color: string }> = {
  active: { label: 'Active', color: 'green' },
  trialing: { label: 'Trial', color: 'green' },
  past_due: { label: 'Payment failed', color: 'red' },
  paused: { label: 'Paused', color: 'yellow' },
  canceled: { label: 'Ended', color: 'gray' },
};

/** One sentence on what happens next with the subscription. */
function nextStep(subscription: BillingSubscription, price: string): string {
  const name = subscription.plan_name ?? 'your plan';
  if (subscription.status === 'canceled') {
    return `Your ${name} subscription ended${
      subscription.ended_at ? ` on ${formatDay(subscription.ended_at)}` : ''
    }. Nothing was deleted; the limits of the free plan apply.`;
  }
  if (subscription.ends_at) {
    return `Cancelled: you keep ${name} until ${formatDay(subscription.ends_at)}. After that the limits of the free plan apply; nothing is deleted.`;
  }
  if (subscription.status === 'past_due') {
    return subscription.grace_until
      ? `The last payment did not go through. Paddle tries again; you keep ${name} until ${formatDay(subscription.grace_until)} at least. Please check your payment method.`
      : `The last payment did not go through, so the limits of the free plan apply until it does. Please check your payment method.`;
  }
  if (subscription.status === 'paused') {
    return 'Your subscription is paused: the limits of the free plan apply.';
  }
  return subscription.renews_at
    ? `Renews on ${formatDay(subscription.renews_at)}${price && ` for ${price}`}.`
    : '';
}

/** Opens Paddle's customer portal in a new tab (the link is made on request). */
async function openPortal(): Promise<void> {
  // Opened before the request, so that the browser does not block it as a pop-up.
  const tab = window.open('', '_blank');
  try {
    const { url } = await api.billing.portal();
    if (tab) {
      tab.opener = null;
      tab.location.href = url;
    } else {
      window.location.assign(url);
    }
  } catch (err) {
    tab?.close();
    throw err;
  }
}

function CancelDialog({
  subscription,
  onClose,
  onCancelled,
}: {
  subscription: BillingSubscription;
  onClose: () => void;
  onCancelled: (state: BillingState) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const name = subscription.plan_name ?? 'your plan';
  const until = subscription.current_period_end;

  const cancel = async () => {
    setBusy(true);
    setError(null);
    try {
      const state = await api.billing.cancel();
      notifications.show({ color: 'green', message: 'Your subscription was cancelled.' });
      onCancelled(state);
      onClose();
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  };

  return (
    <Modal opened onClose={onClose} title="Cancel your subscription?" closeOnClickOutside={!busy}>
      <Stack gap="sm">
        <Text size="sm">
          {until
            ? `You keep ${name} until ${formatDay(until)}, the end of the month you paid for. No further payment is taken.`
            : `Your subscription ends and no further payment is taken.`}{' '}
          After that the limits of the free plan apply. Nothing is deleted, and you can continue the
          subscription until then.
        </Text>
        {error && (
          <Alert color="red" icon={<IconAlertCircle size={18} />}>
            {error}
          </Alert>
        )}
        <Group justify="flex-end" gap="sm">
          <Button variant="default" onClick={onClose} disabled={busy}>
            Keep {name}
          </Button>
          <Button color="red" onClick={cancel} loading={busy}>
            Cancel subscription
          </Button>
        </Group>
      </Stack>
    </Modal>
  );
}

/** The account page's "Plan & billing". Only rendered on a server that sells plans. */
export function PlanSection({ info }: { info: BillingEnabled }) {
  const [state, setState] = useState<BillingState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [busy, setBusy] = useState<'reactivate' | 'portal' | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api.billing
      .state(controller.signal)
      .then(setState)
      .catch((err: Error) => !controller.signal.aborted && setError(err.message));
    return () => controller.abort();
  }, []);

  if (error) {
    return (
      <Alert color="red" icon={<IconAlertCircle size={18} />}>
        {error}
      </Alert>
    );
  }
  if (!state) {
    return <Loader size="sm" />;
  }

  const run = async (what: 'reactivate' | 'portal', action: () => Promise<void>) => {
    setBusy(what);
    try {
      await action();
    } catch (err) {
      notifications.show({ color: 'red', message: (err as Error).message });
    } finally {
      setBusy(null);
    }
  };
  const reactivate = () =>
    run('reactivate', async () => {
      setState(await api.billing.reactivate());
      notifications.show({ color: 'green', message: 'Your subscription continues.' });
    });

  const { plan, subscription, unlimited } = state;
  const paid = info.plans.find((p) => p.key === subscription?.plan);
  const price = paid?.price
    ? `${formatPrice(paid.price, paid.currency ?? info.currency)} per month`
    : '';
  const badge = subscription ? STATUS[subscription.status] : undefined;
  const warn = subscription?.status === 'past_due' || subscription?.status === 'paused';

  return (
    <Stack gap="sm">
      <Group gap="xs" wrap="wrap">
        <Text size="sm" fw={600}>
          {unlimited ? 'No plan needed' : `${plan.name} plan`}
        </Text>
        {subscription && badge && subscription.status !== 'canceled' && (
          <Badge variant="light" color={subscription.ends_at ? 'yellow' : badge.color}>
            {subscription.ends_at ? 'Cancelled' : badge.label}
          </Badge>
        )}
        {info.environment === 'sandbox' && (
          <Badge variant="outline" color="yellow">
            Test mode
          </Badge>
        )}
      </Group>

      {unlimited ? (
        <Text size="sm" c="dimmed">
          No limits apply to your account, so it does not need a paid plan.
        </Text>
      ) : subscription ? (
        warn ? (
          <Alert color="red" icon={<IconAlertTriangle size={18} />}>
            {nextStep(subscription, price)}
          </Alert>
        ) : (
          <Text size="sm" c="dimmed">
            {nextStep(subscription, price)}
          </Text>
        )
      ) : (
        <Text size="sm" c="dimmed">
          You are on the free plan. Paid plans include more compute for trainings and more storage.
        </Text>
      )}

      {!unlimited && (
        <Group gap="sm" wrap="wrap">
          {state.can_subscribe ? (
            <Button component={Link} to="/pricing">
              See plans
            </Button>
          ) : (
            subscription?.can_change_plan && (
              <Button component={Link} to="/pricing" variant="light">
                Change plan
              </Button>
            )
          )}
          {subscription?.can_reactivate && (
            <Button onClick={reactivate} loading={busy === 'reactivate'}>
              Continue subscription
            </Button>
          )}
          {state.has_billing_profile && (
            <Button
              variant="default"
              rightSection={<IconExternalLink size={14} />}
              loading={busy === 'portal'}
              onClick={() => run('portal', openPortal)}
            >
              Payment method and invoices
            </Button>
          )}
          {subscription?.can_cancel && (
            <Button variant="subtle" color="red" onClick={() => setCancelling(true)}>
              Cancel subscription
            </Button>
          )}
        </Group>
      )}
      <Text size="xs" c="dimmed">
        Paid plans are sold by Paddle, which handles payments and invoices.
        {state.has_billing_profile &&
          ' “Payment method and invoices” opens Paddle’s customer portal in a new tab.'}
      </Text>

      {cancelling && subscription && (
        <CancelDialog
          subscription={subscription}
          onClose={() => setCancelling(false)}
          onCancelled={setState}
        />
      )}
    </Stack>
  );
}
