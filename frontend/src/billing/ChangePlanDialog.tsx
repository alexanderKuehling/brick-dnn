import { useEffect, useState } from 'react';
import { IconAlertCircle } from '@tabler/icons-react';
import { Alert, Button, Group, Loader, Modal, Stack, Text } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { api, type BillingPlan, type BillingState, type PlanChangePreview } from '@/api';
import { formatDay, formatPrice } from './format';

/** What Paddle does with the rest of the period that is paid already. */
function nowLine(preview: PlanChangePreview): string {
  const { now } = preview;
  if (!now || now.amount === 0) {
    return preview.upgrade
      ? 'The difference for the rest of the month you paid for is charged now.'
      : 'The difference for the rest of the month you paid for is credited to your next invoice.';
  }
  const amount = formatPrice(now.amount, now.currency);
  return now.action === 'charge'
    ? `${amount} is charged now for the rest of the month you paid for.`
    : `${amount} is credited to your next invoice for the rest of the month you paid for.`;
}

/** Asks before moving the subscription to another plan, with what it costs. */
export function ChangePlanDialog({
  plan,
  currentName,
  onClose,
  onChanged,
}: {
  plan: BillingPlan;
  currentName: string;
  onClose: () => void;
  onChanged: (state: BillingState) => void;
}) {
  const [preview, setPreview] = useState<PlanChangePreview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let current = true;
    api.billing
      .previewChange(plan.key)
      .then((result) => current && setPreview(result))
      .catch((err: Error) => current && setError(err.message));
    return () => {
      current = false;
    };
  }, [plan.key]);

  const change = async () => {
    setBusy(true);
    setError(null);
    try {
      const state = await api.billing.changePlan(plan.key);
      notifications.show({
        color: 'green',
        title: `You are on the ${plan.name} plan now`,
        message: 'Your new limits apply at once.',
      });
      onChanged(state);
      onClose();
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  };

  const price = `${formatPrice(plan.price ?? 0, plan.currency)} per month`;
  return (
    <Modal
      opened
      onClose={onClose}
      title={`Switch to the ${plan.name} plan?`}
      closeOnClickOutside={!busy}
    >
      <Stack gap="sm">
        <Text size="sm">
          Your subscription changes from {currentName} to <strong>{plan.name}</strong> at once and
          costs <strong>{price}</strong> from then on (VAT included).
        </Text>
        {preview ? (
          <Text size="sm">
            {nowLine(preview)}
            {preview.next_billed_at && ` Next payment: ${formatDay(preview.next_billed_at)}.`}
          </Text>
        ) : (
          !error && (
            <Group gap="xs">
              <Loader size="xs" />
              <Text size="sm" c="dimmed">
                Asking Paddle what this costs now…
              </Text>
            </Group>
          )
        )}
        {error && (
          <Alert color="red" icon={<IconAlertCircle size={18} />}>
            {error}
          </Alert>
        )}
        <Group justify="flex-end" gap="sm">
          <Button variant="default" onClick={onClose} disabled={busy}>
            Keep {currentName}
          </Button>
          <Button onClick={change} loading={busy} disabled={!preview}>
            Switch to {plan.name}
          </Button>
        </Group>
      </Stack>
    </Modal>
  );
}
