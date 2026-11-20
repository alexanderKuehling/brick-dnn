import { useEffect, useState } from 'react';
import { IconAlertCircle } from '@tabler/icons-react';
import { Alert, Button, Group, Loader, Modal, Stack, Text } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import {
  api,
  type BillingInterval,
  type BillingPlan,
  type BillingState,
  type PlanChangePreview,
} from '@/api';
import { i18n, Trans, useTranslation } from '@/i18n';
import { formatDay, formatPrice, priceFor } from './format';

/** What Paddle does with the rest of the period that is paid already. */
function nowLine(preview: PlanChangePreview): string {
  const { now } = preview;
  if (!now || now.amount === 0) {
    return preview.upgrade
      ? i18n.t('ui.billing.ChangePlanDialog.theDifferenceForTheRest')
      : i18n.t('ui.billing.ChangePlanDialog.theDifferenceForTheRest2');
  }
  const amount = formatPrice(now.amount, now.currency);
  return now.action === 'charge'
    ? i18n.t('ui.billing.ChangePlanDialog.isChargedNowForThe', { amount })
    : i18n.t('ui.billing.ChangePlanDialog.isCreditedToYourNext', { amount });
}

/** Asks before moving the subscription to another plan, with what it costs. */
export function ChangePlanDialog({
  plan,
  currentName,
  interval = 'month',
  onClose,
  onChanged,
}: {
  plan: BillingPlan;
  currentName: string;
  /** How the subscription is paid; it stays that way. */
  interval?: BillingInterval;
  onClose: () => void;
  onChanged: (state: BillingState) => void;
}) {
  useTranslation();
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
        title: i18n.t('ui.billing.ChangePlanDialog.youAreOnThePlan', { name: plan.name }),
        message: i18n.t('ui.billing.ChangePlanDialog.yourNewLimitsApplyAt'),
      });
      onChanged(state);
      onClose();
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  };

  // The subscription keeps its interval: a yearly one moves to the plan's yearly price.
  const price = priceFor(plan, preview?.interval ?? interval) || priceFor(plan);
  return (
    <Modal
      opened
      onClose={onClose}
      title={i18n.t('ui.billing.ChangePlanDialog.switchToThePlan', { name: plan.name })}
      closeOnClickOutside={!busy}
    >
      <Stack gap="sm">
        <Text size="sm">
          <Trans
            i18nKey="ui.billing.ChangePlanDialog.yourSubscriptionChangesFromTo"
            values={{ currentName, name: plan.name, price }}
            components={[<strong />, <strong />]}
          />
        </Text>
        {preview ? (
          <Text size="sm">
            {nowLine(preview)}
            {preview.next_billed_at &&
              i18n.t('ui.billing.ChangePlanDialog.nextPayment', {
                next_billed_at: formatDay(preview.next_billed_at),
              })}
          </Text>
        ) : (
          !error && (
            <Group gap="xs">
              <Loader size="xs" />
              <Text size="sm" c="dimmed">
                {i18n.t('ui.billing.ChangePlanDialog.askingPaddleWhatThisCosts')}
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
            {i18n.t('ui.billing.ChangePlanDialog.keep', { currentName })}
          </Button>
          <Button onClick={change} loading={busy} disabled={!preview}>
            {i18n.t('ui.billing.ChangePlanDialog.switchTo', { name: plan.name })}
          </Button>
        </Group>
      </Stack>
    </Modal>
  );
}
