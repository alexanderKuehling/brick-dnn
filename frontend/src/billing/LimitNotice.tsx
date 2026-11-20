/**
 * Where a limit leads. The server says which limit stopped a request (`LimitInfo`: the kind,
 * what is used of it, the account's plan) and, where plans are sold, the next plan that lifts
 * it. Here that becomes one sentence about the limit in the language of the app, one about
 * the plan with its price and what it includes, and one button to the pricing page with that
 * plan preselected. On a server that sells no plans there is no offer and nothing of this
 * shows a price.
 */
import { useEffect, useState } from 'react';
import { IconExternalLink } from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import { Anchor, Button, Group, Loader, Modal, Stack, Text, VisuallyHidden } from '@mantine/core';
import { api, type PlanOffer, type Usage } from '@/api';
import { i18n, useTranslation } from '@/i18n';
import { formatBytes } from '@/training/format';
import { formatCpu, formatResetDate } from '@/usage/format';
import { OverageForm } from '@/usage/UsageSection';
import { formatPrice, planIncludes } from './format';
import { limitOf, pricingPath, type Limit } from './limits';

export { type Limit, limitOf, pricingPath };

function amount(value: number | null | undefined, unit: Limit['limit']['unit']): string {
  if (typeof value !== 'number') {
    return '–';
  }
  if (unit === 'bytes') {
    return formatBytes(value);
  }
  return unit === 'cpu_minutes' ? formatCpu(value) : String(value);
}

/**
 * The limit in one sentence. `estimate`: the CPU minutes a planned training is expected to
 * need; `stopped`: the limit ended a run. Null: the server's own message says it better (a
 * kind this version does not know, or the budget of a token or app).
 */
export function limitText(
  info: Limit,
  { estimate, stopped = false }: { estimate?: number | null; stopped?: boolean } = {}
): string | null {
  const { kind, used, limit, unit, resets_at: resets } = info.limit;
  const plan = info.plan?.name;
  const values = {
    plan,
    used: amount(used, unit),
    limit: amount(limit, unit),
    date: resets ? formatResetDate(resets) : '–',
    estimate: typeof estimate === 'number' ? formatCpu(estimate) : '–',
    count: typeof limit === 'number' ? limit : 1,
  };
  switch (kind) {
    case 'cpu_time':
      if (info.reserved) {
        return i18n.t('plans.limit.reserved');
      }
      if (typeof estimate === 'number') {
        return i18n.t('plans.limit.leftThisMonth', {
          ...values,
          limit: formatCpu(Math.max(0, (limit ?? 0) - (used ?? 0))),
        });
      }
      return i18n.t(plan ? 'plans.limit.cpuTimePlan' : 'plans.limit.cpuTime', values);
    case 'overage_cap':
      return i18n.t('plans.limit.overageCap', values);
    case 'own_budget':
      return i18n.t('plans.limit.ownBudget', values);
    case 'run_time':
      if (stopped) {
        return i18n.t(
          plan ? 'plans.limit.runTimeStoppedPlan' : 'plans.limit.runTimeStopped',
          values
        );
      }
      return i18n.t(plan ? 'plans.limit.runTimePlan' : 'plans.limit.runTime', values);
    case 'storage':
      return i18n.t(plan ? 'plans.limit.storagePlan' : 'plans.limit.storage', values);
    case 'concurrent_trainings':
      return i18n.t(plan ? 'plans.limit.concurrentPlan' : 'plans.limit.concurrent', values);
    case 'queued_trainings':
      return i18n.t('plans.limit.queued', values);
    default:
      return null;
  }
}

/** A short title for the limit; null for kinds without one. */
export function limitTitle(info: Limit): string | null {
  switch (info.limit.kind) {
    case 'cpu_time':
      return i18n.t('plans.limit.titles.cpuTime');
    case 'overage_cap':
    case 'own_budget':
    case 'token_budget':
    case 'app_budget':
      return i18n.t('plans.limit.titles.budget');
    case 'run_time':
      return i18n.t('plans.limit.titles.runTime');
    case 'storage':
      return i18n.t('plans.limit.titles.storage');
    case 'concurrent_trainings':
      return i18n.t('plans.limit.titles.concurrent');
    case 'queued_trainings':
      return i18n.t('plans.limit.titles.queued');
    default:
      return null;
  }
}

/** "The Starter plan includes 10 CPU hours per month … for €5 per month." */
export function offerText(offer: PlanOffer, key: 'upgrade' | 'usage' = 'upgrade'): string {
  return i18n.t(`plans.${key}.offer`, {
    name: offer.name,
    includes: planIncludes(offer),
    price: formatPrice(offer.price ?? 0, offer.currency),
  });
}

/**
 * The button to the pricing page with the plan preselected. `newTab`: inside the training
 * panel, which keeps its network and settings that way.
 */
export function UpgradeButton({
  offer,
  newTab = false,
  variant = 'light',
  label,
}: {
  offer: Pick<PlanOffer, 'key' | 'name'>;
  newTab?: boolean;
  variant?: 'light' | 'filled' | 'default';
  label?: string;
}) {
  useTranslation();
  const text = label ?? i18n.t('plans.upgrade.button', { name: offer.name });
  if (!newTab) {
    return (
      <Button component={Link} to={pricingPath(offer.key)} size="compact-sm" variant={variant}>
        {text}
      </Button>
    );
  }
  return (
    <Button
      component="a"
      href={pricingPath(offer.key)}
      target="_blank"
      rel="noopener"
      size="compact-sm"
      variant={variant}
      rightSection={<IconExternalLink size={14} aria-hidden />}
    >
      {text}
      <VisuallyHidden> {i18n.t('plans.upgrade.newTab')}</VisuallyHidden>
    </Button>
  );
}

/** "Usage & budget" on the account page, where own budgets and the overage limit are set. */
function AccountAnchor({ newTab }: { newTab: boolean }) {
  useTranslation();
  const label = i18n.t('plans.upgrade.usageAndBudget');
  return newTab ? (
    <Anchor href="/account" target="_blank" rel="noopener" size="sm">
      {label}
      <VisuallyHidden> {i18n.t('plans.upgrade.newTab')}</VisuallyHidden>
    </Anchor>
  ) : (
    <Anchor component={Link} to="/account" size="sm">
      {label}
    </Anchor>
  );
}

/**
 * Switching on compute beyond the plan where the limit was hit: the same form as on the
 * account page (off until the user decides, with a limit of their own).
 */
function OverageDialog({ onClose, onSaved }: { onClose: () => void; onSaved?: () => void }) {
  useTranslation();
  const [usage, setUsage] = useState<Usage | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    api.account
      .usage(controller.signal)
      .then(setUsage)
      .catch((err: Error) => !controller.signal.aborted && setError(err.message));
    return () => controller.abort();
  }, []);
  return (
    <Modal opened onClose={onClose} title={i18n.t('plans.upgrade.overageTitle')} centered>
      {error ? (
        <Text size="sm" c="red" role="alert">
          {error}
        </Text>
      ) : !usage ? (
        <Loader size="sm" />
      ) : (
        <OverageForm
          key={String(usage.overage?.enabled)}
          usage={usage}
          startEnabled
          onSaved={(saved) => {
            setUsage(saved);
            if (saved.overage?.enabled) {
              onSaved?.();
              onClose();
            }
          }}
        />
      )}
    </Modal>
  );
}

/**
 * What can be done about a limit: the plan that lifts it with one button, for a paid plan
 * that ran out the switch for compute beyond it, else the way to the account page where the
 * own budgets are. Nothing where neither helps.
 */
export function UpgradeHint({
  info,
  newTab = false,
  onChanged,
}: {
  info: Limit;
  newTab?: boolean;
  /** The user allowed compute beyond the plan: the limit may be gone. */
  onChanged?: () => void;
}) {
  useTranslation();
  const [overageOpen, setOverageOpen] = useState(false);
  const offer = info.reserved ? null : (info.upgrade ?? null);
  const overage = info.reserved ? null : (info.overage ?? null);
  const own = ['own_budget', 'overage_cap'].includes(info.limit.kind);
  if (!offer && !overage && !own) {
    return null;
  }
  return (
    <Stack gap={6} mt="xs" align="flex-start">
      {offer && <Text size="sm">{offerText(offer)}</Text>}
      {overage && !offer && (
        <Text size="sm">
          {i18n.t('plans.upgrade.overage', {
            price: formatPrice(overage.price_per_cpu_hour, overage.currency),
          })}
        </Text>
      )}
      <Group gap="xs">
        {offer && <UpgradeButton offer={offer} newTab={newTab} />}
        {overage && (
          <Button
            size="compact-sm"
            variant={offer ? 'default' : 'light'}
            onClick={() => setOverageOpen(true)}
          >
            {i18n.t('plans.upgrade.overageButton')}
          </Button>
        )}
        {own && <AccountAnchor newTab={newTab} />}
      </Group>
      {overageOpen && <OverageDialog onClose={() => setOverageOpen(false)} onSaved={onChanged} />}
    </Stack>
  );
}

/**
 * The body of a message at a limit: the limit in one sentence (the server's own text where
 * this version has none), then where it leads.
 */
export function LimitNotice({
  info,
  fallback,
  estimate,
  stopped,
  newTab = false,
  onChanged,
}: {
  info: Limit;
  /** The server's message, shown for limits this version has no sentence for. */
  fallback?: string | null;
  estimate?: number | null;
  stopped?: boolean;
  newTab?: boolean;
  onChanged?: () => void;
}) {
  useTranslation();
  return (
    <>
      <Text size="sm">{limitText(info, { estimate, stopped }) ?? fallback}</Text>
      <UpgradeHint info={info} newTab={newTab} onChanged={onChanged} />
    </>
  );
}
