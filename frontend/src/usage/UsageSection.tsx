import { useEffect, useState } from 'react';
import { IconAlertCircle } from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import {
  Alert,
  Anchor,
  Badge,
  Button,
  Group,
  Loader,
  NumberInput,
  Progress,
  Stack,
  Switch,
  Text,
} from '@mantine/core';
import { api, type Plan, type Usage } from '@/api';
import { currentLocale, i18n, Trans, useTranslation } from '@/i18n';
import { formatBytes } from '@/training/format';
import { BudgetEditor } from './BudgetEditor';
import { formatCpu, formatMoney, formatResetDate, usageColor, usagePercent } from './format';

function Meter({
  label,
  used,
  limit,
  value,
  detail,
}: {
  label: string;
  used: number;
  limit: number | null;
  /** E.g. "12.5 of 300 CPU min". */
  value: string;
  detail?: React.ReactNode;
}) {
  useTranslation();
  const percent = usagePercent(used, limit);
  return (
    <div>
      <Group justify="space-between" gap="xs" wrap="wrap" mb={4}>
        <Text size="sm" fw={600}>
          {label}
        </Text>
        <Text size="sm">{value}</Text>
      </Group>
      {limit !== null && (
        <Progress
          value={percent}
          color={usageColor(percent)}
          aria-label={i18n.t('ui.usage.UsageSection.used', { label, round: Math.round(percent) })}
        />
      )}
      {detail && (
        <Text size="xs" c="dimmed" mt={4}>
          {detail}
        </Text>
      )}
    </div>
  );
}

/** "Trainings 12 CPU min · predictions 0.5 CPU min · via “agent” 10 CPU min". */
function breakdown(usage: Usage): string {
  const { by_kind: kind, by_source: sources } = usage.compute;
  const parts = [
    `trainings ${formatCpu(kind.training)}`,
    `predictions ${formatCpu(kind.prediction)}`,
  ];
  const agents = sources.filter((source) => source.source !== 'web');
  for (const agent of agents.slice(0, 3)) {
    parts.push(`“${agent.name}” ${formatCpu(agent.used_cpu_minutes)}`);
  }
  if (agents.length > 3) {
    parts.push(`${agents.length - 3} more tokens or apps`);
  }
  return parts.join(' · ');
}

/** "0.5 CPU h per month · 1 GB · 15 CPU min per training · 1 training at a time". */
export function planLimits(plan: Plan): string {
  const parts = [
    plan.cpu_hours_per_month !== null
      ? `${formatHours(plan.cpu_hours_per_month)} per month`
      : 'unlimited compute',
    plan.storage_gb !== null
      ? `${plan.storage_gb.toLocaleString(currentLocale())} GB`
      : 'unlimited storage',
  ];
  if (plan.cpu_minutes_per_run !== null) {
    parts.push(`${formatCpu(plan.cpu_minutes_per_run)} per training`);
  }
  if (plan.concurrent_trainings !== null) {
    const count = plan.concurrent_trainings;
    parts.push(`${count} training${count === 1 ? '' : 's'} at a time`);
  }
  return parts.join(' · ');
}

function formatHours(hours: number): string {
  return `${hours.toLocaleString(currentLocale(), { maximumFractionDigits: 2 })} CPU h`;
}

function planPrice(plan: Plan): string {
  if (plan.price === null) {
    return '';
  }
  return plan.price === 0
    ? 'no charge'
    : i18n.t('ui.usage.UsageSection.perMonth', { price: formatMoney(plan.price, plan.currency) });
}

/** "Starter (€5.00 per month): 10 CPU h, 5 GB". */
function otherPlan(plan: Plan): string {
  const limits = [
    plan.cpu_hours_per_month !== null ? formatHours(plan.cpu_hours_per_month) : 'unlimited',
    plan.storage_gb !== null ? `${plan.storage_gb.toLocaleString(currentLocale())} GB` : null,
  ].filter(Boolean);
  const price = planPrice(plan);
  return i18n.t('ui.usage.UsageSection.text', {
    name: plan.name,
    value: price ? ` (${price})` : '',
    limits: limits.join(', '),
  });
}

function PlanInfo({ usage }: { usage: Usage }) {
  useTranslation();
  const { plan, unlimited } = usage;
  if (!plan || (plan.key === 'default' && !unlimited)) {
    return null; // a server without plans: the limits below say it all
  }
  const others = (usage.plans ?? []).filter((other) => other.key !== plan.key);
  const price = planPrice(plan);
  return (
    <div>
      <Group gap="xs" wrap="wrap">
        <Text size="sm" fw={600}>
          {plan.key === 'default'
            ? i18n.t('ui.usage.UsageSection.yourAccount')
            : i18n.t('ui.usage.UsageSection.plan', { name: plan.name })}
        </Text>
        {price && !unlimited && (
          <Text size="sm" c="dimmed">
            {price}
          </Text>
        )}
        {unlimited && (
          <Badge variant="light" color="green">
            {i18n.t('ui.usage.UsageSection.unlimited')}
          </Badge>
        )}
      </Group>
      <Text size="xs" c="dimmed" mt={2}>
        {unlimited ? i18n.t('ui.usage.UsageSection.noLimitsApplyToYour') : planLimits(plan)}
      </Text>
      {usage.upgrade_url && !unlimited ? (
        // The server sells plans: the pricing page compares them and has the checkout.
        <Text size="xs" c="dimmed" mt={2}>
          <Trans
            i18nKey="ui.usage.UsageSection.needMoreComparePlansAnd"
            components={[<Anchor component={Link} to="/pricing" size="xs" />]}
          />
        </Text>
      ) : (
        others.length > 0 &&
        !unlimited && (
          <Text size="xs" c="dimmed" mt={2}>
            {i18n.t('ui.usage.UsageSection.otherPlans', { map: others.map(otherPlan).join(' · ') })}
          </Text>
        )
      )}
    </div>
  );
}

function OverageForm({ usage, onSaved }: { usage: Usage; onSaved: (usage: Usage) => void }) {
  useTranslation();
  const overage = usage.overage!;
  const [enabled, setEnabled] = useState(overage.enabled);
  const [cap, setCap] = useState<string | number>(overage.cap_cpu_hours ?? '');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const changed =
    enabled !== overage.enabled || (cap === '' ? null : Number(cap)) !== overage.cap_cpu_hours;
  const price = formatMoney(overage.price_per_cpu_hour, overage.currency);

  const save = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onSaved(await api.account.setOverage(enabled, cap === '' ? null : Number(cap)));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={save}>
      <Stack gap="xs">
        <Switch
          label={i18n.t('ui.usage.UsageSection.allowComputeBeyondTheMonthly')}
          description={i18n.t('ui.usage.UsageSection.billedAtPerCpuHour', { price })}
          checked={enabled}
          onChange={(event) => setEnabled(event.currentTarget.checked)}
        />
        {enabled && (
          <NumberInput
            label={i18n.t('ui.usage.UsageSection.mostBeyondTheAllowance')}
            description={i18n.t('ui.usage.UsageSection.cpuHoursPerMonthEmpty')}
            placeholder={i18n.t('ui.usage.UsageSection.noCap')}
            min={0}
            decimalScale={2}
            suffix=" CPU h"
            value={cap}
            onChange={setCap}
            maw="16rem"
          />
        )}
        {overage.used_cpu_minutes > 0 && (
          <Text size="xs" c="dimmed">
            {i18n.t('ui.usage.UsageSection.beyondTheAllowanceThisMonth')}{' '}
            {formatCpu(overage.used_cpu_minutes)}
            {overage.cost !== null && ` (${formatMoney(overage.cost, overage.currency)})`}
          </Text>
        )}
        {error && (
          <Text size="xs" c="red">
            {error}
          </Text>
        )}
        {changed && (
          <Group justify="flex-end">
            <Button type="submit" size="xs" loading={busy}>
              {i18n.t('ui.usage.UsageSection.save')}
            </Button>
          </Group>
        )}
      </Stack>
    </form>
  );
}

/** The account page's "Usage & budget": the plan, compute this month, storage, budgets. */
export function UsageSection() {
  useTranslation();
  const [usage, setUsage] = useState<Usage | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api.account
      .usage(controller.signal)
      .then(setUsage)
      .catch((err: Error) => {
        if (!controller.signal.aborted) {
          setError(err.message);
        }
      });
    return () => controller.abort();
  }, []);

  if (error) {
    return (
      <Alert color="red" icon={<IconAlertCircle size={18} />}>
        {error}
      </Alert>
    );
  }
  if (!usage) {
    return <Loader size="sm" />;
  }

  const { compute, storage, prices } = usage;
  const currency = prices.currency;
  const resets = formatResetDate(usage.period.resets_at);
  const cost = formatMoney(prices.compute_cost, currency);
  const limitCost = formatMoney(prices.compute_limit_cost, currency);
  const computeValue =
    compute.limit !== null
      ? `${formatCpu(compute.used, '')} of ${formatCpu(compute.limit)}`
      : `${formatCpu(compute.used)} · no limit`;
  const money = cost && (limitCost && compute.limit !== null ? `${cost} of ${limitCost}` : cost);

  const save = async (budget: number | null) => {
    setUsage(await api.account.setBudget(budget));
  };

  return (
    <Stack gap="md">
      <PlanInfo usage={usage} />
      <Meter
        label={i18n.t('ui.usage.UsageSection.computeThisMonth')}
        used={compute.used}
        limit={compute.limit}
        value={money ? `${computeValue} (${money})` : computeValue}
        detail={i18n.t('ui.usage.UsageSection.cpuTimeOfTrainingsAnd', {
          breakdown: breakdown(usage),
          resets,
        })}
      />
      <Meter
        label={i18n.t('ui.usage.UsageSection.storage')}
        used={storage.used_bytes}
        limit={storage.limit_bytes}
        value={
          storage.limit_bytes !== null
            ? `${formatBytes(storage.used_bytes)} of ${formatBytes(storage.limit_bytes)}`
            : `${formatBytes(storage.used_bytes)} · no limit`
        }
        detail={i18n.t('ui.usage.UsageSection.importedTablesDatasetsTrainedWeights', {
          data_sources: formatBytes(storage.by_kind.data_sources),
          datasets: formatBytes(storage.by_kind.datasets),
          training_runs: formatBytes(storage.by_kind.training_runs),
        })}
      />
      <Group justify="space-between" align="center" gap="sm">
        <Text size="xs" c="dimmed" style={{ flex: '1 1 14rem' }}>
          {i18n.t('ui.usage.UsageSection.apiTokensAndAppsCan', {
            value:
              compute.own_budget !== null
                ? i18n.t('ui.usage.UsageSection.yourMonthlyBudget', {
                    own_budget: formatCpu(compute.own_budget),
                  })
                : i18n.t('ui.usage.UsageSection.setAMonthlyBudgetTo'),
            value2:
              compute.allowance !== null
                ? i18n.t('ui.usage.UsageSection.yourPlanAllowsPerMonth', {
                    allowance: formatCpu(compute.allowance),
                  })
                : i18n.t('ui.usage.UsageSection.noMonthlyLimitApplies'),
            value3:
              compute.run_limit !== null
                ? i18n.t('ui.usage.UsageSection.andPerTraining', {
                    run_limit: formatCpu(compute.run_limit),
                  })
                : '.',
          })}
        </Text>
        <BudgetEditor
          budget={compute.own_budget}
          label={i18n.t('ui.usage.UsageSection.yourAccount')}
          onSave={save}
          description={i18n.t('ui.usage.UsageSection.cpuMinutesPerMonthFor')}
        />
      </Group>
      {usage.overage?.available && (
        <OverageForm key={String(usage.overage.enabled)} usage={usage} onSaved={setUsage} />
      )}
    </Stack>
  );
}
