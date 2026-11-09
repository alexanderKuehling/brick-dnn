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
          aria-label={`${label}: ${Math.round(percent)}% used`}
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
      ? `${plan.storage_gb.toLocaleString('en-US')} GB`
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
  return `${hours.toLocaleString('en-US', { maximumFractionDigits: 2 })} CPU h`;
}

function planPrice(plan: Plan): string {
  if (plan.price === null) {
    return '';
  }
  return plan.price === 0 ? 'no charge' : `${formatMoney(plan.price, plan.currency)} per month`;
}

/** "Starter (€5.00 per month): 10 CPU h, 5 GB". */
function otherPlan(plan: Plan): string {
  const limits = [
    plan.cpu_hours_per_month !== null ? formatHours(plan.cpu_hours_per_month) : 'unlimited',
    plan.storage_gb !== null ? `${plan.storage_gb.toLocaleString('en-US')} GB` : null,
  ].filter(Boolean);
  const price = planPrice(plan);
  return `${plan.name}${price ? ` (${price})` : ''}: ${limits.join(', ')}`;
}

function PlanInfo({ usage }: { usage: Usage }) {
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
          {plan.key === 'default' ? 'Your account' : `${plan.name} plan`}
        </Text>
        {price && !unlimited && (
          <Text size="sm" c="dimmed">
            {price}
          </Text>
        )}
        {unlimited && (
          <Badge variant="light" color="green">
            Unlimited
          </Badge>
        )}
      </Group>
      <Text size="xs" c="dimmed" mt={2}>
        {unlimited
          ? 'No limits apply to your account; its usage is still shown here. Budgets you set yourself still apply.'
          : planLimits(plan)}
      </Text>
      {usage.upgrade_url && !unlimited ? (
        // The server sells plans: the pricing page compares them and has the checkout.
        <Text size="xs" c="dimmed" mt={2}>
          Need more?{' '}
          <Anchor component={Link} to="/pricing" size="xs">
            Compare plans and prices
          </Anchor>
        </Text>
      ) : (
        others.length > 0 &&
        !unlimited && (
          <Text size="xs" c="dimmed" mt={2}>
            Other plans: {others.map(otherPlan).join(' · ')}.
          </Text>
        )
      )}
    </div>
  );
}

function OverageForm({ usage, onSaved }: { usage: Usage; onSaved: (usage: Usage) => void }) {
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
          label="Allow compute beyond the monthly allowance"
          description={`Billed at ${price} per CPU hour at the end of the month. Off: trainings stop when the allowance is used up.`}
          checked={enabled}
          onChange={(event) => setEnabled(event.currentTarget.checked)}
        />
        {enabled && (
          <NumberInput
            label="Most beyond the allowance"
            description="CPU hours per month. Empty: no cap."
            placeholder="No cap"
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
            Beyond the allowance this month: {formatCpu(overage.used_cpu_minutes)}
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
              Save
            </Button>
          </Group>
        )}
      </Stack>
    </form>
  );
}

/** The account page's "Usage & budget": the plan, compute this month, storage, budgets. */
export function UsageSection() {
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
        label="Compute this month"
        used={compute.used}
        limit={compute.limit}
        value={money ? `${computeValue} (${money})` : computeValue}
        detail={`CPU time of trainings and predictions; ${breakdown(usage)}. Resets on ${resets} (UTC).`}
      />
      <Meter
        label="Storage"
        used={storage.used_bytes}
        limit={storage.limit_bytes}
        value={
          storage.limit_bytes !== null
            ? `${formatBytes(storage.used_bytes)} of ${formatBytes(storage.limit_bytes)}`
            : `${formatBytes(storage.used_bytes)} · no limit`
        }
        detail={`Imported tables ${formatBytes(storage.by_kind.data_sources)} · datasets ${formatBytes(storage.by_kind.datasets)} · trained weights ${formatBytes(storage.by_kind.training_runs)}`}
      />
      <Group justify="space-between" align="center" gap="sm">
        <Text size="xs" c="dimmed" style={{ flex: '1 1 14rem' }}>
          {compute.own_budget !== null
            ? `Your monthly budget: ${formatCpu(compute.own_budget)}. `
            : 'Set a monthly budget to limit what your account may use. '}
          {compute.allowance !== null
            ? `Your plan allows ${formatCpu(compute.allowance)} per month`
            : 'No monthly limit applies'}
          {compute.run_limit !== null ? ` and ${formatCpu(compute.run_limit)} per training.` : '.'}{' '}
          API tokens and apps can have budgets of their own (API tokens page).
        </Text>
        <BudgetEditor
          budget={compute.own_budget}
          label="your account"
          onSave={save}
          description="CPU minutes per month for your whole account. Empty: no budget of your own."
        />
      </Group>
      {usage.overage?.available && (
        <OverageForm key={String(usage.overage.enabled)} usage={usage} onSaved={setUsage} />
      )}
    </Stack>
  );
}
