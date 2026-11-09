import { IconAlertTriangle, IconGauge, IconInfoCircle } from '@tabler/icons-react';
import { Alert, Anchor, Group, Paper, Stack, Text, Tooltip } from '@mantine/core';
import { formatCpu, formatMoney, formatResetDate } from './format';
import type { CostEstimateState } from './useCostEstimate';

/**
 * The account page with the usage and budgets, in a new tab: the training panel keeps its
 * network and settings (it is also shown outside of the router in tests).
 */
export function AccountLink() {
  return (
    <Anchor href="/account" target="_blank" rel="noopener" size="sm">
      Usage & budget
    </Anchor>
  );
}

/** "≈ 2.1 CPU min (0.7–6.3) · €0.02". */
export function estimateText(state: Pick<CostEstimateState, 'estimate'>): string | null {
  const { estimate } = state;
  if (!estimate) {
    return null;
  }
  const range = `${formatCpu(estimate.low_cpu_minutes, '')}–${formatCpu(estimate.high_cpu_minutes, '')}`;
  const money = formatMoney(estimate.cost, estimate.currency);
  return `≈ ${formatCpu(estimate.cpu_minutes)} (${range})${money ? ` · ${money}` : ''}`;
}

/** "Up to 30 CPU min for this run (the limit per training) · 287 CPU min left this month". */
export function budgetText(state: Pick<CostEstimateState, 'budget'>): string | null {
  const { budget } = state;
  if (!budget || !budget.allowed) {
    return null;
  }
  const parts: string[] = [];
  if (budget.run_limit_cpu_minutes !== null) {
    const why =
      budget.limit_reason === 'run_limit'
        ? 'the limit per training'
        : 'what is left of your budget';
    parts.push(`It may use up to ${formatCpu(budget.run_limit_cpu_minutes)} (${why}).`);
  }
  if (budget.remaining_cpu_minutes != null) {
    parts.push(
      `${formatCpu(budget.remaining_cpu_minutes)} left this month, until ${formatResetDate(budget.resets_at)}.`
    );
  }
  return parts.length > 0 ? parts.join(' ') : null;
}

/**
 * The training panel's estimate of the planned run: its CPU time (a rough range), what it may
 * use within the budgets, and why a start would be refused or probably be stopped.
 */
export function CostEstimate({ state }: { state: CostEstimateState }) {
  const { estimate, budget, loading } = state;
  if (!estimate && !budget) {
    return null;
  }
  const text = estimateText(state);
  const limits = budgetText(state);
  return (
    <Stack gap="xs" style={{ opacity: loading ? 0.6 : 1, transition: 'opacity 150ms' }}>
      {estimate && (
        <Paper component="section" withBorder p="sm" radius="md" aria-label="Estimated compute">
          <Group justify="space-between" gap="xs" wrap="wrap">
            <Group gap={6} wrap="nowrap">
              <IconGauge size={16} color="var(--mantine-color-dimmed)" aria-hidden />
              <Text size="sm" fw={600}>
                Estimated compute
              </Text>
              <Tooltip label={`${estimate.basis}. ${estimate.note}`} multiline w={280} withArrow>
                <IconInfoCircle
                  size={14}
                  color="var(--mantine-color-dimmed)"
                  aria-label="How the estimate is made"
                />
              </Tooltip>
            </Group>
            <Text size="sm">{text}</Text>
          </Group>
          {limits && (
            <Text size="xs" c="dimmed" mt={4}>
              {limits}
            </Text>
          )}
        </Paper>
      )}
      {budget && !budget.allowed && budget.reason && (
        <Alert
          color="red"
          icon={<IconAlertTriangle size={18} />}
          title={
            budget.code === 'storage_quota_exceeded'
              ? 'The storage of your account is full'
              : budget.code === 'estimate_too_high'
                ? 'Too much compute for your budget'
                : 'Not enough compute budget'
          }
        >
          <Text size="sm">{budget.reason}</Text>
          <AccountLink />
        </Alert>
      )}
      {budget?.allowed && budget.warning && (
        <Alert color="yellow" icon={<IconInfoCircle size={18} />}>
          <Text size="sm">{budget.warning}</Text>
        </Alert>
      )}
    </Stack>
  );
}
