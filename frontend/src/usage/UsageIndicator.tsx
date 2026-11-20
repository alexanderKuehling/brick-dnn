import { IconAlertTriangle, IconChartBar } from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import { Anchor, Group, Stack, Text, VisuallyHidden } from '@mantine/core';
import type { PlanOffer, Usage } from '@/api';
import { offerText, pricingPath } from '@/billing/LimitNotice';
import { currentLocale, i18n, useTranslation } from '@/i18n';
import { formatBytes } from '@/training/format';
import { formatCpu, formatResetDate, usagePercent } from './format';

/** From this share of a limit on the indicator warns, and at 100 % it says so. */
export const NEARLY = 80;

export type UsageLevel = 'ok' | 'nearly' | 'full';

export function usageLevel(percent: number): UsageLevel {
  return percent >= 100 ? 'full' : percent >= NEARLY ? 'nearly' : 'ok';
}

/** "18 of 30 CPU min", from two hours of allowance on "0.3 of 10 CPU h". */
export function computeText(usedMinutes: number, limitMinutes: number): string {
  if (limitMinutes < 120) {
    return i18n.t('plans.usage.compute', {
      used: formatCpu(usedMinutes, ''),
      limit: formatCpu(limitMinutes),
    });
  }
  const hours = (minutes: number) =>
    (minutes / 60).toLocaleString(currentLocale(), { maximumFractionDigits: 1 });
  return i18n.t('plans.usage.compute', {
    used: hours(usedMinutes),
    limit: i18n.t('plans.short.hours', { value: hours(limitMinutes) }),
  });
}

interface Warning {
  level: UsageLevel;
  text: string;
  offer: PlanOffer | null;
}

/** What to say about the limit that is closest to being used up (null: nothing). */
function warningOf(usage: Usage): Warning | null {
  const { compute, storage } = usage;
  const computeShare = compute.limit !== null ? usagePercent(compute.used, compute.limit) : 0;
  const storageShare =
    storage.limit_bytes !== null ? usagePercent(storage.used_bytes, storage.limit_bytes) : 0;
  const next = usage.next_plans;
  if (computeShare >= NEARLY && computeShare >= storageShare) {
    const full = computeShare >= 100;
    return {
      level: usageLevel(computeShare),
      text: full
        ? i18n.t('plans.usage.computeFull', { date: formatResetDate(usage.period.resets_at) })
        : i18n.t('plans.usage.computeNearly', { percent: Math.floor(computeShare) }),
      // The plan's allowance is the limit, not a budget the user set for the account.
      offer: compute.limit === compute.allowance ? (next?.compute ?? null) : null,
    };
  }
  if (storageShare >= NEARLY) {
    return {
      level: usageLevel(storageShare),
      text:
        storageShare >= 100
          ? i18n.t('plans.usage.storageFull')
          : i18n.t('plans.usage.storageNearly', { percent: Math.floor(storageShare) }),
      offer: next?.storage ?? null,
    };
  }
  return null;
}

/**
 * The usage where trainings are started: the compute and the storage of the month in one
 * quiet line. From 80 % of a limit it says so in a sentence, names the next plan where plans
 * are sold and links to them (else to the account page); at 100 % likewise. Nothing for an
 * account without limits. `newTab`: inside the training panel, links open in a new tab.
 */
export function UsageIndicator({
  usage,
  newTab = false,
  quiet = false,
}: {
  usage: Usage | null;
  newTab?: boolean;
  /** Only the line: the message of the limit is on the screen already. */
  quiet?: boolean;
}) {
  useTranslation();
  if (!usage || usage.unlimited) {
    return null;
  }
  const { compute, storage } = usage;
  if (compute.limit === null && storage.limit_bytes === null) {
    return null;
  }
  const parts: string[] = [];
  if (compute.limit !== null) {
    parts.push(computeText(compute.used, compute.limit));
  }
  if (storage.limit_bytes !== null) {
    parts.push(
      i18n.t('plans.usage.storage', {
        used: formatBytes(storage.used_bytes),
        limit: formatBytes(storage.limit_bytes),
      })
    );
  }
  const warning = quiet ? null : warningOf(usage);
  const selling = Boolean(usage.upgrade_url);
  const target = selling ? pricingPath(warning?.offer?.key) : '/account';
  const label = selling ? i18n.t('plans.upgrade.seePlans') : i18n.t('plans.upgrade.usageAndBudget');
  return (
    <Stack gap={2} data-testid="usage-indicator" data-level={warning?.level ?? 'ok'}>
      <Group gap={6} wrap="nowrap" align="flex-start">
        <IconChartBar
          size={14}
          color="var(--mantine-color-dimmed)"
          style={{ flexShrink: 0, marginTop: 2 }}
          aria-hidden
        />
        <Text size="xs" c="dimmed">
          {i18n.t('plans.usage.label')}: {parts.join(' · ')}
        </Text>
      </Group>
      {warning && (
        <Group gap={6} wrap="nowrap" align="flex-start">
          <IconAlertTriangle
            size={14}
            color={`var(--mantine-color-${warning.level === 'full' ? 'red' : 'yellow'}-filled)`}
            style={{ flexShrink: 0, marginTop: 2 }}
            aria-hidden
          />
          <Text size="xs">
            {warning.text}
            {warning.offer && selling && ` ${offerText(warning.offer, 'usage')}`}{' '}
            {newTab ? (
              <Anchor href={target} target="_blank" rel="noopener" size="xs">
                {label}
                <VisuallyHidden> {i18n.t('plans.upgrade.newTab')}</VisuallyHidden>
              </Anchor>
            ) : (
              <Anchor component={Link} to={target} size="xs">
                {label}
              </Anchor>
            )}
          </Text>
        </Group>
      )}
    </Stack>
  );
}
