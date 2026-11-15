import { useState } from 'react';
import { IconGauge } from '@tabler/icons-react';
import { Button, Group, NumberInput, Popover, Stack, Text, VisuallyHidden } from '@mantine/core';
import { i18n, useTranslation } from '@/i18n';
import { formatCpu } from './format';

interface BudgetEditorProps {
  /** The current monthly budget in CPU minutes; null: none. */
  budget: number | null;
  /** What it is a budget of, for the accessible label, e.g. “agent” or "your account". */
  label: string;
  /** Saves the new budget (null removes it); throws with a message on failure. */
  onSave: (budget: number | null) => Promise<void>;
  description?: string;
}

/**
 * A compact button that opens a small form for a monthly compute budget in CPU minutes. An
 * empty field removes the budget.
 */
export function BudgetEditor({ budget, label, onSave, description }: BudgetEditorProps) {
  useTranslation();
  const [opened, setOpened] = useState(false);
  const [value, setValue] = useState<string | number>(budget ?? '');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const open = () => {
    setValue(budget ?? '');
    setError(null);
    setOpened(true);
  };

  const save = async (next: number | null) => {
    setBusy(true);
    setError(null);
    try {
      await onSave(next);
      setOpened(false);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    void save(value === '' ? null : Number(value));
  };

  return (
    <Popover opened={opened} onChange={setOpened} position="bottom-end" withArrow shadow="md">
      <Popover.Target>
        <Button
          variant="subtle"
          size="compact-xs"
          leftSection={<IconGauge size={14} />}
          onClick={() => (opened ? setOpened(false) : open())}
        >
          {budget === null
            ? i18n.t('ui.usage.BudgetEditor.setBudget')
            : i18n.t('ui.usage.BudgetEditor.changeBudget')}
          {/* Which budget, for screen readers; the visible text comes first (WCAG 2.5.3). */}
          <VisuallyHidden>{`: ${label}`}</VisuallyHidden>
        </Button>
      </Popover.Target>
      <Popover.Dropdown maw="min(20rem, calc(100vw - 2rem))">
        <form onSubmit={submit}>
          <Stack gap="xs">
            <NumberInput
              label={i18n.t('ui.usage.BudgetEditor.monthlyBudget')}
              description={description ?? i18n.t('ui.usage.BudgetEditor.cpuMinutesPerMonthEmpty')}
              placeholder={i18n.t('ui.usage.BudgetEditor.noBudget')}
              min={0}
              max={10_000_000}
              decimalScale={2}
              suffix=" CPU min"
              value={value}
              onChange={setValue}
              error={error}
              data-autofocus
            />
            <Group justify="space-between" gap="xs">
              {budget !== null ? (
                <Button
                  variant="subtle"
                  color="gray"
                  size="xs"
                  onClick={() => void save(null)}
                  disabled={busy}
                >
                  {i18n.t('ui.usage.BudgetEditor.removeBudget')}
                </Button>
              ) : (
                <span />
              )}
              <Button type="submit" size="xs" loading={busy}>
                {i18n.t('ui.usage.BudgetEditor.save')}
              </Button>
            </Group>
          </Stack>
        </form>
      </Popover.Dropdown>
    </Popover>
  );
}

/** "2.5 of 10 CPU min" or "2.5 CPU min" (without a budget). */
export function usedOfBudget(used: number | undefined, budget: number | null | undefined): string {
  const spent = formatCpu(used ?? 0, budget == null ? 'CPU min' : '');
  return budget == null
    ? spent
    : i18n.t('ui.usage.BudgetEditor.of', { spent, budget: formatCpu(budget) });
}

/** The usage of a token or app this month, red once its budget is used up. */
export function MonthUsage({ used, budget }: { used?: number; budget?: number | null }) {
  const exhausted = budget != null && (used ?? 0) >= budget;
  return (
    <Text component="span" size="sm" c={exhausted ? 'red' : undefined}>
      {usedOfBudget(used, budget)}
    </Text>
  );
}
