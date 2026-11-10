import { useState } from 'react';
import { IconGauge } from '@tabler/icons-react';
import { Button, Group, NumberInput, Popover, Stack, Text } from '@mantine/core';
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
          aria-label={`Set the monthly budget of ${label}`}
        >
          {budget === null ? 'Set budget' : 'Change budget'}
        </Button>
      </Popover.Target>
      <Popover.Dropdown maw="min(20rem, calc(100vw - 2rem))">
        <form onSubmit={submit}>
          <Stack gap="xs">
            <NumberInput
              label="Monthly budget"
              description={description ?? 'CPU minutes per month. Empty: no budget.'}
              placeholder="No budget"
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
                  Remove budget
                </Button>
              ) : (
                <span />
              )}
              <Button type="submit" size="xs" loading={busy}>
                Save
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
  return budget == null ? spent : `${spent} of ${formatCpu(budget)}`;
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
