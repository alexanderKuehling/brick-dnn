import { useState } from 'react';
import { IconShieldCheck } from '@tabler/icons-react';
import { Button, Checkbox, Group, NumberInput, Popover, Stack, Text } from '@mantine/core';
import type { ApprovalRule } from '@/api';
import { ruleSummary } from './format';
import classes from './Approvals.module.css';

interface ApprovalRuleEditorProps {
  rule: ApprovalRule;
  /** Whose rule it is, for the accessible label, e.g. “Claude on my laptop”. */
  label: string;
  /** Saves the rule; throws with a message on failure. */
  onSave: (rule: ApprovalRule) => Promise<void>;
}

/**
 * What a token or app may do without asking, as a sentence, and a small form to change it:
 * trainings above N CPU minutes, publishing and deleting.
 */
export function ApprovalRuleEditor({ rule, label, onSave }: ApprovalRuleEditorProps) {
  const [opened, setOpened] = useState(false);
  const [training, setTraining] = useState(rule.training_cpu_minutes !== null);
  const [minutes, setMinutes] = useState<string | number>(rule.training_cpu_minutes ?? 15);
  const [publishing, setPublishing] = useState(rule.publishing);
  const [deleting, setDeleting] = useState(rule.deleting);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const open = () => {
    setTraining(rule.training_cpu_minutes !== null);
    setMinutes(rule.training_cpu_minutes ?? 15);
    setPublishing(rule.publishing);
    setDeleting(rule.deleting);
    setError(null);
    setOpened(true);
  };

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (training && (minutes === '' || Number(minutes) < 0)) {
      setError('Enter the CPU minutes (0: ask before every training).');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await onSave({
        training_cpu_minutes: training ? Number(minutes) : null,
        publishing,
        deleting,
      });
      setOpened(false);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Group gap={6} wrap="wrap" align="center">
      <Text span size="sm" className={classes.ruleText}>
        {ruleSummary(rule)}
      </Text>
      <Popover opened={opened} onChange={setOpened} position="bottom-end" withArrow shadow="md">
        <Popover.Target>
          <Button
            variant="subtle"
            size="compact-xs"
            leftSection={<IconShieldCheck size={14} />}
            onClick={() => (opened ? setOpened(false) : open())}
            aria-label={`Change what ${label} may do without asking`}
          >
            Change
          </Button>
        </Popover.Target>
        <Popover.Dropdown maw="min(22rem, calc(100vw - 2rem))">
          <form onSubmit={submit}>
            <Stack gap="sm">
              <Text size="sm" fw={600}>
                Ask me first before …
              </Text>
              <div>
                <Checkbox
                  label="Trainings above"
                  checked={training}
                  onChange={(event) => setTraining(event.currentTarget.checked)}
                />
                <NumberInput
                  mt={6}
                  ml={30}
                  aria-label="CPU minutes a training may use without asking"
                  description="Estimated CPU minutes; 0: every training"
                  min={0}
                  max={10_000_000}
                  decimalScale={2}
                  suffix=" CPU min"
                  value={minutes}
                  onChange={setMinutes}
                  disabled={!training}
                />
              </div>
              <Checkbox
                label="Publishing"
                description="Weights links that work without logging in, sharing data in a project"
                checked={publishing}
                onChange={(event) => setPublishing(event.currentTarget.checked)}
              />
              <Checkbox
                label="Deleting"
                description="Networks, training runs, datasets and data"
                checked={deleting}
                onChange={(event) => setDeleting(event.currentTarget.checked)}
              />
              {error && (
                <Text size="sm" c="red">
                  {error}
                </Text>
              )}
              <Group justify="flex-end">
                <Button type="submit" size="xs" loading={busy}>
                  Save
                </Button>
              </Group>
            </Stack>
          </form>
        </Popover.Dropdown>
      </Popover>
    </Group>
  );
}
