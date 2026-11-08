import { useState } from 'react';
import { IconAlertTriangle, IconCheck, IconCircleCheck, IconInfoCircle } from '@tabler/icons-react';
import { Anchor, Badge, Button, Group, Paper, Stack, Text } from '@mantine/core';
import type { DataWarning, NewStep } from '@/api/types';

export interface WarningListProps {
  warnings: DataWarning[];
  /** Fix buttons are disabled once clicked until this object changes (a new preview). */
  resetKey: unknown;
  onAddStep: (step: NewStep) => void;
  onShowColumn?: (column: string) => void;
  /** Shown when there are no warnings; nothing if omitted. */
  emptyText?: string;
}

const keyOf = (w: DataWarning, i: number) => `${w.code}:${w.column ?? ''}:${i}`;

/** Profile warnings, most serious first, each with its one-click fix. */
export function WarningList({
  warnings,
  resetKey,
  onAddStep,
  onShowColumn,
  emptyText,
}: WarningListProps) {
  const [added, setAdded] = useState<{ resetKey: unknown; keys: Set<string> }>({
    resetKey,
    keys: new Set(),
  });
  const addedKeys = added.resetKey === resetKey ? added.keys : new Set<string>();
  const sorted = [...warnings].sort(
    (a, b) => Number(b.severity === 'warning') - Number(a.severity === 'warning')
  );

  if (sorted.length === 0) {
    return emptyText ? (
      <Group gap={6}>
        <IconCircleCheck size={18} color="var(--chart-status-good)" aria-hidden="true" />
        <Text size="sm">{emptyText}</Text>
      </Group>
    ) : null;
  }

  return (
    <Stack gap={6}>
      {sorted.map((warning, i) => {
        const key = keyOf(warning, i);
        const isAdded = addedKeys.has(key);
        const isWarning = warning.severity === 'warning';
        const Icon = isWarning ? IconAlertTriangle : IconInfoCircle;
        return (
          <Paper key={key} withBorder p="xs" radius="sm">
            <Group gap="sm" wrap="wrap" align="flex-start">
              <Group
                gap="sm"
                wrap="nowrap"
                align="flex-start"
                style={{ flex: '1 1 240px', minWidth: 0 }}
              >
                <Icon
                  size={18}
                  role="img"
                  aria-label={isWarning ? 'Warning' : 'Note'}
                  color={isWarning ? 'var(--chart-status-warning)' : 'var(--mantine-color-dimmed)'}
                  style={{ flexShrink: 0, marginTop: 2 }}
                />
                <Stack gap={4} style={{ minWidth: 0 }}>
                  <Text size="sm">{warning.message}</Text>
                  <Group gap="xs">
                    <Badge size="xs" variant="light" color="gray">
                      {warning.code.replace(/_/g, ' ')}
                    </Badge>
                    {warning.column && onShowColumn && (
                      <Anchor
                        component="button"
                        type="button"
                        size="xs"
                        onClick={() => onShowColumn(warning.column!)}
                      >
                        Show {warning.column}
                      </Anchor>
                    )}
                  </Group>
                </Stack>
              </Group>
              {warning.fix && (
                <Button
                  size="compact-sm"
                  variant="light"
                  disabled={isAdded}
                  leftSection={isAdded ? <IconCheck size={14} /> : undefined}
                  onClick={() => {
                    onAddStep({ type: warning.fix!.type, params: warning.fix!.params });
                    setAdded({ resetKey, keys: new Set([...addedKeys, key]) });
                  }}
                >
                  {isAdded ? 'Step added' : warning.fix.label}
                </Button>
              )}
            </Group>
          </Paper>
        );
      })}
    </Stack>
  );
}
