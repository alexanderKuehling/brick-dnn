import { useMemo, useState } from 'react';
import {
  IconAlertTriangle,
  IconCheck,
  IconChevronDown,
  IconChevronRight,
  IconCircleCheck,
  IconInfoCircle,
} from '@tabler/icons-react';
import { Anchor, Badge, Button, Group, Paper, Stack, Text, UnstyledButton } from '@mantine/core';
import type { DataFix, DataProfile, DataWarning, NewStep } from '@/api/types';
import { formatCount } from '@/components/charts/format';
import { usePrepAccess } from '../prepAccess';
import { groupWarnings, type WarningGroup } from './warningGroups';

export interface WarningListProps {
  warnings: DataWarning[];
  /** Fix buttons are disabled once clicked until this object changes (a new preview). */
  resetKey: unknown;
  onAddStep: (step: NewStep) => void;
  onShowColumn?: (column: string) => void;
  /** Shown when there are no warnings; nothing if omitted. */
  emptyText?: string;
  /** For the ranges in the titles of grouped warnings ("6–9 % of the rows"). */
  profile?: DataProfile | null;
}

/** Messages of a group listed before "Show all". */
const GROUP_PREVIEW = 20;

function SeverityIcon({ severity }: { severity: DataWarning['severity'] }) {
  const isWarning = severity === 'warning';
  const Icon = isWarning ? IconAlertTriangle : IconInfoCircle;
  return (
    <Icon
      size={18}
      role="img"
      aria-label={isWarning ? 'Warning' : 'Note'}
      color={isWarning ? 'var(--chart-status-warning)' : 'var(--mantine-color-dimmed)'}
      style={{ flexShrink: 0, marginTop: 2 }}
    />
  );
}

function FixButton({ fix, added, onAdd }: { fix: DataFix; added: boolean; onAdd: () => void }) {
  return (
    <Button
      size="compact-sm"
      variant="light"
      disabled={added}
      leftSection={added ? <IconCheck size={14} /> : undefined}
      onClick={onAdd}
    >
      {added ? 'Step added' : fix.label}
    </Button>
  );
}

function GroupEntry({
  group,
  readOnly,
  added,
  onAdd,
  onShowColumn,
}: {
  group: WarningGroup;
  readOnly: boolean;
  added: boolean;
  onAdd: () => void;
  onShowColumn?: (column: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [all, setAll] = useState(false);
  const shown = all ? group.warnings : group.warnings.slice(0, GROUP_PREVIEW);
  const Chevron = open ? IconChevronDown : IconChevronRight;
  return (
    <Paper withBorder p="xs" radius="sm">
      <Group gap="sm" wrap="wrap" align="flex-start">
        <Group gap="sm" wrap="nowrap" align="flex-start" style={{ flex: '1 1 240px', minWidth: 0 }}>
          <SeverityIcon severity={group.severity} />
          <Stack gap={4} style={{ minWidth: 0 }}>
            <Text size="sm">{group.title}</Text>
            <Group gap="xs">
              <Badge size="xs" variant="light" color="gray">
                {group.code.replace(/_/g, ' ')}
              </Badge>
              <UnstyledButton
                onClick={() => setOpen((value) => !value)}
                aria-expanded={open}
                style={{ display: 'inline-flex', alignItems: 'center', gap: 2 }}
              >
                <Chevron size={14} aria-hidden="true" />
                <Text size="xs" c="var(--mantine-color-anchor)">
                  {open ? 'Hide' : 'Show'} the {formatCount(group.columns.length)} columns
                </Text>
              </UnstyledButton>
            </Group>
          </Stack>
        </Group>
        {group.fix && !readOnly && <FixButton fix={group.fix} added={added} onAdd={onAdd} />}
      </Group>
      {open && (
        <Stack gap={2} mt={6} ml={30}>
          {shown.map((warning) => (
            <Group key={warning.column} gap={6} wrap="nowrap">
              {onShowColumn ? (
                <Anchor
                  component="button"
                  type="button"
                  size="xs"
                  onClick={() => onShowColumn(warning.column!)}
                  style={{ flexShrink: 0 }}
                >
                  {warning.column}
                </Anchor>
              ) : null}
              <Text size="xs" c="dimmed" truncate>
                {warning.message}
              </Text>
            </Group>
          ))}
          {group.warnings.length > shown.length && (
            <Button
              size="compact-xs"
              variant="subtle"
              onClick={() => setAll(true)}
              style={{ alignSelf: 'flex-start' }}
            >
              Show all {formatCount(group.warnings.length)}
            </Button>
          )}
        </Stack>
      )}
    </Paper>
  );
}

/**
 * Profile warnings, most serious first, each with its one-click fix. Warnings of one kind
 * about many columns are grouped into one entry (expandable) with one fix for all of them.
 */
export function WarningList({
  warnings,
  resetKey,
  onAddStep,
  onShowColumn,
  emptyText,
  profile = null,
}: WarningListProps) {
  // Viewers of a shared preparation see the warnings without their fixes.
  const { readOnly } = usePrepAccess();
  const [added, setAdded] = useState<{ resetKey: unknown; keys: Set<string> }>({
    resetKey,
    keys: new Set(),
  });
  const addedKeys = added.resetKey === resetKey ? added.keys : new Set<string>();
  const items = useMemo(() => {
    const sorted = [...warnings].sort(
      (a, b) => Number(b.severity === 'warning') - Number(a.severity === 'warning')
    );
    return groupWarnings(sorted, profile);
  }, [warnings, profile]);
  const add = (key: string, fix: DataFix) => {
    onAddStep({ type: fix.type, params: fix.params });
    setAdded({ resetKey, keys: new Set([...addedKeys, key]) });
  };

  if (items.length === 0) {
    return emptyText ? (
      <Group gap={6}>
        <IconCircleCheck size={18} color="var(--chart-status-good)" aria-hidden="true" />
        <Text size="sm">{emptyText}</Text>
      </Group>
    ) : null;
  }

  return (
    <Stack gap={6}>
      {items.map((item) => {
        if (item.type === 'group') {
          return (
            <GroupEntry
              key={item.key}
              group={item.group}
              readOnly={readOnly}
              added={addedKeys.has(item.key)}
              onAdd={() => item.group.fix && add(item.key, item.group.fix)}
              onShowColumn={onShowColumn}
            />
          );
        }
        const { warning, key } = item;
        return (
          <Paper key={key} withBorder p="xs" radius="sm">
            <Group gap="sm" wrap="wrap" align="flex-start">
              <Group
                gap="sm"
                wrap="nowrap"
                align="flex-start"
                style={{ flex: '1 1 240px', minWidth: 0 }}
              >
                <SeverityIcon severity={warning.severity} />
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
              {warning.fix && !readOnly && (
                <FixButton
                  fix={warning.fix}
                  added={addedKeys.has(key)}
                  onAdd={() => add(key, warning.fix!)}
                />
              )}
            </Group>
          </Paper>
        );
      })}
    </Stack>
  );
}
