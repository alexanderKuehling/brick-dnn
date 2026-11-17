import { IconColumns } from '@tabler/icons-react';
import { Button, Checkbox, Group, Popover, ScrollArea, Stack, Text } from '@mantine/core';
import { useTranslation } from '@/i18n';
import type { Column, ColumnGroup } from './columns';

const GROUPS: ColumnGroup[] = ['run', 'settings', 'metrics'];

/** Which columns the runs table shows (kept on the account). */
export function ColumnPicker({
  available,
  shown,
  onChange,
  onReset,
}: {
  available: Column[];
  shown: string[];
  onChange: (columns: string[]) => void;
  onReset: () => void;
}) {
  const { t } = useTranslation();
  const toggle = (id: string, on: boolean) => {
    if (on) {
      // Keep the order of the available columns.
      const order = available.map((c) => c.id);
      onChange(order.filter((c) => c === id || shown.includes(c)));
    } else {
      onChange(shown.filter((c) => c !== id));
    }
  };
  return (
    <Popover position="bottom-end" shadow="md" width={320} trapFocus>
      <Popover.Target>
        <Button variant="default" size="xs" leftSection={<IconColumns size={14} />}>
          {t('experiments.columns.choose')}
        </Button>
      </Popover.Target>
      <Popover.Dropdown>
        <ScrollArea.Autosize mah={420} type="auto">
          <Stack gap="sm">
            {GROUPS.map((group) => {
              const columns = available.filter((c) => c.group === group);
              if (columns.length === 0) {
                return null;
              }
              return (
                <fieldset key={group} style={{ border: 0, margin: 0, padding: 0 }}>
                  <Text component="legend" size="xs" fw={700} c="dimmed" mb={4}>
                    {t(`experiments.columns.${group}`)}
                  </Text>
                  <Stack gap={4}>
                    {columns.map((column) => (
                      <Checkbox
                        key={column.id}
                        size="xs"
                        label={column.label}
                        checked={shown.includes(column.id)}
                        onChange={(event) => toggle(column.id, event.currentTarget.checked)}
                      />
                    ))}
                  </Stack>
                </fieldset>
              );
            })}
          </Stack>
        </ScrollArea.Autosize>
        <Group justify="flex-end" mt="sm">
          <Button size="compact-xs" variant="subtle" onClick={onReset}>
            {t('experiments.columns.reset')}
          </Button>
        </Group>
      </Popover.Dropdown>
    </Popover>
  );
}
