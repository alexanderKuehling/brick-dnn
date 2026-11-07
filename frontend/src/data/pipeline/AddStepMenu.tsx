import { useMemo, useState } from 'react';
import { IconPlus, IconSearch } from '@tabler/icons-react';
import { Button, Menu, ScrollArea, Text, type ButtonProps } from '@mantine/core';
import type { StepCatalog, StepSpec } from '@/api/types';
import { GroupIcon } from './groups';
import classes from './PipelineEditor.module.css';

interface AddStepMenuProps {
  catalog: StepCatalog;
  onAdd: (spec: StepSpec) => void;
  /** Why combine steps cannot be added (e.g. no other table), or null. */
  combineDisabled: string | null;
  /** Why no step can be added at all (e.g. the step limit), or null. */
  disabled?: string | null;
  buttonProps?: ButtonProps;
  label?: string;
}

function matches(spec: StepSpec, groupLabel: string, query: string): boolean {
  if (!query) {
    return true;
  }
  const text = `${spec.label} ${spec.description} ${spec.type} ${groupLabel}`.toLowerCase();
  return query
    .toLowerCase()
    .split(/\s+/)
    .every((word) => text.includes(word));
}

/** "Add step" button with a searchable menu of all step types, grouped like the catalog. */
export function AddStepMenu({
  catalog,
  onAdd,
  combineDisabled,
  disabled = null,
  buttonProps,
  label = 'Add step',
}: AddStepMenuProps) {
  const [query, setQuery] = useState('');

  const groups = useMemo(() => {
    const q = query.trim();
    const known = new Set(catalog.groups.map((group) => group.id));
    const groupList = [
      ...catalog.groups,
      // Steps of groups the catalog does not list still show up.
      ...[...new Set(catalog.steps.map((s) => s.group))]
        .filter((id) => !known.has(id))
        .map((id) => ({ id, label: id })),
    ];
    return groupList
      .map((group) => ({
        group,
        steps: catalog.steps.filter((s) => s.group === group.id && matches(s, group.label, q)),
      }))
      .filter((entry) => entry.steps.length > 0);
  }, [catalog, query]);

  return (
    <Menu position="bottom-start" width={340} shadow="md" withinPortal>
      <Menu.Target>
        <Button
          leftSection={<IconPlus size={16} />}
          disabled={Boolean(disabled)}
          title={disabled ?? undefined}
          {...buttonProps}
        >
          {label}
        </Button>
      </Menu.Target>
      <Menu.Dropdown maw="calc(100vw - 32px)">
        <Menu.Search
          aria-label="Search steps"
          placeholder="Search steps"
          leftSection={<IconSearch size={14} />}
          value={query}
          onChange={(event) => setQuery(event.currentTarget.value)}
        />
        <ScrollArea.Autosize mah={420} type="auto" offsetScrollbars>
          {groups.map(({ group, steps }) => (
            <div key={group.id}>
              <Menu.Label>{group.label}</Menu.Label>
              {steps.map((spec) => {
                const reason = spec.combine ? combineDisabled : null;
                return (
                  <Menu.Item
                    key={spec.type}
                    leftSection={<GroupIcon group={spec.group} size={22} />}
                    disabled={Boolean(reason)}
                    onClick={() => onAdd(spec)}
                    className={classes.menuItem}
                  >
                    <Text size="sm" fw={500} component="span" display="block">
                      {spec.label}
                    </Text>
                    <Text size="xs" c="dimmed" component="span" lineClamp={2}>
                      {reason ?? spec.description}
                    </Text>
                  </Menu.Item>
                );
              })}
            </div>
          ))}
          {groups.length === 0 && (
            <Text size="sm" c="dimmed" px="sm" py="xs">
              No step matches “{query.trim()}”.
            </Text>
          )}
        </ScrollArea.Autosize>
      </Menu.Dropdown>
    </Menu>
  );
}
