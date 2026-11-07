import type { ReactNode } from 'react';
import {
  IconCalendar,
  IconCategory,
  IconHash,
  IconLetterCase,
  IconToggleRight,
  type Icon,
} from '@tabler/icons-react';
import { Group, Select, Text, type SelectProps } from '@mantine/core';
import type { ColumnKind } from '@/api/types';
import classes from './AnalysisPanel.module.css';

export const KIND_ORDER: ColumnKind[] = ['numeric', 'boolean', 'datetime', 'categorical', 'text'];

export const KIND_LABELS: Record<ColumnKind, string> = {
  numeric: 'Number',
  boolean: 'Yes/no',
  datetime: 'Date/time',
  categorical: 'Category',
  text: 'Text',
};

const KIND_ICONS: Record<ColumnKind, Icon> = {
  numeric: IconHash,
  boolean: IconToggleRight,
  datetime: IconCalendar,
  categorical: IconCategory,
  text: IconLetterCase,
};

export function KindIcon({ kind, size = 14 }: { kind: ColumnKind; size?: number }) {
  const Glyph = KIND_ICONS[kind];
  return (
    <Glyph
      size={size}
      stroke={1.75}
      role="img"
      aria-label={KIND_LABELS[kind]}
      className={classes.kindIcon}
    />
  );
}

export interface ColumnOption {
  name: string;
  kind: ColumnKind;
}

/** A searchable column picker with kind icons. */
export function ColumnSelect({
  columns,
  value,
  onChange,
  label,
  placeholder = 'Pick a column',
  clearable = false,
  extra,
  ...props
}: {
  columns: ColumnOption[];
  value: string | null;
  onChange: (value: string | null) => void;
  label: string;
  placeholder?: string;
  clearable?: boolean;
  /** Options before the columns, e.g. "None". */
  extra?: { value: string; label: string }[];
} & Pick<SelectProps, 'w' | 'miw' | 'maw' | 'size' | 'style' | 'description'>) {
  const kinds = new Map(columns.map((c) => [c.name, c.kind]));
  return (
    <Select
      label={label}
      placeholder={placeholder}
      data={[...(extra ?? []), ...columns.map((c) => ({ value: c.name, label: c.name }))]}
      value={value}
      onChange={(next) => onChange(next)}
      searchable
      clearable={clearable}
      allowDeselect={false}
      nothingFoundMessage="No column with this name"
      maxDropdownHeight={280}
      comboboxProps={{ withinPortal: true }}
      leftSection={value && kinds.has(value) ? <KindIcon kind={kinds.get(value)!} /> : undefined}
      renderOption={({ option }) => {
        const kind = kinds.get(option.value);
        return (
          <Group gap={6} wrap="nowrap">
            {kind ? <KindIcon kind={kind} /> : <span style={{ width: 18 }} />}
            <Text size="sm" truncate>
              {option.label}
            </Text>
          </Group>
        );
      }}
      size="xs"
      {...props}
    />
  );
}

/** Honest labelling under a section: what the numbers are based on. */
export function Note({ children }: { children: ReactNode }) {
  return (
    <Text size="xs" c="dimmed">
      {children}
    </Text>
  );
}
