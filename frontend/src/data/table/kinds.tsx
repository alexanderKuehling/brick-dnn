/** Icons, colours and names of the column kinds (table headers, column pickers). */
import {
  IconAlignLeft,
  IconCalendar,
  IconHash,
  IconTag,
  IconToggleLeft,
  type Icon,
} from '@tabler/icons-react';
import type { ColumnKind } from '@/api/types';

export const KIND_LABELS: Record<ColumnKind, string> = {
  numeric: 'Numeric',
  boolean: 'Boolean',
  datetime: 'Date/time',
  categorical: 'Categorical',
  text: 'Text',
};

export const KIND_COLORS: Record<ColumnKind, string> = {
  numeric: 'blue',
  boolean: 'teal',
  datetime: 'orange',
  categorical: 'grape',
  text: 'gray',
};

const KIND_ICONS: Record<ColumnKind, Icon> = {
  numeric: IconHash,
  boolean: IconToggleLeft,
  datetime: IconCalendar,
  categorical: IconTag,
  text: IconAlignLeft,
};

export const KINDS: ColumnKind[] = ['numeric', 'categorical', 'boolean', 'datetime', 'text'];

interface KindIconProps {
  kind: ColumnKind;
  size?: number;
  /** Accessible name; without it the icon is decorative. */
  label?: string;
}

/** The icon of a column kind in its colour. */
export function KindIcon({ kind, size = 14, label }: KindIconProps) {
  const IconComponent = KIND_ICONS[kind] ?? IconHash;
  return (
    <IconComponent
      size={size}
      stroke={1.8}
      color={`var(--mantine-color-${KIND_COLORS[kind] ?? 'gray'}-light-color)`}
      style={{ flexShrink: 0 }}
      role={label ? 'img' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    />
  );
}
