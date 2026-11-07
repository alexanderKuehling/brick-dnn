/** Icon and colour of each step group (step cards and the "Add step" menu). */
import {
  IconArrowMerge,
  IconBinary,
  IconCalendarEvent,
  IconChartDots,
  IconColumns,
  IconDropletQuestion,
  IconFilterCheck,
  IconLayoutRows,
  IconMathFunction,
  IconQuestionMark,
  IconTarget,
  type Icon,
} from '@tabler/icons-react';
import { ThemeIcon } from '@mantine/core';
import type { StepGroup } from '@/api/types';

const GROUPS: Record<StepGroup, { icon: Icon; color: string }> = {
  combine: { icon: IconArrowMerge, color: 'teal' },
  columns: { icon: IconColumns, color: 'brick' },
  rows: { icon: IconLayoutRows, color: 'cyan' },
  missing: { icon: IconDropletQuestion, color: 'orange' },
  outliers: { icon: IconChartDots, color: 'red' },
  transform: { icon: IconMathFunction, color: 'violet' },
  encode: { icon: IconBinary, color: 'grape' },
  datetime: { icon: IconCalendarEvent, color: 'yellow' },
  target: { icon: IconTarget, color: 'green' },
  selection: { icon: IconFilterCheck, color: 'indigo' },
};

export function GroupIcon({ group, size = 24 }: { group: StepGroup | undefined; size?: number }) {
  const entry = group ? GROUPS[group] : undefined;
  const IconComponent = entry?.icon ?? IconQuestionMark;
  return (
    <ThemeIcon
      size={size}
      radius="sm"
      variant="light"
      color={entry?.color ?? 'red'}
      aria-hidden
      style={{ flexShrink: 0 }}
    >
      <IconComponent size={Math.round(size * 0.65)} stroke={1.8} />
    </ThemeIcon>
  );
}
