import type { ReactNode } from 'react';
import { Group, Paper, Text } from '@mantine/core';
import { LegendKey } from './ChartFrame';
import { formatCount, formatPercent } from './format';
import classes from './Chart.module.css';

export interface StatTileProps {
  /** Sentence case, no colon. */
  label: string;
  /** The figure, already formatted (e.g. "1,284", "12.9K"). */
  value: ReactNode;
  /** A short line under the value (e.g. "4.2% of all cells"). */
  hint?: ReactNode;
  /** Extra content, e.g. a SegmentedBar. */
  children?: ReactNode;
}

/** A headline number: label, value and an optional hint. */
export function StatTile({ label, value, hint, children }: StatTileProps) {
  return (
    <Paper withBorder p="xs" radius="sm" className={classes.tile}>
      <Text size="xs" c="dimmed">
        {label}
      </Text>
      <Text fw={600} size="lg" lh={1.3} className={classes.tileValue}>
        {value}
      </Text>
      {hint && (
        <Text size="xs" c="dimmed">
          {hint}
        </Text>
      )}
      {children}
    </Paper>
  );
}

export interface Segment {
  key: string;
  label: string;
  value: number;
  color: string;
}

/**
 * One whole split into parts: a thin bar (8px, rounded ends, 2px gaps) with a legend of
 * names, amounts and shares beside it. Parts at zero draw nothing.
 */
export function SegmentedBar({
  segments,
  label,
  showLegend = true,
}: {
  segments: Segment[];
  /** What the bar shows, for screen readers. */
  label: string;
  showLegend?: boolean;
}) {
  const total = segments.reduce((sum, s) => sum + Math.max(0, s.value), 0);
  const parts = segments.filter((s) => s.value > 0);
  const text = segments
    .map(
      (s) => `${s.label} ${formatCount(s.value)} (${formatPercent(total ? s.value / total : 0)})`
    )
    .join(', ');
  return (
    <div>
      <div
        className={classes.segments}
        style={{ height: 8, marginTop: 6 }}
        role="img"
        aria-label={`${label}: ${text}`}
      >
        {parts.map((s) => (
          <div
            key={s.key}
            className={classes.segment}
            style={{ flexGrow: s.value, flexBasis: 0, background: s.color }}
            title={`${s.label}: ${formatCount(s.value)} (${formatPercent(total ? s.value / total : 0)})`}
          />
        ))}
      </div>
      {showLegend && (
        <Group
          component="ul"
          gap={4}
          m={0}
          mt={6}
          p={0}
          style={{ listStyle: 'none', rowGap: 2 }}
          aria-hidden="true"
        >
          {segments.map((s) => (
            <Group key={s.key} component="li" gap={4} wrap="nowrap" mr={8}>
              <LegendKey color={s.color} />
              <Text size="xs" c="dimmed">
                {s.label}{' '}
                <Text span size="xs" c="var(--mantine-color-text)" fw={500}>
                  {formatCount(s.value)}
                </Text>
              </Text>
            </Group>
          ))}
        </Group>
      )}
    </div>
  );
}
