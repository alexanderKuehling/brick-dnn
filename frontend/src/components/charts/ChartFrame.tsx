import {
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';
import { IconChartBar, IconTable } from '@tabler/icons-react';
import { ActionIcon, Box, Group, ScrollArea, Table, Text, Tooltip } from '@mantine/core';
import classes from './Chart.module.css';

export interface LegendItem {
  key: string;
  label: string;
  color: string;
  /** Mirrors the mark: rect for bars and areas, line for lines, dot for points. */
  shape?: 'rect' | 'line' | 'dot';
}

/** The data of a chart as a table (the text twin of every chart). */
export interface ChartTable {
  head: string[];
  rows: (string | number)[][];
  /** Columns aligned right (numbers); defaults to all but the first. */
  numeric?: boolean[];
}

export function LegendKey({
  color,
  shape = 'rect',
}: {
  color: string;
  shape?: LegendItem['shape'];
}) {
  return (
    <svg width={14} height={10} aria-hidden="true" style={{ flexShrink: 0 }}>
      {shape === 'line' && (
        <line x1={1} x2={13} y1={5} y2={5} stroke={color} strokeWidth={2} strokeLinecap="round" />
      )}
      {shape === 'dot' && <circle cx={7} cy={5} r={4} style={{ fill: color }} />}
      {shape === 'rect' && (
        <rect x={1} y={0} width={12} height={10} rx={2} style={{ fill: color }} />
      )}
    </svg>
  );
}

export function Legend({ items }: { items: LegendItem[] }) {
  return (
    <ul className={classes.legend} aria-label="Legend">
      {items.map((item) => (
        <li key={item.key} className={classes.legendItem}>
          <LegendKey color={item.color} shape={item.shape} />
          <Text size="xs" c="dimmed" truncate maw={200}>
            {item.label}
          </Text>
        </li>
      ))}
    </ul>
  );
}

/** A colour ramp legend for sequential and diverging scales. */
export function GradientLegend({
  stops,
  labels,
  label,
}: {
  /** CSS colours from low to high. */
  stops: string[];
  labels: [string, string] | [string, string, string];
  label: string;
}) {
  return (
    <Group gap={6} wrap="nowrap" aria-label={label} role="group">
      <Text size="xs" c="dimmed">
        {labels[0]}
      </Text>
      <Box
        className={classes.gradient}
        style={{ background: `linear-gradient(in oklab to right, ${stops.join(', ')})` }}
      />
      <Text size="xs" c="dimmed">
        {labels[labels.length - 1]}
      </Text>
    </Group>
  );
}

export function ChartTableView({ table, title }: { table: ChartTable; title: string }) {
  const numeric = table.numeric ?? table.head.map((_, i) => i > 0);
  return (
    <ScrollArea.Autosize mah={360} type="auto">
      <Table
        className={classes.table}
        striped
        withTableBorder
        fz="xs"
        verticalSpacing={4}
        aria-label={`${title} (table)`}
        stickyHeader
      >
        <Table.Thead>
          <Table.Tr>
            {table.head.map((cell, i) => (
              <Table.Th key={i} ta={numeric[i] ? 'right' : 'left'}>
                {cell}
              </Table.Th>
            ))}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {table.rows.map((row, r) => (
            <Table.Tr key={r}>
              {row.map((cell, i) => (
                <Table.Td key={i} ta={numeric[i] ? 'right' : 'left'}>
                  {cell}
                </Table.Td>
              ))}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </ScrollArea.Autosize>
  );
}

export interface ChartFrameProps {
  title: string;
  /** One line under the title: what is shown and what it is based on. */
  subtitle?: ReactNode;
  legend?: LegendItem[];
  /** Extra legend content (e.g. a GradientLegend or the trend line). */
  legendExtra?: ReactNode;
  /** Chart options (toggles) next to the title. */
  controls?: ReactNode;
  /** The chart data as a table; computed only when the table is opened. */
  table?: () => ChartTable;
  /** A note under the chart. */
  caption?: ReactNode;
  children: ReactNode;
}

/**
 * Title, legend, options and a table view around a chart. A legend is shown for two or more
 * series; the table toggle shows the same data as text.
 */
export function ChartFrame({
  title,
  subtitle,
  legend,
  legendExtra,
  controls,
  table,
  caption,
  children,
}: ChartFrameProps) {
  const [showTable, setShowTable] = useState(false);
  const titleId = useId();
  const showLegend = (legend && legend.length > 1) || legendExtra;
  return (
    <Box component="figure" className={classes.frame} aria-labelledby={titleId}>
      <Group justify="space-between" align="flex-start" gap="xs" mb={4} wrap="nowrap">
        <Box miw={0}>
          <Text id={titleId} size="sm" fw={600}>
            {title}
          </Text>
          {subtitle && (
            <Text size="xs" c="dimmed">
              {subtitle}
            </Text>
          )}
        </Box>
        <Group gap={6} wrap="wrap" justify="flex-end">
          {controls}
          {table && (
            <Tooltip label={showTable ? 'Show the chart' : 'Show as table'} withinPortal>
              <ActionIcon
                variant="subtle"
                color="gray"
                size="sm"
                aria-label={showTable ? `Show ${title} as chart` : `Show ${title} as table`}
                aria-pressed={showTable}
                onClick={() => setShowTable((value) => !value)}
              >
                {showTable ? <IconChartBar size={16} /> : <IconTable size={16} />}
              </ActionIcon>
            </Tooltip>
          )}
        </Group>
      </Group>
      {showLegend && !showTable && (
        <Group gap="md" mb={6} wrap="wrap">
          {legend && legend.length > 1 && <Legend items={legend} />}
          {legendExtra}
        </Group>
      )}
      {showTable && table ? <ChartTableView table={table()} title={title} /> : children}
      {caption && (
        <Text component="figcaption" size="xs" c="dimmed" mt={4}>
          {caption}
        </Text>
      )}
    </Box>
  );
}

export function ChartEmpty({ children = 'No values to show' }: { children?: ReactNode }) {
  return (
    <div className={classes.empty}>
      <Text size="sm" c="dimmed" ta="center" px="sm">
        {children}
      </Text>
    </div>
  );
}

const TOOLTIP_GAP = 12;

/**
 * Where a tooltip box goes next to point (x, y): right of it, or left of it in the right half
 * of the plot; on the other side if it does not fit, and clamped to the plot if neither side
 * does. Vertically it stays inside the plot when the plot is tall enough (`plotHeight` > 0).
 */
export function tooltipPosition(
  x: number,
  y: number,
  box: { width: number; height: number },
  plotWidth: number,
  plotHeight: number
): { left: number; top: number } {
  const right = x + TOOLTIP_GAP;
  const left = x - TOOLTIP_GAP - box.width;
  const fitsRight = right + box.width <= plotWidth;
  const fitsLeft = left >= 0;
  const preferLeft = x > plotWidth / 2;
  let position = preferLeft ? left : right;
  if (preferLeft ? !fitsLeft && fitsRight : !fitsRight && fitsLeft) {
    position = preferLeft ? right : left;
  }
  position = Math.min(Math.max(0, position), Math.max(0, plotWidth - box.width));
  const top =
    plotHeight > 0 && box.height <= plotHeight
      ? Math.min(Math.max(0, y), plotHeight - box.height)
      : Math.max(0, y);
  return { left: position, top };
}

/**
 * Tooltip box positioned next to an x/y point of the plot (its positioned parent), flipped
 * and clamped to stay inside it.
 */
export function ChartTooltip({
  x,
  y,
  width,
  children,
}: {
  x: number;
  y: number;
  /** Width of the plot, to keep the tooltip inside. */
  width: number;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<{ width: number; height: number; plot: number } | null>(null);
  // Measured after every render (the content changes with the point); set only on change.
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }
    const next = {
      width: element.offsetWidth,
      height: element.offsetHeight,
      plot: element.parentElement?.clientHeight ?? 0,
    };
    setSize((current) =>
      current &&
      current.width === next.width &&
      current.height === next.height &&
      current.plot === next.plot
        ? current
        : next
    );
  });
  // Until measured: flip by the half of the plot the point is in.
  const style: CSSProperties =
    size && size.width > 0
      ? tooltipPosition(x, y, size, width, size.plot)
      : {
          left: x,
          top: y,
          transform:
            x > width / 2
              ? `translate(calc(-100% - ${TOOLTIP_GAP}px), 0)`
              : `translate(${TOOLTIP_GAP}px, 0)`,
        };
  return (
    <div ref={ref} className={classes.tooltip} style={style} aria-hidden="true">
      {children}
    </div>
  );
}

/** One tooltip line: value first (strong), then what it is (secondary), keyed by a line. */
export function TooltipRow({
  color,
  value,
  label,
}: {
  color?: string;
  value: ReactNode;
  label?: ReactNode;
}) {
  return (
    <Group gap={6} wrap="nowrap">
      {color && <LegendKey color={color} shape="line" />}
      <Text size="sm" fw={600} className={classes.tooltipValue}>
        {value}
      </Text>
      {label && (
        <Text size="xs" c="dimmed" truncate>
          {label}
        </Text>
      )}
    </Group>
  );
}
