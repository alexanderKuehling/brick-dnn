import type { ReactNode } from 'react';
import { Text } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { formatPercentTick, linearScale } from '@/components/chartScale';
import { valueAxis } from './axes';
import { ChartEmpty, ChartFrame, ChartTooltip, TooltipRow, type LegendItem } from './ChartFrame';
import { OTHER_COLOR, seriesColor } from './colors';
import { formatCount, formatPercent } from './format';
import { barPath, textWidth, truncate } from './marks';
import { useChartCursor } from './useChartCursor';
import classes from './Chart.module.css';

export interface BarRow {
  key: string;
  label: string;
  /** One value per series. */
  values: number[];
  /** Drawn in the de-emphasis gray (e.g. "Other", "Missing"). */
  muted?: boolean;
}

export interface BarSeries {
  key: string;
  label: string;
  color: string;
}

export interface BarChartProps {
  title: string;
  subtitle?: ReactNode;
  caption?: ReactNode;
  controls?: ReactNode;
  rows: BarRow[];
  /** Several series: grouped bars or stacked segments; one series by default. */
  series?: BarSeries[];
  layout?: 'grouped' | 'stacked';
  /** Stacked: show the share of each series within its row (100 % bars). */
  normalize?: boolean;
  /** Upper end of the value axis (e.g. 1 for coefficients); the largest value by default. */
  max?: number;
  formatValue?: (value: number) => string;
  formatTick?: (value: number, step: number) => string;
  /** Text at the end of each bar; the formatted total by default, '' for none. */
  tipLabel?: (row: BarRow) => string;
  /** Click / Enter on a row. */
  onSelect?: (key: string) => void;
  /** What selecting does, for the keyboard hint (e.g. "show the column"). */
  selectHint?: string;
  /** Unit in the text alternative, e.g. "rows". */
  unit?: string;
  width?: number;
}

const MARGIN = { top: 4, right: 8, bottom: 22 };
const ONE_SERIES: BarSeries[] = [{ key: 'value', label: 'Value', color: seriesColor(0) }];

/**
 * Horizontal bars for categories (long labels fit, many rows stay readable): one series, or
 * several grouped or stacked (e.g. classes per category). Values at the bar ends, a row
 * tooltip by pointer and keyboard, and a table view.
 */
export function BarChart({
  title,
  subtitle,
  caption,
  controls,
  rows,
  series = ONE_SERIES,
  layout = 'stacked',
  normalize = false,
  max: fixedMax,
  formatValue = formatCount,
  formatTick: formatAxisTick,
  tipLabel,
  onSelect,
  selectHint,
  unit = '',
  width: fixedWidth,
}: BarChartProps) {
  const { ref, width: measured } = useElementSize();
  const width = fixedWidth ?? measured;
  const cursor = useChartCursor(rows.length, {
    orientation: 'vertical',
    onActivate: onSelect ? (i) => onSelect(rows[i].key) : undefined,
  });
  const stacked = series.length > 1 && layout === 'stacked';
  const grouped = series.length > 1 && !stacked;

  const shown = (row: BarRow) => {
    const total = row.values.reduce((sum, v) => sum + (v || 0), 0);
    return normalize && stacked ? row.values.map((v) => (total > 0 ? v / total : 0)) : row.values;
  };
  const total = (row: BarRow) => row.values.reduce((sum, v) => sum + (v || 0), 0);
  const defaultTip = (row: BarRow) =>
    grouped ? '' : normalize && stacked ? '' : formatValue(stacked ? total(row) : row.values[0]);
  const tip = tipLabel ?? defaultTip;

  const geometry = (() => {
    const barHeight = grouped ? Math.max(6, Math.min(12, Math.floor(20 / series.length))) : 16;
    const rowHeight = grouped ? series.length * (barHeight + 2) + 8 : barHeight + 10;
    const plotHeight = rows.length * rowHeight;
    const height = MARGIN.top + plotHeight + MARGIN.bottom;
    const labelChars = Math.max(4, Math.floor(Math.min(180, width * 0.38) / 7));
    const labels = rows.map((row) => truncate(row.label, labelChars));
    const labelWidth = Math.min(
      Math.max(...labels.map((l) => textWidth(l, 12)), 20) + 10,
      Math.max(60, width * 0.4)
    );
    const tips = rows.map((row) => tip(row));
    const tipWidth = Math.max(0, ...tips.map((t) => (t ? textWidth(t) + 8 : 0)));
    const dataMax =
      fixedMax ??
      Math.max(
        0,
        ...rows.map((row) =>
          normalize && stacked ? 1 : stacked ? total(row) : Math.max(0, ...row.values)
        )
      );
    const axis = valueAxis(0, dataMax || 1, Math.max(2, Math.floor((width - labelWidth) / 80)), [
      0,
      fixedMax ?? Infinity,
    ]);
    const plotRight = Math.max(labelWidth + 20, width - MARGIN.right - tipWidth);
    const sx = linearScale([0, axis.domain[1]], [labelWidth, plotRight]);
    const format =
      formatAxisTick ?? (normalize && stacked ? formatPercentTick : (v: number) => axis.format(v));
    return { barHeight, rowHeight, height, labels, labelWidth, tips, axis, sx, format };
  })();

  const { barHeight, rowHeight, height, labels, labelWidth, tips, axis, sx, format } = geometry;
  const rowTop = (i: number) => MARGIN.top + i * rowHeight;
  const active = cursor.active;

  const colorOf = (row: BarRow, s: number) => (row.muted ? OTHER_COLOR : series[s].color);
  const valueText = (row: BarRow, s: number) => {
    const value = row.values[s] ?? 0;
    const sum = total(row);
    return stacked
      ? `${formatValue(value)} (${formatPercent(sum ? value / sum : 0)})`
      : formatValue(value);
  };
  const rowText = (i: number) => {
    const row = rows[i];
    if (!row) {
      return '';
    }
    if (series.length === 1) {
      return `${row.label}: ${tip(row) || formatValue(row.values[0])}`;
    }
    return `${row.label}: ${series.map((s, k) => `${s.label} ${valueText(row, k)}`).join(', ')}`;
  };
  const describe = () => {
    const top = rows
      .slice(0, 5)
      .map((_, i) => rowText(i))
      .join('; ');
    return `${title}. ${rows.length} ${rows.length === 1 ? 'bar' : 'bars'}${unit ? ` (${unit})` : ''}. ${top}${rows.length > 5 ? '; …' : ''}.`;
  };

  const pick = (clientY: number, target: HTMLElement) => {
    const py = clientY - target.getBoundingClientRect().top - MARGIN.top;
    const index = Math.floor(py / rowHeight);
    cursor.setActive(index >= 0 && index < rows.length ? index : null);
  };

  const legend: LegendItem[] =
    series.length > 1 ? series.map((s) => ({ key: s.key, label: s.label, color: s.color })) : [];

  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      caption={caption}
      controls={controls}
      legend={legend}
      table={() => ({
        head: [
          'Category',
          ...(series.length > 1 ? series.map((s) => s.label) : [unit ? capitalize(unit) : 'Value']),
          ...(stacked ? ['Total'] : []),
        ],
        rows: rows.map((row) => [
          row.label,
          ...series.map((_, k) => (stacked ? valueText(row, k) : formatValue(row.values[k] ?? 0))),
          ...(stacked ? [formatValue(total(row))] : []),
        ]),
      })}
    >
      {rows.length === 0 ? (
        <ChartEmpty />
      ) : (
        <div ref={ref} className={classes.plot} style={{ height }}>
          {width > 0 && (
            <svg
              className={classes.svg}
              width={width}
              height={height}
              role="img"
              aria-label={describe()}
            >
              {active !== null && (
                <rect
                  className={classes.band}
                  x={0}
                  y={rowTop(active)}
                  width={width}
                  height={rowHeight}
                  rx={4}
                />
              )}
              {axis.ticks.map((tick) => (
                <g key={tick}>
                  <line
                    className={classes.grid}
                    x1={sx(tick)}
                    x2={sx(tick)}
                    y1={MARGIN.top}
                    y2={height - MARGIN.bottom}
                  />
                  <text
                    className={classes.tick}
                    x={sx(tick)}
                    y={height - MARGIN.bottom + 15}
                    textAnchor="middle"
                  >
                    {format(tick, axis.step)}
                  </text>
                </g>
              ))}
              {rows.map((row, i) => {
                const top = rowTop(i);
                const values = shown(row);
                const barTop =
                  top +
                  (rowHeight - (grouped ? series.length * (barHeight + 2) - 2 : barHeight)) / 2;
                let end = sx(0);
                if (grouped) {
                  end = Math.max(...values.map((v) => sx(v || 0)));
                } else if (stacked) {
                  end = sx(values.reduce((sum, v) => sum + (v || 0), 0));
                } else {
                  end = sx(values[0] || 0);
                }
                // The last non-zero segment carries the rounded end.
                const lastSegment = values.reduce((last, v, k) => (v > 0 ? k : last), -1);
                let cumulative = 0;
                return (
                  <g key={row.key}>
                    <title>{rowText(i)}</title>
                    <text
                      className={classes.label}
                      x={labelWidth - 8}
                      y={top + rowHeight / 2}
                      dy="0.32em"
                      textAnchor="end"
                    >
                      {labels[i]}
                    </text>
                    {values.map((value, k) => {
                      if (!(value > 0)) {
                        return null;
                      }
                      if (grouped) {
                        return (
                          <path
                            key={k}
                            d={barPath(sx(0), sx(value), barTop + k * (barHeight + 2), barHeight)}
                            style={{ fill: colorOf(row, k) }}
                          />
                        );
                      }
                      const start = sx(cumulative) + (cumulative > 0 ? 1 : 0);
                      cumulative += value;
                      const stop = sx(cumulative) - (k < lastSegment ? 1 : 0);
                      return (
                        <path
                          key={k}
                          d={barPath(
                            start,
                            Math.max(start + 0.5, stop),
                            barTop,
                            barHeight,
                            k === lastSegment
                          )}
                          style={{ fill: colorOf(row, k) }}
                        />
                      );
                    })}
                    {tips[i] && (
                      <text
                        className={classes.value}
                        x={end + 6}
                        y={top + rowHeight / 2}
                        dy="0.32em"
                      >
                        {tips[i]}
                      </text>
                    )}
                  </g>
                );
              })}
              <line
                className={classes.axis}
                x1={sx(0)}
                x2={sx(0)}
                y1={MARGIN.top}
                y2={height - MARGIN.bottom}
              />
            </svg>
          )}

          <div
            className={classes.overlay}
            data-clickable={onSelect ? true : undefined}
            role="slider"
            tabIndex={0}
            aria-orientation="vertical"
            aria-label={`${title}: use the arrow keys to read the bars${onSelect && selectHint ? `, Enter to ${selectHint}` : ''}`}
            aria-valuemin={1}
            aria-valuemax={rows.length}
            aria-valuenow={(active ?? 0) + 1}
            aria-valuetext={rowText(active ?? 0)}
            onPointerMove={(event) => pick(event.clientY, event.currentTarget)}
            onPointerLeave={() => cursor.setActive(null)}
            onClick={(event) => {
              const py =
                event.clientY - event.currentTarget.getBoundingClientRect().top - MARGIN.top;
              const index = Math.floor(py / rowHeight);
              if (onSelect && rows[index]) {
                onSelect(rows[index].key);
              }
            }}
            onKeyDown={cursor.onKeyDown}
            onFocus={cursor.onFocus}
            onBlur={cursor.onBlur}
          />

          {active !== null && width > 0 && (
            <ChartTooltip
              x={Math.min(width - 10, sx(stacked ? axis.domain[1] / 2 : 0) + 40)}
              y={rowTop(active) + rowHeight}
              width={width}
            >
              <Text size="xs" c="dimmed" mb={2}>
                {rows[active].label}
              </Text>
              {series.map((s, k) => (
                <TooltipRow
                  key={s.key}
                  color={series.length > 1 ? colorOf(rows[active], k) : undefined}
                  value={
                    series.length > 1
                      ? valueText(rows[active], k)
                      : tips[active] || formatValue(rows[active].values[0])
                  }
                  label={series.length > 1 ? s.label : undefined}
                />
              ))}
            </ChartTooltip>
          )}
        </div>
      )}
    </ChartFrame>
  );
}

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}
