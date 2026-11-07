import type { ReactNode } from 'react';
import { Text } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { ChartFrame, ChartTooltip, GradientLegend, TooltipRow } from './ChartFrame';
import { divergingColor, textOn, TRACK, useChartScheme } from './colors';
import { formatCoefficient } from './format';
import { textWidth, truncate } from './marks';
import { useChartCursor } from './useChartCursor';
import classes from './Chart.module.css';

export interface HeatmapProps {
  title: string;
  subtitle?: ReactNode;
  caption?: ReactNode;
  controls?: ReactNode;
  rows: string[];
  columns: string[];
  /** values[row][column] in [-1, 1]; null where undefined. */
  values: (number | null)[][];
  /** Name of the value in tooltips, e.g. "Pearson r". */
  measure: string;
  onCellClick?: (row: number, column: number) => void;
  /** What a click does, for the keyboard hint (e.g. "open the scatter plot"). */
  clickHint?: string;
  width?: number;
}

const MAX_CELL = 44;
const LABEL_CHARS = 16;
const SIN45 = Math.SQRT1_2;

/** Strongest off-diagonal pairs of a symmetric matrix, by absolute value. */
export function strongestPairs(
  labels: string[],
  values: (number | null)[][],
  limit = 10
): { a: number; b: number; value: number }[] {
  const pairs: { a: number; b: number; value: number }[] = [];
  for (let a = 0; a < labels.length; a += 1) {
    for (let b = a + 1; b < labels.length; b += 1) {
      const value = values[a]?.[b];
      if (typeof value === 'number' && Number.isFinite(value)) {
        pairs.push({ a, b, value });
      }
    }
  }
  return pairs.sort((p, q) => Math.abs(q.value) - Math.abs(p.value)).slice(0, limit);
}

/**
 * A matrix of coefficients on a diverging scale (−1 red, 0 gray, +1 blue), with values in the
 * cells when they fit. Cells are read by pointer and arrow keys and can be clicked.
 */
export function Heatmap({
  title,
  subtitle,
  caption,
  controls,
  rows,
  columns,
  values,
  measure,
  onCellClick,
  clickHint,
  width: fixedWidth,
}: HeatmapProps) {
  const { ref, width: measured } = useElementSize();
  const width = fixedWidth ?? measured;
  const scheme = useChartScheme();
  const n = rows.length;
  const m = columns.length;
  const cursor = useChartCursor(n * m, {
    orientation: 'grid',
    columns: m,
    onActivate: onCellClick ? (i) => onCellClick(Math.floor(i / m), i % m) : undefined,
  });

  const rowLabels = rows.map((r) => truncate(r, LABEL_CHARS));
  const columnLabels = columns.map((c) => truncate(c, LABEL_CHARS));
  const rowLabelWidth = Math.min(
    Math.max(0, ...rowLabels.map((l) => textWidth(l))) + 10,
    width * 0.35
  );
  const columnLabelWidth = Math.max(0, ...columnLabels.map((l) => textWidth(l)));
  const rightPad = Math.min(columnLabelWidth * SIN45, 90);
  const rawCell = Math.floor((width - rowLabelWidth - rightPad) / Math.max(1, m));
  const cell = Math.max(4, Math.min(MAX_CELL, rawCell));
  const showLabels = cell >= 10;
  const left = showLabels ? rowLabelWidth : 0;
  const top = showLabels ? columnLabelWidth * SIN45 + 10 : 4;
  const height = top + n * cell + 4;
  const gap = cell >= 14 ? 2 : cell >= 6 ? 1 : 0;
  const annotate = cell >= 30;

  const active = cursor.active;
  const activeRow = active !== null ? Math.floor(active / m) : null;
  const activeColumn = active !== null ? active % m : null;
  const valueAt = (r: number, c: number) => values[r]?.[c] ?? null;
  const cellText = (r: number, c: number) => {
    const value = valueAt(r, c);
    return `${rows[r]} × ${columns[c]}: ${measure} ${value === null ? 'not defined' : formatCoefficient(value)}`;
  };
  const top3 = strongestPairs(rows, values, 3)
    .map((p) => `${rows[p.a]} and ${rows[p.b]} (${formatCoefficient(p.value)})`)
    .join(', ');
  const summary = `${title}. ${n} × ${m} matrix of ${measure}.${top3 ? ` Strongest pairs: ${top3}.` : ''}`;

  const pickCell = (clientX: number, clientY: number, target: HTMLElement) => {
    const rect = target.getBoundingClientRect();
    const c = Math.floor((clientX - rect.left - left) / cell);
    const r = Math.floor((clientY - rect.top - top) / cell);
    return r >= 0 && r < n && c >= 0 && c < m ? [r, c] : null;
  };

  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      caption={caption}
      controls={controls}
      legendExtra={
        <GradientLegend
          label="Colour scale: −1 red, 0 gray, +1 blue"
          stops={[
            'var(--chart-diverging-negative)',
            'var(--chart-diverging-midpoint)',
            'var(--chart-diverging-positive)',
          ]}
          labels={['−1', '0', '+1']}
        />
      }
      table={() => ({
        head: ['', ...columns],
        rows: rows.map((row, r) => [
          row,
          ...columns.map((_, c) => formatCoefficient(valueAt(r, c))),
        ]),
      })}
    >
      <div ref={ref} className={classes.plot} style={{ height }}>
        {width > 0 && (
          <svg
            className={classes.svg}
            width={width}
            height={height}
            role="img"
            aria-label={summary}
          >
            {showLabels &&
              columnLabels.map((label, c) => (
                <text
                  key={`c${c}`}
                  className={classes.tick}
                  transform={`translate(${left + c * cell + cell / 2},${top - 6}) rotate(-45)`}
                  style={activeColumn === c ? { fill: 'var(--mantine-color-text)' } : undefined}
                >
                  <title>{columns[c]}</title>
                  {label}
                </text>
              ))}
            {showLabels &&
              rowLabels.map((label, r) => (
                <text
                  key={`r${r}`}
                  className={classes.tick}
                  x={left - 6}
                  y={top + r * cell + cell / 2}
                  dy="0.32em"
                  textAnchor="end"
                  style={activeRow === r ? { fill: 'var(--mantine-color-text)' } : undefined}
                >
                  <title>{rows[r]}</title>
                  {label}
                </text>
              ))}
            {rows.map((_, r) =>
              columns.map((__, c) => {
                const value = valueAt(r, c);
                const fill = value === null ? TRACK[scheme] : divergingColor(value, scheme);
                const x = left + c * cell + gap / 2;
                const y = top + r * cell + gap / 2;
                const size = Math.max(1, cell - gap);
                return (
                  <g key={`${r}-${c}`}>
                    <rect
                      x={x}
                      y={y}
                      width={size}
                      height={size}
                      rx={cell >= 14 ? 2 : 0}
                      fill={fill}
                    />
                    {annotate && (
                      <text
                        x={x + size / 2}
                        y={y + size / 2}
                        dy="0.32em"
                        textAnchor="middle"
                        fontSize={10.5}
                        fill={value === null ? 'var(--mantine-color-dimmed)' : textOn(fill)}
                        style={{ fontVariantNumeric: 'tabular-nums' }}
                      >
                        {value === null ? '–' : formatCoefficient(value)}
                      </text>
                    )}
                  </g>
                );
              })
            )}
            {activeRow !== null && activeColumn !== null && (
              <rect
                className={classes.focusCell}
                x={left + activeColumn * cell}
                y={top + activeRow * cell}
                width={cell}
                height={cell}
                rx={2}
              />
            )}
          </svg>
        )}
        <div
          className={classes.overlay}
          data-clickable={onCellClick ? true : undefined}
          role="slider"
          tabIndex={0}
          aria-label={`${title}: use the arrow keys to read the cells${onCellClick && clickHint ? `, Enter to ${clickHint}` : ''}`}
          aria-valuemin={1}
          aria-valuemax={n * m}
          aria-valuenow={(active ?? 0) + 1}
          aria-valuetext={n * m > 0 ? cellText(activeRow ?? 0, activeColumn ?? 0) : ''}
          onPointerMove={(event) => {
            const hit = pickCell(event.clientX, event.clientY, event.currentTarget);
            cursor.setActive(hit ? hit[0] * m + hit[1] : null);
          }}
          onPointerLeave={() => cursor.setActive(null)}
          onClick={(event) => {
            const hit = pickCell(event.clientX, event.clientY, event.currentTarget);
            if (hit && onCellClick) {
              onCellClick(hit[0], hit[1]);
            }
          }}
          onKeyDown={cursor.onKeyDown}
          onFocus={cursor.onFocus}
          onBlur={cursor.onBlur}
        />
        {activeRow !== null && activeColumn !== null && width > 0 && (
          <ChartTooltip
            x={left + activeColumn * cell + cell}
            y={top + activeRow * cell + cell}
            width={width}
          >
            <Text size="xs" c="dimmed" mb={2}>
              {rows[activeRow]} × {columns[activeColumn]}
            </Text>
            <TooltipRow
              value={(() => {
                const value = valueAt(activeRow, activeColumn);
                return value === null ? 'not defined' : formatCoefficient(value);
              })()}
              label={measure}
            />
          </ChartTooltip>
        )}
      </div>
    </ChartFrame>
  );
}
