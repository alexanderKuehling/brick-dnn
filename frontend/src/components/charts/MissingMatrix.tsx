import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Text } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { ChartFrame, ChartTooltip, TooltipRow, type LegendItem } from './ChartFrame';
import { mixColors, seriesHex, TRACK, useChartScheme } from './colors';
import { formatCount, formatPercent } from './format';
import { textWidth, truncate } from './marks';
import { useChartCursor } from './useChartCursor';
import classes from './Chart.module.css';

export interface MissingMatrixProps {
  title: string;
  subtitle?: ReactNode;
  caption?: ReactNode;
  controls?: ReactNode;
  /** One entry per column; null / undefined values are missing. */
  columns: { name: string; values: readonly unknown[] }[];
  /** Height of the matrix (rows are grouped into pixel rows when there are more). */
  height?: number;
  width?: number;
}

const LABEL_CHARS = 14;
const SIN45 = Math.SQRT1_2;
const LEFT = 44;
/** Widest strip per column (few columns would otherwise make large blocks). */
const MAX_COLUMN = 36;
/** Narrowest strip; with more columns than fit at this width the matrix scrolls sideways. */
const MIN_COLUMN = 1;

/**
 * Where values are missing: one strip per column, rows of the sample from top to bottom.
 * With more rows than pixels each pixel row shows the share missing in its rows. Drawn on a
 * canvas; columns are read by pointer and arrow keys. Many columns get thin strips; when even
 * those do not fit, the matrix scrolls sideways in its own box (never wider than the page).
 */
export function MissingMatrix({
  title,
  subtitle,
  caption,
  controls,
  columns,
  height: plotHeight = 200,
  width: fixedWidth,
}: MissingMatrixProps) {
  const { ref, width: measured } = useElementSize();
  const width = fixedWidth ?? measured;
  const scheme = useChartScheme();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const m = columns.length;
  const rows = Math.max(0, ...columns.map((c) => c.values.length));
  const cursor = useChartCursor(m);

  const buckets = Math.max(1, Math.min(rows, Math.floor(plotHeight / 2)));
  const stats = useMemo(
    () =>
      columns.map((column) => {
        const counts = new Uint32Array(buckets);
        const sizes = new Uint32Array(buckets);
        let missing = 0;
        for (let i = 0; i < rows; i += 1) {
          const b = Math.min(buckets - 1, Math.floor((i * buckets) / rows));
          sizes[b] += 1;
          const value = column.values[i];
          if (value === null || value === undefined) {
            counts[b] += 1;
            missing += 1;
          }
        }
        return { counts, sizes, missing };
      }),
    [columns, rows, buckets]
  );

  const labels = columns.map((c) => truncate(c.name, LABEL_CHARS));
  const labelWidth = Math.max(0, ...labels.map((l) => textWidth(l)));
  const rightPad = Math.min(labelWidth * SIN45, 80);
  const columnWidth =
    m > 0 ? Math.min(MAX_COLUMN, Math.max(MIN_COLUMN, (width - LEFT - rightPad) / m)) : 0;
  const scrolls = width > 0 && LEFT + m * columnWidth > width;
  /** Width of the drawing: the box, or more when it scrolls. */
  const plotWidth = scrolls ? Math.ceil(LEFT + m * columnWidth + 6) : width;
  const showLabels = columnWidth >= 11;
  const top = showLabels ? labelWidth * SIN45 + 10 : 6;
  const height = top + plotHeight + 4;
  const gap = columnWidth >= 6 ? 2 : columnWidth >= 3 ? 1 : 0;
  const bucketHeight = plotHeight / buckets;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width <= 0 || m === 0) {
      return;
    }
    const ratio = window.devicePixelRatio || 1;
    const w = Math.max(1, m * columnWidth);
    canvas.width = Math.round(w * ratio);
    canvas.height = Math.round(plotHeight * ratio);
    let ctx: CanvasRenderingContext2D | null = null;
    try {
      ctx = canvas.getContext('2d');
    } catch {
      ctx = null;
    }
    if (!ctx) {
      return;
    }
    const ramp = Array.from({ length: 33 }, (_, k) =>
      mixColors(TRACK[scheme], seriesHex(0, scheme), k / 32)
    );
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, plotHeight);
    stats.forEach(({ counts, sizes }, c) => {
      const x = c * columnWidth + gap / 2;
      for (let b = 0; b < buckets; b += 1) {
        const share = sizes[b] ? counts[b] / sizes[b] : 0;
        // Any missing value stays visible next to a full bucket.
        const step = share > 0 ? Math.max(6, Math.round(share * 32)) : 0;
        ctx.fillStyle = ramp[step];
        ctx.fillRect(x, b * bucketHeight, Math.max(1, columnWidth - gap), Math.ceil(bucketHeight));
      }
    });
  }, [stats, scheme, width, m, columnWidth, gap, buckets, bucketHeight, plotHeight]);

  const active = cursor.active;
  const columnText = (c: number) => {
    const s = stats[c];
    if (!s) {
      return '';
    }
    return `${columns[c].name}: ${formatCount(s.missing)} of ${formatCount(rows)} rows missing (${formatPercent(rows ? s.missing / rows : 0)})`;
  };
  const withMissing = stats
    .map((s, c) => ({ c, missing: s.missing }))
    .filter((s) => s.missing > 0)
    .sort((a, b) => b.missing - a.missing);
  const summary = `${title}. ${m} columns, ${formatCount(rows)} rows. ${
    withMissing.length === 0
      ? 'No missing values.'
      : `Missing values in ${withMissing.length} ${withMissing.length === 1 ? 'column' : 'columns'}: ${withMissing
          .slice(0, 6)
          .map(({ c }) => columnText(c))
          .join('; ')}.`
  }`;

  const legend: LegendItem[] = [
    { key: 'missing', label: 'Missing', color: 'var(--chart-series-1)' },
    { key: 'present', label: 'Present', color: 'var(--chart-track)' },
  ];

  const pick = (clientX: number, clientY: number, target: HTMLElement) => {
    const rect = target.getBoundingClientRect();
    const c = Math.floor((clientX - rect.left - LEFT) / columnWidth);
    const b = Math.floor((clientY - rect.top - top) / bucketHeight);
    cursor.setActive(c >= 0 && c < m ? c : null);
    return b >= 0 && b < buckets ? b : null;
  };
  const [hoverBucket, setHoverBucket] = useState<number | null>(null);
  const bucketRange = (b: number) => {
    const start = Math.ceil((b * rows) / buckets);
    const end = Math.ceil(((b + 1) * rows) / buckets);
    return [start + 1, Math.max(start + 1, end)];
  };

  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      caption={caption}
      controls={controls}
      legend={legend}
      table={() => ({
        head: ['Column', 'Missing', 'Share'],
        rows: stats.map((s, c) => [
          columns[c].name,
          formatCount(s.missing),
          formatPercent(rows ? s.missing / rows : 0),
        ]),
      })}
    >
      <div ref={ref} className={classes.scrollX} data-scroll={scrolls || undefined}>
        <div className={classes.plot} style={{ height, width: scrolls ? plotWidth : undefined }}>
          {width > 0 && (
            <>
              <svg
                className={classes.svg}
                width={plotWidth}
                height={height}
                role="img"
                aria-label={summary}
              >
                {showLabels &&
                  labels.map((label, c) => (
                    <text
                      key={c}
                      className={classes.tick}
                      transform={`translate(${LEFT + c * columnWidth + columnWidth / 2},${top - 6}) rotate(-45)`}
                      style={active === c ? { fill: 'var(--mantine-color-text)' } : undefined}
                    >
                      <title>{columns[c].name}</title>
                      {label}
                    </text>
                  ))}
                <text className={classes.tick} x={LEFT - 6} y={top + 8} textAnchor="end">
                  1
                </text>
                <text className={classes.tick} x={LEFT - 6} y={top + plotHeight} textAnchor="end">
                  {formatCount(rows)}
                </text>
                <text
                  className={classes.axisTitle}
                  transform={`translate(${12},${top + plotHeight / 2}) rotate(-90)`}
                  textAnchor="middle"
                >
                  Rows
                </text>
                {active !== null && (
                  <rect
                    className={classes.focusCell}
                    x={LEFT + active * columnWidth}
                    y={top - 1}
                    width={columnWidth}
                    height={plotHeight + 2}
                    rx={2}
                  />
                )}
              </svg>
              <canvas
                ref={canvasRef}
                className={classes.canvas}
                style={{ left: LEFT, top, width: Math.max(1, m * columnWidth), height: plotHeight }}
                aria-hidden="true"
              />
            </>
          )}
          <div
            className={classes.overlay}
            role="slider"
            tabIndex={0}
            aria-label={`${title}: use the arrow keys to read the columns`}
            aria-valuemin={1}
            aria-valuemax={Math.max(1, m)}
            aria-valuenow={(active ?? 0) + 1}
            aria-valuetext={columnText(active ?? 0)}
            onPointerMove={(event) => {
              setHoverBucket(pick(event.clientX, event.clientY, event.currentTarget));
            }}
            onPointerLeave={() => {
              setHoverBucket(null);
              cursor.setActive(null);
            }}
            onKeyDown={(event) => {
              setHoverBucket(null);
              cursor.onKeyDown(event);
            }}
            onFocus={cursor.onFocus}
            onBlur={cursor.onBlur}
          />
          {active !== null && width > 0 && stats[active] && (
            <ChartTooltip x={LEFT + (active + 1) * columnWidth} y={top} width={plotWidth}>
              <Text size="xs" c="dimmed" mb={2}>
                {columns[active].name}
              </Text>
              <TooltipRow
                value={`${formatCount(stats[active].missing)} (${formatPercent(rows ? stats[active].missing / rows : 0)})`}
                label="missing"
              />
              {hoverBucket !== null &&
                (() => {
                  const b = hoverBucket;
                  const [from, to] = bucketRange(b);
                  const s = stats[active];
                  return (
                    <TooltipRow
                      value={`${formatCount(s.counts[b])} of ${formatCount(s.sizes[b])}`}
                      label={
                        from === to
                          ? `row ${formatCount(from)}`
                          : `rows ${formatCount(from)}–${formatCount(to)}`
                      }
                    />
                  );
                })()}
            </ChartTooltip>
          )}
        </div>
      </div>
    </ChartFrame>
  );
}
