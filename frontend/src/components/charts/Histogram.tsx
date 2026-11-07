import { useMemo, type ReactNode } from 'react';
import { Text } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { formatPercentTick, linearScale } from '@/components/chartScale';
import { innerTicks, logAxis, logScale, timeTicks, valueAxis } from './axes';
import { ChartEmpty, ChartFrame, ChartTooltip, TooltipRow, type LegendItem } from './ChartFrame';
import { formatCount, formatDate, formatNumber, formatPercent } from './format';
import { columnPath, textWidth } from './marks';
import { useChartCursor } from './useChartCursor';
import classes from './Chart.module.css';

export interface HistogramSeries {
  key: string;
  label: string;
  /** A CSS colour, normally `seriesColor(i)`. */
  color: string;
  /** One count per bin. */
  counts: number[];
}

export interface HistogramProps {
  title: string;
  subtitle?: ReactNode;
  caption?: ReactNode;
  controls?: ReactNode;
  /** Bin edges: bin i covers edges[i] … edges[i + 1]. */
  edges: number[];
  /** One series draws bars; several draw overlaid outlines (e.g. one per class). */
  series: HistogramSeries[];
  /** Edges are timestamps (ms since the epoch). */
  time?: boolean;
  /** Whole-number data: bins are labelled with the whole numbers they contain. */
  integer?: boolean;
  /** Logarithmic count axis (for long tails). */
  log?: boolean;
  /** Show each series as shares of its own total (compares shapes of different sizes). */
  normalize?: boolean;
  /** Counts are estimates (moved onto other bins). */
  approximate?: boolean;
  /** Name of the values, e.g. the column. */
  xLabel?: string;
  /** Unit of the counts in the text alternative and table, e.g. "rows". */
  unit?: string;
  formatX?: (value: number) => string;
  /** Height of the plot including the axis labels. */
  height?: number;
  width?: number;
}

const MARGIN = { top: 10, right: 12, bottom: 24 };

/** Bin labels: "18–23" for whole numbers, "2021-01-01 – 2022-01-01" for dates. */
export function binLabels(
  edges: number[],
  { time = false, integer = false, formatX }: Pick<HistogramProps, 'time' | 'integer' | 'formatX'>
): string[] {
  const labels: string[] = [];
  for (let i = 0; i + 1 < edges.length; i += 1) {
    const [a, b] = [edges[i], edges[i + 1]];
    const last = i + 2 === edges.length;
    if (time) {
      const unit = b - a >= 86_400_000 ? 'day' : 'minute';
      labels.push(`${formatDate(a, unit)} – ${formatDate(b, unit)}`);
    } else if (integer) {
      const lo = Math.ceil(a);
      const hi = last ? Math.floor(b) : Math.ceil(b) - 1;
      labels.push(hi <= lo ? formatCount(lo) : `${formatCount(lo)}–${formatCount(hi)}`);
    } else {
      const format = formatX ?? ((v: number) => formatNumber(v, 5));
      labels.push(`${format(a)} – ${format(b)}`);
    }
  }
  return labels;
}

/**
 * A histogram: bars for one series, overlaid outlines with a light wash for several (e.g. one
 * per class), optional log axis. Bins are read by pointer and arrow keys; the summary and
 * table view give the same numbers as text.
 */
export function Histogram({
  title,
  subtitle,
  caption,
  controls,
  edges,
  series,
  time = false,
  integer = false,
  log = false,
  normalize = false,
  approximate = false,
  xLabel,
  unit = 'values',
  formatX,
  height = 200,
  width: fixedWidth,
}: HistogramProps) {
  const { ref, width: measured } = useElementSize();
  const width = fixedWidth ?? measured;
  const bins = Math.max(0, edges.length - 1);
  const cursor = useChartCursor(bins);

  const data = useMemo(
    () =>
      series.map((s) => {
        const counts = Array.from({ length: bins }, (_, i) => s.counts[i] ?? 0);
        const total = counts.reduce((sum, c) => sum + c, 0);
        return {
          ...s,
          counts,
          total,
          values: counts.map((c) => (normalize ? (total > 0 ? c / total : 0) : c)),
        };
      }),
    [series, bins, normalize]
  );
  const labels = useMemo(
    () => binLabels(edges, { time, integer, formatX }),
    [edges, time, integer, formatX]
  );
  const hasData = bins > 0 && data.some((s) => s.total > 0);

  const geometry = useMemo(() => {
    const base = height - MARGIN.bottom;
    const plotHeight = base - MARGIN.top;
    const all = data.flatMap((s) => s.values);
    const max = Math.max(0, ...all);
    const minPositive = Math.min(...all.filter((v) => v > 0));
    const useLog = log && Number.isFinite(minPositive);
    const yAxis = useLog
      ? logAxis(minPositive, max, normalize)
      : (() => {
          const axis = valueAxis(0, max || 1, Math.max(3, Math.floor(plotHeight / 36)), [
            0,
            Infinity,
          ]);
          return normalize
            ? { ...axis, format: (v: number) => formatPercentTick(v, axis.step) }
            : axis;
        })();
    const yLabels = yAxis.ticks.map(yAxis.format);
    const left = Math.max(28, Math.max(...yLabels.map((l) => textWidth(l))) + 10);
    const plotRight = Math.max(left + 1, width - MARGIN.right);
    const lo = edges[0] ?? 0;
    const hi = edges[bins] ?? 1;
    const sx = linearScale([lo, hi], [left, plotRight]);
    const sy = (useLog ? logScale : linearScale)(yAxis.domain, [base, MARGIN.top]);
    const count = Math.max(2, Math.floor((plotRight - left) / (time ? 96 : 64)));
    const xAxis = time ? timeTicks(lo, hi, count) : innerTicks(lo, hi, count, integer);
    const y = (v: number) => (v > 0 || !useLog ? Math.min(base, sy(v)) : base);
    return { base, yAxis, yLabels, left, plotRight, lo, hi, sx, y, xAxis };
  }, [data, log, normalize, height, width, edges, bins, time, integer]);

  const { base, yAxis, yLabels, left, plotRight, lo, hi, sx, y, xAxis } = geometry;
  const formatX0 = formatX ?? xAxis.format;

  const formatValue = (value: number) => {
    if (normalize) {
      return formatPercent(value);
    }
    return approximate ? `≈ ${formatCount(value)}` : formatCount(value);
  };
  const share = (s: (typeof data)[number], i: number) =>
    s.total > 0 ? formatPercent(s.counts[i] / s.total) : '–';

  const describe = () => {
    if (!hasData) {
      return `${title}: no values.`;
    }
    const range = `${bins} bins from ${formatX0(lo)} to ${formatX0(hi)}`;
    const parts = data.map((s) => {
      let peak = 0;
      s.counts.forEach((c, i) => {
        if (c > s.counts[peak]) {
          peak = i;
        }
      });
      const name = data.length > 1 ? `${s.label}: ` : '';
      return (
        `${name}${formatCount(s.total)} ${unit}, most in ${labels[peak]} ` +
        `(${formatValue(s.values[peak])}${normalize ? '' : `, ${share(s, peak)}`}).`
      );
    });
    return `${title}. Histogram${xLabel ? ` of ${xLabel}` : ''}, ${range}. ${parts.join(' ')}`;
  };

  const binText = (i: number) =>
    `${labels[i]}: ${data
      .map(
        (s) =>
          `${data.length > 1 ? `${s.label} ` : ''}${formatValue(s.values[i])}${normalize ? '' : ` (${share(s, i)})`}`
      )
      .join(', ')}`;

  const pick = (clientX: number, target: HTMLElement) => {
    const px = clientX - target.getBoundingClientRect().left;
    const value = lo + ((px - left) / Math.max(1, plotRight - left)) * (hi - lo);
    let index = 0;
    while (index < bins - 1 && edges[index + 1] <= value) {
      index += 1;
    }
    cursor.setActive(index);
  };

  const legend: LegendItem[] = data.map((s) => ({
    key: s.key,
    label: s.label,
    color: s.color,
    shape: 'rect',
  }));

  const active = cursor.active;
  const binX = (i: number) => sx(edges[i]);
  const gapFor = (w: number) => (w >= 8 ? 2 : w >= 4 ? 1 : 0);

  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      caption={caption}
      controls={controls}
      legend={legend}
      table={() => ({
        head: [
          xLabel ?? 'Range',
          ...data.flatMap((s) =>
            normalize ? [`${s.label} (share)`] : [data.length > 1 ? s.label : 'Count', 'Share']
          ),
        ],
        rows: labels.map((label, i) => [
          label,
          ...data.flatMap((s) =>
            normalize ? [formatValue(s.values[i])] : [formatValue(s.values[i]), share(s, i)]
          ),
        ]),
      })}
    >
      {!hasData ? (
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
                  x={binX(active)}
                  y={MARGIN.top}
                  width={Math.max(1, binX(active + 1) - binX(active))}
                  height={base - MARGIN.top}
                />
              )}
              {yAxis.ticks.map((tick, i) => (
                <g key={tick}>
                  <line
                    className={classes.grid}
                    x1={left}
                    x2={plotRight}
                    y1={y(tick)}
                    y2={y(tick)}
                  />
                  <text
                    className={classes.tick}
                    x={left - 8}
                    y={y(tick)}
                    dy="0.32em"
                    textAnchor="end"
                  >
                    {yLabels[i]}
                  </text>
                </g>
              ))}

              {data.length === 1
                ? data[0].values.map((value, i) => {
                    const x0 = binX(i);
                    const w = binX(i + 1) - x0;
                    const gap = gapFor(w);
                    return value > 0 ? (
                      <path
                        key={i}
                        d={columnPath(
                          x0 + gap / 2,
                          Math.max(0.5, w - gap),
                          y(value),
                          base,
                          w > 6 ? 4 : 1
                        )}
                        style={{ fill: data[0].color }}
                        opacity={active === null || active === i ? 1 : 0.75}
                      />
                    ) : null;
                  })
                : data.map((s) => {
                    let d = `M${binX(0)},${base}`;
                    s.values.forEach((value, i) => {
                      d += `V${y(value).toFixed(1)}H${binX(i + 1).toFixed(1)}`;
                    });
                    d += `V${base}`;
                    return (
                      <g key={s.key}>
                        <path d={`${d}Z`} style={{ fill: s.color }} opacity={0.1} />
                        <path
                          d={d}
                          fill="none"
                          style={{ stroke: s.color }}
                          strokeWidth={2}
                          strokeLinejoin="round"
                        />
                      </g>
                    );
                  })}

              <line className={classes.axis} x1={left} x2={plotRight} y1={base} y2={base} />
              {xAxis.ticks.map((tick) => (
                <text
                  key={tick}
                  className={classes.tick}
                  x={sx(tick)}
                  y={base + 16}
                  textAnchor="middle"
                >
                  {xAxis.format(tick)}
                </text>
              ))}
            </svg>
          )}

          <div
            className={classes.overlay}
            role="slider"
            tabIndex={0}
            aria-label={`${title}: use the arrow keys to read the bins`}
            aria-valuemin={1}
            aria-valuemax={bins}
            aria-valuenow={(active ?? 0) + 1}
            aria-valuetext={binText(active ?? 0)}
            onPointerMove={(event) => pick(event.clientX, event.currentTarget)}
            onPointerDown={(event) => pick(event.clientX, event.currentTarget)}
            onPointerLeave={() => cursor.setActive(null)}
            onKeyDown={cursor.onKeyDown}
            onFocus={cursor.onFocus}
            onBlur={cursor.onBlur}
          />

          {active !== null && width > 0 && (
            <ChartTooltip x={(binX(active) + binX(active + 1)) / 2} y={MARGIN.top} width={width}>
              <Text size="xs" c="dimmed" mb={2}>
                {labels[active]}
              </Text>
              {data.map((s) => (
                <TooltipRow
                  key={s.key}
                  color={data.length > 1 ? s.color : undefined}
                  value={formatValue(s.values[active])}
                  label={
                    data.length > 1
                      ? `${s.label}${normalize ? '' : ` · ${share(s, active)}`}`
                      : normalize
                        ? undefined
                        : share(s, active)
                  }
                />
              ))}
            </ChartTooltip>
          )}
        </div>
      )}
    </ChartFrame>
  );
}
