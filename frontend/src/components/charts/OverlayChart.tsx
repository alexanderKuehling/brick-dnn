import { useId, useMemo, useState, type ReactNode } from 'react';
import { Text } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { extent, linearScale, linePath } from '@/components/chartScale';
import { i18n, useTranslation } from '@/i18n';
import { valueAxis } from './axes';
import {
  ChartEmpty,
  ChartFrame,
  ChartTooltip,
  TooltipRow,
  type ChartTable,
  type LegendItem,
} from './ChartFrame';
import { MAX_SERIES, seriesColor } from './colors';
import { formatNumber } from './format';
import { textWidth } from './marks';
import classes from './Chart.module.css';

/** One line of an overlay chart: its own x positions (e.g. the epochs of one run). */
export interface OverlaySeries {
  key: string;
  label: string;
  x: number[];
  /** One value per x; null leaves a gap. */
  y: (number | null)[];
}

export interface OverlayChartProps {
  title: string;
  subtitle?: ReactNode;
  series: OverlaySeries[];
  xLabel: string;
  yLabel: string;
  formatX?: (value: number) => string;
  formatY?: (value: number) => string;
  /** Clamp the y axis, e.g. [0, 1] for accuracies. */
  yBounds?: [number, number];
  controls?: ReactNode;
  caption?: ReactNode;
  table?: () => ChartTable;
  height?: number;
  width?: number;
}

const MARGIN = { top: 22, right: 14, bottom: 38 };
/** Dash patterns after the eight colours are used up: series 9 and 10 are dashed. */
export const OVERLAY_DASH = '6 4';

/** The colour of series `index`, and whether it is dashed (from the ninth on). */
export function overlayStyle(index: number): { color: string; dashed: boolean } {
  return { color: seriesColor(index % MAX_SERIES), dashed: index >= MAX_SERIES };
}

const defaultFormat = (value: number) => formatNumber(value);

/** The index of the point of `xs` (sorted) nearest to `x`, or -1 for none. */
function nearest(xs: number[], x: number): number {
  let best = -1;
  let distance = Infinity;
  for (let i = 0; i < xs.length; i += 1) {
    const d = Math.abs(xs[i] - x);
    if (d < distance) {
      distance = d;
      best = i;
    }
  }
  return best;
}

/**
 * Several lines that each have their own x values (runs with different numbers of epochs, or
 * on a time axis), in one SVG chart: a colour per line (dashed after the eighth), a legend, a
 * crosshair with the value of every line at the nearest x (pointer and arrow keys), a text
 * summary for screen readers and the data as a table.
 */
export function OverlayChart({
  title,
  subtitle,
  series,
  xLabel,
  yLabel,
  formatX = defaultFormat,
  formatY = defaultFormat,
  yBounds,
  controls,
  caption,
  table,
  height = 280,
  width: fixedWidth,
}: OverlayChartProps) {
  useTranslation();
  const { ref, width: measured } = useElementSize();
  const width = fixedWidth ?? measured;
  const clipId = useId();
  const drawn = series.filter((s) => s.y.some((v) => typeof v === 'number'));
  const legend: LegendItem[] = series.map((s, i) => ({
    key: s.key,
    label: s.label,
    color: overlayStyle(i).color,
    shape: 'line',
    dashed: overlayStyle(i).dashed,
  }));
  // Every x value of any line, sorted: the stops of the crosshair.
  const stops = useMemo(
    () => Array.from(new Set(drawn.flatMap((s) => s.x))).sort((a, b) => a - b),
    [drawn]
  );
  const [active, setActive] = useState<number | null>(null);
  const activeIndex = active !== null && active < stops.length ? active : null;

  const geometry = useMemo(() => {
    const xr = extent(drawn.flatMap((s) => s.x)) ?? [0, 1];
    const yr = extent(drawn.flatMap((s) => s.y)) ?? [0, 1];
    const plotHeight = height - MARGIN.top - MARGIN.bottom;
    const yAxis = valueAxis(yr[0], yr[1], Math.max(3, Math.floor(plotHeight / 40)), yBounds);
    const yLabels = yAxis.ticks.map((t) => formatY(t));
    const left = Math.max(32, Math.max(0, ...yLabels.map((l) => textWidth(l))) + 12);
    const plotRight = Math.max(left + 10, width - MARGIN.right);
    const xAxis = valueAxis(
      xr[0],
      xr[1] === xr[0] ? xr[0] + 1 : xr[1],
      Math.max(2, Math.floor((plotRight - left) / 72))
    );
    const sx = linearScale(xAxis.domain, [left, plotRight]);
    const sy = linearScale(yAxis.domain, [height - MARGIN.bottom, MARGIN.top]);
    return { xAxis, yAxis, yLabels, left, plotRight, sx, sy };
  }, [drawn, width, height, yBounds, formatY]);
  const { xAxis, yAxis, yLabels, left, plotRight, sx, sy } = geometry;
  const base = height - MARGIN.bottom;

  const valuesAt = (x: number) =>
    series.map((s, i) => {
      const index = nearest(s.x, x);
      const value = index >= 0 ? s.y[index] : null;
      // Only the point at this x, or the nearest one if the line has no point there.
      return { s, i, x: index >= 0 ? s.x[index] : null, value };
    });

  const describe = (s: OverlaySeries) => {
    const points = s.x
      .map((x, i) => [x, s.y[i]] as const)
      .filter((p): p is readonly [number, number] => typeof p[1] === 'number');
    if (points.length === 0) {
      return i18n.t('ui.components.OverlayChart.noValues', { label: s.label });
    }
    const first = points[0];
    const last = points[points.length - 1];
    return i18n.t('ui.components.OverlayChart.fromTo', {
      label: s.label,
      first: formatY(first[1]),
      firstX: formatX(first[0]),
      last: formatY(last[1]),
      lastX: formatX(last[0]),
      xLabel,
    });
  };
  const summary = `${title}. ${i18n.t('ui.components.OverlayChart.against', {
    yLabel,
    xLabel,
    count: series.length,
  })} ${series.map(describe).join(' ')}`;

  const pointText = (stop: number) =>
    `${xLabel} ${formatX(stop)}: ${valuesAt(stop)
      .map(({ s, value }) =>
        typeof value === 'number'
          ? `${s.label} ${formatY(value)}`
          : `${s.label} ${i18n.t('a11y.charts.noValue')}`
      )
      .join(', ')}`;

  const pick = (clientX: number, target: HTMLElement) => {
    if (stops.length === 0) {
      return;
    }
    const rect = target.getBoundingClientRect();
    const px = clientX - rect.left;
    const [d0, d1] = xAxis.domain;
    const value = d0 + ((px - left) / Math.max(1, plotRight - left)) * (d1 - d0);
    setActive(nearest(stops, value));
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (stops.length === 0) {
      return;
    }
    const last = stops.length - 1;
    const current = activeIndex ?? last;
    const moves: Record<string, number | null> = {
      ArrowLeft: Math.max(0, current - 1),
      ArrowDown: Math.max(0, current - 1),
      ArrowRight: Math.min(last, current + 1),
      ArrowUp: Math.min(last, current + 1),
      Home: 0,
      End: last,
      Escape: null,
    };
    if (event.key in moves) {
      event.preventDefault();
      setActive(moves[event.key]);
    }
  };

  const activeX = activeIndex !== null ? stops[activeIndex] : null;
  const activeValues = activeX !== null ? valuesAt(activeX) : [];

  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      legend={legend}
      controls={controls}
      caption={caption}
      table={table}
    >
      {drawn.length === 0 ? (
        <ChartEmpty>{i18n.t('ui.components.OverlayChart.noValuesYet')}</ChartEmpty>
      ) : (
        <div ref={ref} className={classes.plot} style={{ height }}>
          {width > 0 && (
            <svg
              className={classes.svg}
              width={width}
              height={height}
              role="img"
              aria-label={summary}
            >
              <defs>
                <clipPath id={clipId}>
                  <rect
                    x={left - 4}
                    y={MARGIN.top - 4}
                    width={Math.max(0, plotRight - left + 8)}
                    height={base - MARGIN.top + 8}
                  />
                </clipPath>
              </defs>
              {yAxis.ticks.map((tick, i) => (
                <g key={`y${tick}`}>
                  <line
                    className={classes.grid}
                    x1={left}
                    x2={plotRight}
                    y1={sy(tick)}
                    y2={sy(tick)}
                  />
                  <text
                    className={classes.tick}
                    x={left - 8}
                    y={sy(tick)}
                    dy="0.32em"
                    textAnchor="end"
                  >
                    {yLabels[i]}
                  </text>
                </g>
              ))}
              <line className={classes.axis} x1={left} x2={plotRight} y1={base} y2={base} />
              {xAxis.ticks.map((tick) => (
                <text
                  key={`x${tick}`}
                  className={classes.tick}
                  x={sx(tick)}
                  y={base + 15}
                  textAnchor="middle"
                >
                  {formatX(tick)}
                </text>
              ))}
              <text
                className={classes.axisTitle}
                x={(left + plotRight) / 2}
                y={height - 4}
                textAnchor="middle"
              >
                {xLabel}
              </text>
              <text className={classes.axisTitle} x={0} y={10} textAnchor="start">
                {`↑ ${yLabel}`}
              </text>
              {activeX !== null && (
                <line
                  className={classes.axis}
                  x1={sx(activeX)}
                  x2={sx(activeX)}
                  y1={MARGIN.top}
                  y2={base}
                  strokeDasharray="3 3"
                />
              )}
              <g clipPath={`url(#${clipId})`}>
                {series.map((s, i) => {
                  const style = overlayStyle(i);
                  const points = s.x.map((x, k) => {
                    const value = s.y[k];
                    return typeof value === 'number' && Number.isFinite(value)
                      ? ([sx(x), sy(value)] as [number, number])
                      : null;
                  });
                  const single = points.filter(Boolean).length === 1;
                  return (
                    <g key={s.key} data-testid={`overlay-${s.key}`}>
                      <path
                        d={linePath(points)}
                        fill="none"
                        stroke={style.color}
                        strokeWidth={2}
                        strokeLinejoin="round"
                        strokeLinecap="round"
                        strokeDasharray={style.dashed ? OVERLAY_DASH : undefined}
                      />
                      {single &&
                        points.map((p, k) =>
                          p ? <circle key={k} cx={p[0]} cy={p[1]} r={4} fill={style.color} /> : null
                        )}
                    </g>
                  );
                })}
                {activeValues.map(({ s, i, x, value }) =>
                  typeof value === 'number' && x !== null ? (
                    <circle
                      key={`active-${s.key}`}
                      cx={sx(x)}
                      cy={sy(value)}
                      r={5}
                      fill={overlayStyle(i).color}
                      className={classes.ring}
                    />
                  ) : null
                )}
              </g>
            </svg>
          )}
          <div
            className={classes.overlay}
            role="slider"
            tabIndex={0}
            aria-label={i18n.t('ui.components.OverlayChart.useTheArrowKeys', { title, xLabel })}
            aria-valuemin={stops[0]}
            aria-valuemax={stops[stops.length - 1]}
            aria-valuenow={stops[activeIndex ?? stops.length - 1]}
            aria-valuetext={pointText(stops[activeIndex ?? stops.length - 1])}
            onPointerMove={(event) => pick(event.clientX, event.currentTarget)}
            onPointerDown={(event) => pick(event.clientX, event.currentTarget)}
            onPointerLeave={() => setActive(null)}
            onKeyDown={onKeyDown}
            onFocus={() => setActive((current) => current ?? stops.length - 1)}
            onBlur={() => setActive(null)}
          />
          {activeX !== null && width > 0 && (
            <ChartTooltip x={sx(activeX)} y={MARGIN.top} width={width}>
              <Text size="xs" c="dimmed" mb={2}>
                {xLabel} {formatX(activeX)}
              </Text>
              {activeValues.map(({ s, i, value }) => (
                <TooltipRow
                  key={s.key}
                  color={overlayStyle(i).color}
                  value={typeof value === 'number' ? formatY(value) : '–'}
                  label={s.label}
                />
              ))}
            </ChartTooltip>
          )}
        </div>
      )}
    </ChartFrame>
  );
}
