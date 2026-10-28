import { useMemo, useState } from 'react';
import { Box, Group, Text } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import {
  extent,
  formatTick,
  integerTicks,
  linearScale,
  linePath,
  nearestIndex,
  niceScale,
} from './chartScale';
import classes from './LineChart.module.css';

/** Series colours in fixed order: a series keeps its colour in every chart. */
export const SERIES_COLORS = ['var(--chart-series-1)', 'var(--chart-series-2)'] as const;

export interface LineSeries {
  id: string;
  label: string;
  /** A CSS colour, normally one of SERIES_COLORS. */
  color: string;
  /** One value per entry of `x`; null leaves a gap. */
  values: (number | null | undefined)[];
}

export interface LineChartProps {
  title: string;
  /** Sorted x positions shared by all series (e.g. epochs). Whole numbers. */
  x: number[];
  series: LineSeries[];
  /** Range of the x axis; defaults to the first and last x. */
  xDomain?: [number, number];
  /** Clamp the y axis, e.g. [0, 1] for accuracies. */
  yBounds?: [number, number];
  /** Name of the x values, e.g. "Epoch". */
  xLabel?: string;
  formatValue?: (value: number) => string;
  formatYTick?: (value: number, step: number) => string;
  /** Height of the plot including the axis labels. */
  height?: number;
  /** Fixed width in pixels; measured from the container when omitted. */
  width?: number;
}

const MARGIN = { top: 8, right: 10, bottom: 24 };
/** Keeps the first and last dot clear of the axis labels. */
const INSET = 8;
const TICK_CHAR_WIDTH = 6.6;

const defaultFormat = (value: number) =>
  value.toLocaleString('en-US', { maximumSignificantDigits: 4 });

function LineKey({ color }: { color: string }) {
  return (
    <svg width={14} height={8} aria-hidden="true" style={{ flexShrink: 0 }}>
      <line x1={1} x2={13} y1={4} y2={4} stroke={color} strokeWidth={2} strokeLinecap="round" />
    </svg>
  );
}

/**
 * A small, dependency-free SVG line chart: nice axes, a legend for two or more series,
 * a crosshair with a tooltip (pointer and arrow keys) and a text summary for screen readers.
 */
export function LineChart({
  title,
  x,
  series,
  xDomain,
  yBounds,
  xLabel = 'Epoch',
  formatValue = defaultFormat,
  formatYTick = formatTick,
  height = 190,
  width: fixedWidth,
}: LineChartProps) {
  const { ref, width: measured } = useElementSize();
  const width = fixedWidth ?? measured;
  const [active, setActive] = useState<number | null>(null);
  const activeIndex = active !== null && active < x.length ? active : null;

  const geometry = useMemo(() => {
    const plotHeight = height - MARGIN.top - MARGIN.bottom;
    const yRange = extent(series.flatMap((s) => s.values));
    const y = niceScale(
      yRange?.[0] ?? 0,
      yRange?.[1] ?? 1,
      Math.max(3, Math.floor(plotHeight / 36)),
      yBounds
    );
    const yLabels = y.ticks.map((tick) => formatYTick(tick, y.step));
    const left = Math.max(28, Math.max(...yLabels.map((l) => l.length)) * TICK_CHAR_WIDTH + 10);
    const x0 = xDomain?.[0] ?? x[0] ?? 1;
    const x1 = Math.max(x0, xDomain?.[1] ?? x[x.length - 1] ?? x0);
    const plotLeft = left + INSET;
    const plotRight = Math.max(plotLeft + 1, width - MARGIN.right - INSET);
    const sx = linearScale([x0, x1], [plotLeft, plotRight]);
    const sy = linearScale([y.min, y.max], [height - MARGIN.bottom, MARGIN.top]);
    const xTicks = integerTicks(x0, x1, Math.max(2, Math.floor((plotRight - plotLeft) / 64)));
    // Dots on every point only while they do not crowd each other.
    let gap = Infinity;
    for (let i = 1; i < x.length; i += 1) {
      gap = Math.min(gap, x[i] - x[i - 1]);
    }
    const showAllDots = x1 === x0 || (gap * (plotRight - plotLeft)) / (x1 - x0) >= 16;
    return { y, yLabels, left, x0, x1, sx, sy, xTicks, showAllDots };
  }, [series, x, xDomain, yBounds, height, width, formatYTick]);

  const { y, yLabels, left, x0, x1, sx, sy, xTicks, showAllDots } = geometry;

  const describe = (s: LineSeries) => {
    const points = x
      .map((xv, i) => [xv, s.values[i]] as const)
      .filter((p): p is readonly [number, number] => typeof p[1] === 'number');
    if (points.length === 0) {
      return `${s.label}: no data yet.`;
    }
    const [first, last] = [points[0], points[points.length - 1]];
    if (points.length === 1) {
      return `${s.label}: ${formatValue(first[1])} at ${xLabel.toLowerCase()} ${first[0]}.`;
    }
    return (
      `${s.label}: ${formatValue(first[1])} at ${xLabel.toLowerCase()} ${first[0]}, ` +
      `${formatValue(last[1])} at ${xLabel.toLowerCase()} ${last[0]}.`
    );
  };
  const summary = `${title}. ${series.map(describe).join(' ')}`;

  const pointText = (index: number) =>
    `${xLabel} ${x[index]}: ${series
      .map((s) => {
        const value = s.values[index];
        return `${s.label} ${typeof value === 'number' ? formatValue(value) : 'no value'}`;
      })
      .join(', ')}`;

  const pick = (clientX: number, target: HTMLElement) => {
    if (x.length === 0) {
      return;
    }
    const rect = target.getBoundingClientRect();
    const px = clientX - rect.left;
    const [r0, r1] = [sx(x0), sx(x1)];
    const value = r1 === r0 ? x0 : x0 + ((px - r0) / (r1 - r0)) * (x1 - x0);
    setActive(nearestIndex(x, value));
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (x.length === 0) {
      return;
    }
    const last = x.length - 1;
    const current = activeIndex ?? last;
    const next: Record<string, number | null> = {
      ArrowLeft: Math.max(0, current - 1),
      ArrowDown: Math.max(0, current - 1),
      ArrowRight: Math.min(last, current + 1),
      ArrowUp: Math.min(last, current + 1),
      Home: 0,
      End: last,
      Escape: null,
    };
    if (event.key in next) {
      event.preventDefault();
      setActive(next[event.key]);
    }
  };

  const activeX = activeIndex !== null ? sx(x[activeIndex]) : 0;
  const tooltipLeft = activeX > width / 2;

  return (
    <Box className={classes.root}>
      <Group justify="space-between" gap="xs" mb={4} wrap="wrap">
        <Text size="sm" fw={600}>
          {title}
        </Text>
        {series.length > 1 && (
          <Group gap="md" component="ul" m={0} p={0} style={{ listStyle: 'none' }}>
            {series.map((s) => (
              <Group key={s.id} component="li" gap={6} wrap="nowrap">
                <LineKey color={s.color} />
                <Text size="xs" c="dimmed">
                  {s.label}
                </Text>
              </Group>
            ))}
          </Group>
        )}
      </Group>

      <div ref={ref} className={classes.plot} style={{ height }}>
        {width > 0 && (
          <svg
            className={classes.svg}
            width={width}
            height={height}
            role="img"
            aria-label={summary}
          >
            {y.ticks.map((tick, i) => (
              <g key={tick}>
                <line className={classes.grid} x1={left} x2={width} y1={sy(tick)} y2={sy(tick)} />
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
            <line
              className={classes.axis}
              x1={left}
              x2={width}
              y1={height - MARGIN.bottom}
              y2={height - MARGIN.bottom}
            />
            {xTicks.map((tick) => (
              <text
                key={tick}
                className={classes.tick}
                x={sx(tick)}
                y={height - MARGIN.bottom + 16}
                textAnchor="middle"
              >
                {tick}
              </text>
            ))}

            {activeIndex !== null && (
              <line
                className={classes.crosshair}
                x1={activeX}
                x2={activeX}
                y1={MARGIN.top}
                y2={height - MARGIN.bottom}
              />
            )}

            {series.map((s) => {
              const points = x.map((xv, i) => {
                const value = s.values[i];
                return typeof value === 'number' && Number.isFinite(value)
                  ? ([sx(xv), sy(value)] as [number, number])
                  : null;
              });
              let lastIndex = points.length - 1;
              while (lastIndex >= 0 && !points[lastIndex]) {
                lastIndex -= 1;
              }
              const dots = points
                .map((p, i) => ({ p, i }))
                .filter(
                  ({ p, i }) =>
                    p !== null &&
                    (showAllDots ||
                      i === lastIndex ||
                      i === activeIndex ||
                      // Isolated points have no line, so they need a dot.
                      (!points[i - 1] && !points[i + 1]))
                );
              return (
                <g key={s.id}>
                  <path className={classes.line} d={linePath(points)} stroke={s.color} />
                  {dots.map(({ p, i }) => (
                    <circle
                      key={i}
                      className={classes.dot}
                      cx={p![0]}
                      cy={p![1]}
                      r={i === activeIndex ? 5 : 4}
                      fill={s.color}
                    />
                  ))}
                </g>
              );
            })}
          </svg>
        )}

        {x.length === 0 && (
          <div className={classes.empty}>
            <Text size="sm" c="dimmed">
              No data yet
            </Text>
          </div>
        )}

        {x.length > 0 && (
          <div
            className={classes.overlay}
            role="slider"
            tabIndex={0}
            aria-label={`${title}: use the arrow keys to read the values per ${xLabel.toLowerCase()}`}
            aria-valuemin={x[0]}
            aria-valuemax={x[x.length - 1]}
            aria-valuenow={x[activeIndex ?? x.length - 1]}
            aria-valuetext={pointText(activeIndex ?? x.length - 1)}
            onPointerMove={(event) => pick(event.clientX, event.currentTarget)}
            onPointerDown={(event) => pick(event.clientX, event.currentTarget)}
            onPointerLeave={() => setActive(null)}
            onKeyDown={onKeyDown}
            onFocus={() => setActive((current) => current ?? x.length - 1)}
            onBlur={() => setActive(null)}
          />
        )}

        {activeIndex !== null && width > 0 && (
          <div
            className={classes.tooltip}
            style={{
              left: activeX,
              top: MARGIN.top,
              transform: tooltipLeft ? 'translateX(calc(-100% - 12px))' : 'translateX(12px)',
            }}
            aria-hidden="true"
          >
            <Text size="xs" c="dimmed" mb={2}>
              {xLabel} {x[activeIndex]}
            </Text>
            {series.map((s) => {
              const value = s.values[activeIndex];
              return (
                <Group key={s.id} gap={6} wrap="nowrap">
                  <LineKey color={s.color} />
                  <Text size="sm" fw={600}>
                    {typeof value === 'number' ? formatValue(value) : '–'}
                  </Text>
                  <Text size="xs" c="dimmed">
                    {s.label}
                  </Text>
                </Group>
              );
            })}
          </div>
        )}
      </div>
    </Box>
  );
}
