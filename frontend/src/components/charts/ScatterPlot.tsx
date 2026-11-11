import { useEffect, useId, useMemo, useRef, type ReactNode } from 'react';
import { Text } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { extent, linearScale } from '@/components/chartScale';
import { timeTicks, valueAxis, type Axis } from './axes';
import {
  ChartEmpty,
  ChartFrame,
  ChartTooltip,
  GradientLegend,
  LegendKey,
  TooltipRow,
  type LegendItem,
} from './ChartFrame';
import { OTHER_HEX, sequentialColor, seriesHex, useChartScheme, type ChartScheme } from './colors';
import { formatCoefficient, formatCount, formatDateTime, formatNumber } from './format';
import { textWidth } from './marks';
import { ols } from './stats';
import { useChartCursor } from './useChartCursor';
import classes from './Chart.module.css';

export interface ScatterCategory {
  key: string;
  label: string;
  /** Categorical slot (0-based); -1 draws the de-emphasis gray ("Other"). */
  slot: number;
}

export type ScatterColor =
  | {
      type: 'category';
      label: string;
      /** Index into `categories` per point. */
      values: ArrayLike<number>;
      categories: ScatterCategory[];
    }
  | {
      type: 'numeric';
      label: string;
      /** A value per point; NaN draws gray. */
      values: ArrayLike<number>;
      format?: (value: number) => string;
    };

export interface ScatterPlotProps {
  title: string;
  subtitle?: ReactNode;
  caption?: ReactNode;
  controls?: ReactNode;
  /** Finite coordinates, one pair per point. */
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  xLabel: string;
  yLabel: string;
  xTime?: boolean;
  yTime?: boolean;
  formatX?: (value: number) => string;
  formatY?: (value: number) => string;
  color?: ScatterColor;
  /** Draw the least-squares line and report r. */
  trend?: boolean;
  /** Draw the line y = x (dashed) with this legend label, e.g. "Predicted = actual". */
  diagonal?: string;
  /** Spread overlapping points of discrete values a little (display only). */
  jitter?: boolean;
  /** Name of point i in the tooltip, e.g. "Row 1,204". */
  pointLabel?: (index: number) => string;
  /** Click (or Enter) on a point: its index. */
  onPointClick?: (index: number) => void;
  /** A point to mark with a ring (e.g. the selected row). */
  highlight?: number | null;
  height?: number;
  width?: number;
}

/** The top margin holds the y axis title. */
const MARGIN = { top: 22, right: 14, bottom: 38 };
/** Above this many points the dots are drawn on a canvas. */
export const CANVAS_THRESHOLD = 1500;
const HIT_RADIUS = 14;
const TABLE_ROWS = 500;

function dotStyle(n: number) {
  if (n <= 300) {
    return { r: 4, alpha: 0.85, ring: true };
  }
  if (n <= CANVAS_THRESHOLD) {
    return { r: 3, alpha: 0.7, ring: false };
  }
  return { r: n <= 6000 ? 2.2 : 1.8, alpha: n <= 6000 ? 0.5 : 0.35, ring: false };
}

/** Deterministic offset in [-0.5, 0.5) for jitter. */
function hash(i: number, salt: number): number {
  const v = Math.sin(i * 12.9898 + salt * 78.233) * 43758.5453;
  return v - Math.floor(v) - 0.5;
}

function makeAxis(lo: number, hi: number, count: number, time: boolean): Axis {
  if (time) {
    const pad = (hi - lo) * 0.03 || 86_400_000;
    return timeTicks(lo - pad, hi + pad, count);
  }
  return valueAxis(lo, hi, count);
}

/**
 * A scatter plot for up to tens of thousands of points: SVG dots for small samples, a canvas
 * above CANVAS_THRESHOLD, SVG axes either way. Colour by category (3 slots + Other) or by a
 * number (one-hue ramp), optional least-squares trend line. The nearest point is read by
 * pointer; the arrow keys walk the points from left to right.
 */
export function ScatterPlot({
  title,
  subtitle,
  caption,
  controls,
  x,
  y,
  xLabel,
  yLabel,
  xTime = false,
  yTime = false,
  formatX,
  formatY,
  color,
  trend = false,
  diagonal,
  jitter = false,
  pointLabel,
  onPointClick,
  highlight = null,
  height = 320,
  width: fixedWidth,
}: ScatterPlotProps) {
  const { ref, width: measured } = useElementSize();
  const width = fixedWidth ?? measured;
  const scheme = useChartScheme();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const clipId = useId();
  const n = Math.min(x.length, y.length);
  const style = dotStyle(n);

  const order = useMemo(() => {
    const indices = Array.from({ length: n }, (_, i) => i);
    return indices.sort((a, b) => x[a] - x[b] || y[a] - y[b]);
  }, [x, y, n]);
  const cursor = useChartCursor(n, {
    onActivate: onPointClick ? (rank) => onPointClick(order[rank]) : undefined,
  });
  const activePoint = cursor.active !== null ? order[cursor.active] : null;

  const fit = useMemo(() => (trend ? ols(x, y) : null), [trend, x, y]);
  const colorRange = useMemo(
    () => (color?.type === 'numeric' ? extent(Array.from(color.values)) : null),
    [color]
  );

  const geometry = useMemo(() => {
    const xr = extent(Array.from({ length: n }, (_, i) => x[i])) ?? [0, 1];
    const yr = extent(Array.from({ length: n }, (_, i) => y[i])) ?? [0, 1];
    const plotHeight = height - MARGIN.top - MARGIN.bottom;
    const yAxis = makeAxis(yr[0], yr[1], Math.max(3, Math.floor(plotHeight / 40)), yTime);
    const yLabels = yAxis.ticks.map((t) => (formatY && !yTime ? formatY(t) : yAxis.format(t)));
    const left = Math.max(32, Math.max(...yLabels.map((l) => textWidth(l))) + 12);
    const plotRight = Math.max(left + 10, width - MARGIN.right);
    const xAxis = makeAxis(
      xr[0],
      xr[1],
      Math.max(2, Math.floor((plotRight - left) / (xTime ? 100 : 72))),
      xTime
    );
    const sx = linearScale(xAxis.domain, [left, plotRight]);
    const sy = linearScale(yAxis.domain, [height - MARGIN.bottom, MARGIN.top]);
    const px = new Float32Array(n);
    const py = new Float32Array(n);
    const spread = jitter ? 6 : 0;
    for (let i = 0; i < n; i += 1) {
      px[i] = sx(x[i]) + spread * hash(i, 1);
      py[i] = sy(y[i]) + spread * hash(i, 2);
    }
    // Grid of point indices for nearest-point lookup.
    const cell = HIT_RADIUS;
    const grid = new Map<number, number[]>();
    for (let i = 0; i < n; i += 1) {
      const key = Math.floor(px[i] / cell) * 100_000 + Math.floor(py[i] / cell);
      const bucket = grid.get(key);
      if (bucket) {
        bucket.push(i);
      } else {
        grid.set(key, [i]);
      }
    }
    return { xAxis, yAxis, yLabels, left, plotRight, sx, sy, px, py, grid, cell };
  }, [x, y, n, width, height, xTime, yTime, jitter, formatY]);

  const { xAxis, yAxis, yLabels, left, plotRight, sx, sy, px, py, grid, cell } = geometry;
  const base = height - MARGIN.bottom;

  const colors = useMemo(() => {
    // A numeric colour takes one of 65 ramp steps (computed once, not per point).
    const ramp = Array.from({ length: 65 }, (_, k) => sequentialColor(k / 64, scheme));
    const out: string[] = new Array(n);
    for (let i = 0; i < n; i += 1) {
      out[i] = pointColor(color, colorRange, i, scheme, ramp);
    }
    return out;
  }, [color, colorRange, n, scheme]);

  // Gray ("Other") points first, so coloured ones stay on top.
  const drawOrder = useMemo(() => {
    const indices = Array.from({ length: n }, (_, i) => i);
    if (color?.type === 'category') {
      const rank = (i: number) => (color.categories[color.values[i]]?.slot ?? -1) < 0;
      return [...indices.filter(rank), ...indices.filter((i) => !rank(i))];
    }
    return indices;
  }, [color, n]);

  const useCanvas = n > CANVAS_THRESHOLD;
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!useCanvas || !canvas || width <= 0) {
      return;
    }
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    let ctx: CanvasRenderingContext2D | null = null;
    try {
      ctx = canvas.getContext('2d');
    } catch {
      ctx = null;
    }
    if (!ctx) {
      return;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, MARGIN.top, plotRight - left, base - MARGIN.top);
    ctx.clip();
    ctx.globalAlpha = style.alpha;
    for (const i of drawOrder) {
      ctx.fillStyle = colors[i];
      ctx.beginPath();
      ctx.arc(px[i], py[i], style.r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }, [
    useCanvas,
    width,
    height,
    px,
    py,
    colors,
    drawOrder,
    style.alpha,
    style.r,
    left,
    plotRight,
    base,
  ]);

  const nearest = (mx: number, my: number): number | null => {
    const cx = Math.floor(mx / cell);
    const cy = Math.floor(my / cell);
    let best: number | null = null;
    let bestDistance = HIT_RADIUS * HIT_RADIUS;
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) {
        for (const i of grid.get((cx + dx) * 100_000 + cy + dy) ?? []) {
          const d = (px[i] - mx) ** 2 + (py[i] - my) ** 2;
          if (d <= bestDistance) {
            bestDistance = d;
            best = i;
          }
        }
      }
    }
    return best;
  };

  // Position in `order` of each point, to move the keyboard cursor to a hovered point.
  const rankOf = useMemo(() => {
    const rank = new Int32Array(n);
    order.forEach((point, k) => {
      rank[point] = k;
    });
    return rank;
  }, [order, n]);

  // Values in full (the axes only label at tick precision, e.g. "2010").
  const fx = formatX ?? (xTime ? formatDateTime : (v: number) => formatNumber(v));
  const fy = formatY ?? (yTime ? formatDateTime : (v: number) => formatNumber(v));
  const colorText = (i: number) => {
    if (!color) {
      return null;
    }
    if (color.type === 'category') {
      return color.categories[color.values[i]]?.label ?? '–';
    }
    const value = color.values[i];
    return Number.isFinite(value) ? (color.format ?? formatNumber)(value) : 'missing';
  };
  const pointText = (i: number) => {
    const c = colorText(i);
    return (
      `${pointLabel ? `${pointLabel(i)}: ` : ''}${xLabel} ${fx(x[i])}, ${yLabel} ${fy(y[i])}` +
      `${c !== null && color ? `, ${color.label} ${c}` : ''}`
    );
  };
  const fitText = fit
    ? `Trend (least squares): r = ${formatCoefficient(fit.r)}, R² = ${formatCoefficient(fit.r2)}`
    : null;
  const summary =
    `${title}. ${formatCount(n)} points, ${xLabel} on the horizontal axis, ${yLabel} on the vertical axis.` +
    `${fitText ? ` ${fitText}, slope ${formatNumber(fit!.slope)}.` : ''}`;

  const legend: LegendItem[] =
    color?.type === 'category'
      ? color.categories.map((c) => ({
          key: c.key,
          label: c.label,
          color: c.slot >= 0 ? seriesHex(c.slot, scheme) : OTHER_HEX,
          shape: 'dot' as const,
        }))
      : [];
  const legendExtra = (
    <>
      {color?.type === 'numeric' && colorRange && (
        <GradientLegend
          label={`Colour: ${color.label} from ${(color.format ?? formatNumber)(colorRange[0])} to ${(color.format ?? formatNumber)(colorRange[1])}`}
          stops={[
            sequentialColor(0, scheme),
            sequentialColor(0.5, scheme),
            sequentialColor(1, scheme),
          ]}
          labels={[
            `${color.label} ${(color.format ?? formatNumber)(colorRange[0])}`,
            (color.format ?? formatNumber)(colorRange[1]),
          ]}
        />
      )}
      {diagonal && (
        <span className={classes.legendItem}>
          <LegendKey color="var(--mantine-color-dimmed)" shape="line" />
          <Text size="xs" c="dimmed">
            {diagonal}
          </Text>
        </span>
      )}
      {fitText && (
        <span className={classes.legendItem}>
          <LegendKey color="var(--mantine-color-text)" shape="line" />
          <Text size="xs" c="dimmed">
            {fitText}
          </Text>
        </span>
      )}
    </>
  );

  const trendLine = (() => {
    if (!fit) {
      return null;
    }
    const [x0, x1] = xAxis.domain;
    return {
      x1: sx(x0),
      y1: sy(fit.intercept + fit.slope * x0),
      x2: sx(x1),
      y2: sy(fit.intercept + fit.slope * x1),
    };
  })();
  const diagonalLine = (() => {
    if (!diagonal) {
      return null;
    }
    const lo = Math.min(xAxis.domain[0], yAxis.domain[0]);
    const hi = Math.max(xAxis.domain[1], yAxis.domain[1]);
    return { x1: sx(lo), y1: sy(lo), x2: sx(hi), y2: sy(hi) };
  })();
  // Trend line and focused point: in the SVG, or in a layer above the canvas.
  const marks = (
    <>
      {diagonalLine && (
        <line
          className={classes.diagonal}
          clipPath={`url(#${clipId})`}
          data-testid="scatter-diagonal"
          {...diagonalLine}
        />
      )}
      {trendLine && <line className={classes.trend} clipPath={`url(#${clipId})`} {...trendLine} />}
      {highlight !== null && highlight >= 0 && highlight < n && (
        <circle
          cx={px[highlight]}
          cy={py[highlight]}
          r={style.r + 5}
          fill="none"
          stroke="var(--mantine-color-text)"
          strokeWidth={2}
          data-testid="scatter-highlight"
        />
      )}
      {activePoint !== null && (
        <circle
          cx={px[activePoint]}
          cy={py[activePoint]}
          r={style.r + 3}
          fill={colors[activePoint]}
          className={classes.ring}
          style={{ stroke: 'var(--mantine-color-text)' }}
        />
      )}
    </>
  );

  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      caption={caption}
      controls={controls}
      legend={legend}
      legendExtra={color?.type === 'numeric' || fitText || diagonal ? legendExtra : undefined}
      table={() => ({
        head: [pointLabel ? 'Point' : '#', xLabel, yLabel, ...(color ? [color.label] : [])],
        rows: order
          .slice(0, TABLE_ROWS)
          .map((i, k) => [
            pointLabel ? pointLabel(i) : k + 1,
            fx(x[i]),
            fy(y[i]),
            ...(color ? [colorText(i) ?? '–'] : []),
          ]),
      })}
    >
      {n === 0 ? (
        <ChartEmpty>No rows with both values</ChartEmpty>
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
                    x={left}
                    y={MARGIN.top}
                    width={Math.max(0, plotRight - left)}
                    height={base - MARGIN.top}
                  />
                </clipPath>
              </defs>
              {yAxis.ticks.map((tick, i) => (
                <g key={tick}>
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
              {xAxis.ticks.map((tick) => (
                <g key={tick}>
                  <line
                    className={classes.grid}
                    x1={sx(tick)}
                    x2={sx(tick)}
                    y1={MARGIN.top}
                    y2={base}
                  />
                  <text className={classes.tick} x={sx(tick)} y={base + 15} textAnchor="middle">
                    {formatX && !xTime ? formatX(tick) : xAxis.format(tick)}
                  </text>
                </g>
              ))}
              <line className={classes.axis} x1={left} x2={plotRight} y1={base} y2={base} />
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
              {!useCanvas && (
                <g clipPath={`url(#${clipId})`}>
                  {drawOrder.map((i) => (
                    <circle
                      key={i}
                      cx={px[i]}
                      cy={py[i]}
                      r={style.r}
                      fill={colors[i]}
                      fillOpacity={style.alpha}
                      className={style.ring ? classes.ring : undefined}
                    />
                  ))}
                </g>
              )}
              {!useCanvas && marks}
            </svg>
          )}
          {useCanvas && width > 0 && (
            <>
              <canvas
                ref={canvasRef}
                className={classes.canvas}
                style={{ width, height }}
                aria-hidden="true"
              />
              <svg className={classes.layer} width={width} height={height} aria-hidden="true">
                {marks}
              </svg>
            </>
          )}
          <div
            className={classes.overlay}
            role="slider"
            tabIndex={0}
            aria-label={`${title}: use the arrow keys to read the points from left to right`}
            aria-valuemin={1}
            aria-valuemax={n}
            aria-valuenow={(cursor.active ?? 0) + 1}
            aria-valuetext={pointText(order[cursor.active ?? 0])}
            onPointerMove={(event) => {
              const rect = event.currentTarget.getBoundingClientRect();
              const hit = nearest(event.clientX - rect.left, event.clientY - rect.top);
              cursor.setActive(hit === null ? null : rankOf[hit]);
            }}
            onPointerLeave={() => cursor.setActive(null)}
            onClick={
              onPointClick
                ? (event) => {
                    const rect = event.currentTarget.getBoundingClientRect();
                    const hit = nearest(event.clientX - rect.left, event.clientY - rect.top);
                    if (hit !== null) {
                      onPointClick(hit);
                    }
                  }
                : undefined
            }
            style={onPointClick ? { cursor: 'pointer' } : undefined}
            onKeyDown={cursor.onKeyDown}
            onFocus={cursor.onFocus}
            onBlur={cursor.onBlur}
          />
          {activePoint !== null && width > 0 && (
            <ChartTooltip x={px[activePoint]} y={Math.max(0, py[activePoint] - 20)} width={width}>
              {pointLabel && (
                <Text size="xs" c="dimmed" mb={2}>
                  {pointLabel(activePoint)}
                </Text>
              )}
              <TooltipRow value={fx(x[activePoint])} label={xLabel} />
              <TooltipRow value={fy(y[activePoint])} label={yLabel} />
              {color && (
                <TooltipRow
                  color={colors[activePoint]}
                  value={colorText(activePoint)}
                  label={color.label}
                />
              )}
            </ChartTooltip>
          )}
        </div>
      )}
    </ChartFrame>
  );
}

function pointColor(
  color: ScatterColor | undefined,
  range: [number, number] | null,
  i: number,
  scheme: ChartScheme,
  ramp: string[]
): string {
  if (!color) {
    return seriesHex(0, scheme);
  }
  if (color.type === 'category') {
    const slot = color.categories[color.values[i]]?.slot ?? -1;
    return slot >= 0 ? seriesHex(slot, scheme) : OTHER_HEX;
  }
  const value = color.values[i];
  if (!Number.isFinite(value) || !range) {
    return OTHER_HEX;
  }
  const t = range[1] > range[0] ? (value - range[0]) / (range[1] - range[0]) : 0.5;
  return ramp[Math.round(Math.min(1, Math.max(0, t)) * (ramp.length - 1))];
}
