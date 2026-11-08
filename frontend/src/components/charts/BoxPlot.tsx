import type { ReactNode } from 'react';
import { Text } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { linearScale } from '@/components/chartScale';
import { innerTicks, timeTicks } from './axes';
import { ChartEmpty, ChartFrame, ChartTooltip, TooltipRow } from './ChartFrame';
import { seriesColor } from './colors';
import { formatCount, formatDateTime, formatNumber } from './format';
import { textWidth, truncate } from './marks';
import type { BoxSummary } from './stats';
import { useChartCursor } from './useChartCursor';
import classes from './Chart.module.css';

export interface BoxGroup {
  key: string;
  label: string;
  /** CSS colour; the first series colour by default. */
  color?: string;
  box: BoxSummary;
}

export interface BoxPlotProps {
  title: string;
  subtitle?: ReactNode;
  caption?: ReactNode;
  controls?: ReactNode;
  groups: BoxGroup[];
  /** Values are timestamps (ms). */
  time?: boolean;
  formatValue?: (value: number) => string;
  width?: number;
}

const MARGIN = { top: 4, right: 14, bottom: 22 };
const ROW = 34;
const BOX = 16;
/** Beyond this many whisker spans the axis is cut and the extremes are named at the edge. */
const MAX_SPREAD = 5;

/**
 * Horizontal box plots on one axis: box from Q1 to Q3, median line, whiskers to the most
 * extreme values within 1.5 × IQR, outliers as dots and the mean as a ring. One row per group.
 */
export function BoxPlot({
  title,
  subtitle,
  caption,
  controls,
  groups,
  time = false,
  formatValue,
  width: fixedWidth,
}: BoxPlotProps) {
  const { ref, width: measured } = useElementSize();
  const width = fixedWidth ?? measured;
  const cursor = useChartCursor(groups.length, { orientation: 'vertical' });
  const height = MARGIN.top + groups.length * ROW + MARGIN.bottom;

  const lows = groups.map((g) => g.box.low);
  const highs = groups.map((g) => g.box.high);
  const mins = groups.map((g) => g.box.min);
  const maxs = groups.map((g) => g.box.max);
  const whiskerLo = Math.min(...lows);
  const whiskerHi = Math.max(...highs);
  const fullLo = Math.min(...mins);
  const fullHi = Math.max(...maxs);
  const span = Math.max(whiskerHi - whiskerLo, 1e-12);
  const cutLo = whiskerLo - fullLo > MAX_SPREAD * span;
  const cutHi = fullHi - whiskerHi > MAX_SPREAD * span;
  const lo = cutLo ? whiskerLo - span * 0.05 : fullLo;
  const hi = cutHi ? whiskerHi + span * 0.05 : fullHi;
  const pad = (hi - lo) * 0.02 || Math.abs(hi) * 0.05 || 1;
  const domain: [number, number] = [lo - pad, hi + pad];

  const showLabels = groups.length > 1;
  const labelChars = Math.max(4, Math.floor(Math.min(160, width * 0.3) / 7));
  const labels = groups.map((g) => truncate(g.label, labelChars));
  const labelWidth = showLabels ? Math.max(...labels.map((l) => textWidth(l, 12))) + 12 : 8;
  // Extremes beyond a cut axis, named at the edge (dates in full, not as milliseconds).
  const edgeText = (v: number) => (time ? formatDateTime(v) : formatNumber(v, 3));
  /** Room for the edge labels of `values` (the right one may also use the margin). */
  const edgeRoom = (values: number[], spare: number) =>
    Math.max(46, ...values.map((v) => textWidth(`▸ ${edgeText(v)}`) + 8 - spare));
  const beyondLo = mins.filter((v) => v < domain[0]);
  const beyondHi = maxs.filter((v) => v > domain[1]);
  const lowEdge = cutLo ? edgeRoom(beyondLo, 0) : 0;
  const highEdge = cutHi ? edgeRoom(beyondHi, MARGIN.right) : 0;
  const plotLeft = labelWidth + lowEdge;
  const plotRight = Math.max(plotLeft + 20, width - MARGIN.right - highEdge);
  const sx = linearScale(domain, [plotLeft, plotRight]);
  const count = Math.max(2, Math.floor((plotRight - plotLeft) / (time ? 96 : 70)));
  const axis = time
    ? timeTicks(domain[0], domain[1], count)
    : innerTicks(domain[0], domain[1], count);
  const format = formatValue ?? (time ? formatDateTime : (v: number) => formatNumber(v));
  const rowTop = (i: number) => MARGIN.top + i * ROW;
  const active = cursor.active;

  const facts = (box: BoxSummary): [string, string][] => [
    ['Median', format(box.median)],
    ['Q1 – Q3', `${format(box.q1)} – ${format(box.q3)}`],
    [box.exact ? 'Whiskers' : 'Whiskers (estimated)', `${format(box.low)} – ${format(box.high)}`],
    ['Min – max', `${format(box.min)} – ${format(box.max)}`],
    ...(box.mean !== null ? ([['Mean', format(box.mean)]] as [string, string][]) : []),
    ...(box.n !== null ? ([['Values', formatCount(box.n)]] as [string, string][]) : []),
    ...(box.outlierCount !== null
      ? ([['Outliers (1.5 × IQR)', formatCount(box.outlierCount)]] as [string, string][])
      : []),
  ];
  const groupText = (i: number) => {
    const g = groups[i];
    return g
      ? `${showLabels ? `${g.label}: ` : ''}${facts(g.box)
          .map(([k, v]) => `${k.toLowerCase()} ${v}`)
          .join(', ')}`
      : '';
  };

  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      caption={caption}
      controls={controls}
      table={() => ({
        head: [
          'Group',
          'Min',
          'Whisker low',
          'Q1',
          'Median',
          'Q3',
          'Whisker high',
          'Max',
          'Mean',
          'Values',
        ],
        rows: groups.map((g) => [
          g.label,
          format(g.box.min),
          format(g.box.low),
          format(g.box.q1),
          format(g.box.median),
          format(g.box.q3),
          format(g.box.high),
          format(g.box.max),
          g.box.mean !== null ? format(g.box.mean) : '–',
          g.box.n !== null ? formatCount(g.box.n) : '–',
        ]),
      })}
    >
      {groups.length === 0 ? (
        <ChartEmpty />
      ) : (
        <div ref={ref} className={classes.plot} style={{ height }}>
          {width > 0 && (
            <svg
              className={classes.svg}
              width={width}
              height={height}
              role="img"
              aria-label={`${title}. ${groups.map((_, i) => groupText(i)).join('. ')}.`}
            >
              {active !== null && (
                <rect
                  className={classes.band}
                  x={0}
                  y={rowTop(active)}
                  width={width}
                  height={ROW}
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
                    {axis.format(tick)}
                  </text>
                </g>
              ))}
              {groups.map((g, i) => {
                const color = g.color ?? seriesColor(0);
                const { box } = g;
                const cy = rowTop(i) + ROW / 2;
                const y0 = cy - BOX / 2;
                const clampX = (v: number) => Math.min(plotRight, Math.max(plotLeft, sx(v)));
                const visible = box.outliers.filter((v) => v >= domain[0] && v <= domain[1]);
                return (
                  <g key={g.key}>
                    {showLabels && (
                      <text
                        className={classes.label}
                        x={labelWidth - 10}
                        y={cy}
                        dy="0.32em"
                        textAnchor="end"
                      >
                        {labels[i]}
                      </text>
                    )}
                    <line
                      className={classes.whisker}
                      x1={clampX(box.low)}
                      x2={clampX(box.q1)}
                      y1={cy}
                      y2={cy}
                    />
                    <line
                      className={classes.whisker}
                      x1={clampX(box.q3)}
                      x2={clampX(box.high)}
                      y1={cy}
                      y2={cy}
                    />
                    <line
                      className={classes.whisker}
                      x1={clampX(box.low)}
                      x2={clampX(box.low)}
                      y1={cy - 5}
                      y2={cy + 5}
                    />
                    <line
                      className={classes.whisker}
                      x1={clampX(box.high)}
                      x2={clampX(box.high)}
                      y1={cy - 5}
                      y2={cy + 5}
                    />
                    <rect
                      x={clampX(box.q1)}
                      y={y0}
                      width={Math.max(1, clampX(box.q3) - clampX(box.q1))}
                      height={BOX}
                      rx={3}
                      style={{ fill: color, stroke: color }}
                      fillOpacity={0.18}
                      strokeWidth={1.5}
                    />
                    <line
                      x1={clampX(box.median)}
                      x2={clampX(box.median)}
                      y1={y0}
                      y2={y0 + BOX}
                      style={{ stroke: color }}
                      strokeWidth={2.5}
                    />
                    {visible.map((v, k) => (
                      <circle
                        key={k}
                        cx={sx(v)}
                        cy={cy}
                        r={2.5}
                        style={{ fill: color }}
                        opacity={0.55}
                      />
                    ))}
                    {box.mean !== null && box.mean >= domain[0] && box.mean <= domain[1] && (
                      <circle
                        cx={sx(box.mean)}
                        cy={cy}
                        r={3.5}
                        style={{ stroke: color, fill: 'var(--chart-surface)' }}
                        strokeWidth={2}
                      />
                    )}
                    {cutHi && box.max > domain[1] && (
                      <text className={classes.value} x={plotRight + 4} y={cy} dy="0.32em">
                        {`▸ ${edgeText(box.max)}`}
                      </text>
                    )}
                    {cutLo && box.min < domain[0] && (
                      <text
                        className={classes.value}
                        x={plotLeft - 4}
                        y={cy}
                        dy="0.32em"
                        textAnchor="end"
                      >
                        {`${edgeText(box.min)} ◂`}
                      </text>
                    )}
                  </g>
                );
              })}
              <line
                className={classes.axis}
                x1={plotLeft}
                x2={plotRight}
                y1={height - MARGIN.bottom}
                y2={height - MARGIN.bottom}
              />
            </svg>
          )}
          <div
            className={classes.overlay}
            role="slider"
            tabIndex={0}
            aria-orientation="vertical"
            aria-label={`${title}: use the arrow keys to read the groups`}
            aria-valuemin={1}
            aria-valuemax={groups.length}
            aria-valuenow={(active ?? 0) + 1}
            aria-valuetext={groupText(active ?? 0)}
            onPointerMove={(event) => {
              const py =
                event.clientY - event.currentTarget.getBoundingClientRect().top - MARGIN.top;
              const index = Math.floor(py / ROW);
              cursor.setActive(index >= 0 && index < groups.length ? index : null);
            }}
            onPointerLeave={() => cursor.setActive(null)}
            onKeyDown={cursor.onKeyDown}
            onFocus={cursor.onFocus}
            onBlur={cursor.onBlur}
          />
          {active !== null && width > 0 && (
            <ChartTooltip x={sx(groups[active].box.q3)} y={rowTop(active) + ROW - 4} width={width}>
              {showLabels && (
                <Text size="xs" c="dimmed" mb={2}>
                  {groups[active].label}
                </Text>
              )}
              {facts(groups[active].box).map(([label, value]) => (
                <TooltipRow key={label} value={value} label={label} />
              ))}
            </ChartTooltip>
          )}
        </div>
      )}
    </ChartFrame>
  );
}
