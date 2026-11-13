import { useId } from 'react';
import { Group, Text } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { linearScale } from '@/components/chartScale';
import { innerTicks } from './axes';
import { ChartEmpty, ChartFrame, type ChartTable, type LegendItem } from './ChartFrame';
import { seriesColor } from './colors';
import classes from './Chart.module.css';

export interface CurveSeries {
  key: string;
  label: string;
  /** Points in drawing order, both coordinates from 0 to 1. */
  x: number[];
  y: number[];
  /** A highlighted point of the series (e.g. the chosen threshold), or null. */
  marker?: { x: number; y: number } | null;
}

const MARGIN = { top: 22, right: 14, bottom: 38, left: 40 };

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/**
 * Curves in the unit square (rates from 0 to 1 on both axes), such as a precision-recall or a
 * ROC curve: one line per series with an optional marker, and an optional dashed diagonal
 * (a model that guesses). The data is also available as a table.
 */
export function CurveChart({
  title,
  subtitle,
  series,
  xLabel,
  yLabel,
  diagonal,
  height = 240,
  caption,
  table,
  width: fixedWidth,
}: {
  title: string;
  subtitle?: string;
  series: CurveSeries[];
  xLabel: string;
  yLabel: string;
  /** Legend text of the dashed line y = x; no line without it. */
  diagonal?: string;
  height?: number;
  caption?: string;
  table?: () => ChartTable;
  /** A fixed width in pixels (tests); by default the width of the container. */
  width?: number;
}) {
  const { ref, width: measured } = useElementSize();
  const width = fixedWidth ?? measured;
  const clipId = useId();
  const drawn = series.filter((s) => s.x.length > 0);
  const legend: LegendItem[] = drawn.map((s, i) => ({
    key: s.key,
    label: s.label,
    color: seriesColor(i),
    shape: 'line',
  }));
  const plotRight = Math.max(MARGIN.left + 10, width - MARGIN.right);
  const base = height - MARGIN.bottom;
  const sx = linearScale([0, 1], [MARGIN.left, plotRight]);
  const sy = linearScale([0, 1], [base, MARGIN.top]);
  const ticks = innerTicks(0, 1, 5).ticks;
  const summary = `${title}: ${yLabel} against ${xLabel}${
    drawn.length ? `, ${drawn.map((s) => s.label).join(' and ')}` : ''
  }.`;
  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      legend={legend}
      legendExtra={
        diagonal ? (
          <Group gap={6} wrap="nowrap">
            <svg width={14} height={10} aria-hidden="true">
              <line className={classes.diagonal} x1={1} x2={13} y1={9} y2={1} />
            </svg>
            <Text size="xs" c="dimmed">
              {diagonal}
            </Text>
          </Group>
        ) : undefined
      }
      caption={caption}
      table={table}
    >
      {drawn.length === 0 ? (
        <ChartEmpty>No points to show</ChartEmpty>
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
                    x={MARGIN.left - 4}
                    y={MARGIN.top - 4}
                    width={Math.max(0, plotRight - MARGIN.left + 8)}
                    height={base - MARGIN.top + 8}
                  />
                </clipPath>
              </defs>
              {ticks.map((tick) => (
                <g key={`y${tick}`}>
                  <line
                    className={classes.grid}
                    x1={MARGIN.left}
                    x2={plotRight}
                    y1={sy(tick)}
                    y2={sy(tick)}
                  />
                  <text
                    className={classes.tick}
                    x={MARGIN.left - 8}
                    y={sy(tick)}
                    dy="0.32em"
                    textAnchor="end"
                  >
                    {percent(tick)}
                  </text>
                </g>
              ))}
              {ticks.map((tick) => (
                <g key={`x${tick}`}>
                  <line
                    className={classes.grid}
                    x1={sx(tick)}
                    x2={sx(tick)}
                    y1={MARGIN.top}
                    y2={base}
                  />
                  <text className={classes.tick} x={sx(tick)} y={base + 15} textAnchor="middle">
                    {percent(tick)}
                  </text>
                </g>
              ))}
              <line className={classes.axis} x1={MARGIN.left} x2={plotRight} y1={base} y2={base} />
              <text
                className={classes.axisTitle}
                x={(MARGIN.left + plotRight) / 2}
                y={height - 4}
                textAnchor="middle"
              >
                {xLabel}
              </text>
              <text className={classes.axisTitle} x={0} y={10} textAnchor="start">
                {`↑ ${yLabel}`}
              </text>
              {diagonal && (
                <line
                  className={classes.diagonal}
                  x1={sx(0)}
                  y1={sy(0)}
                  x2={sx(1)}
                  y2={sy(1)}
                  data-testid="curve-diagonal"
                />
              )}
              <g clipPath={`url(#${clipId})`}>
                {drawn.map((s, i) => (
                  <polyline
                    key={s.key}
                    points={s.x.map((x, k) => `${sx(x)},${sy(s.y[k])}`).join(' ')}
                    fill="none"
                    stroke={seriesColor(i)}
                    strokeWidth={2}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                    data-testid={`curve-${s.key}`}
                  />
                ))}
                {drawn.map((s, i) =>
                  s.marker ? (
                    <circle
                      key={`${s.key}-marker`}
                      cx={sx(s.marker.x)}
                      cy={sy(s.marker.y)}
                      r={5}
                      fill={seriesColor(i)}
                      className={classes.ring}
                      data-testid={`curve-marker-${s.key}`}
                    />
                  ) : null
                )}
              </g>
            </svg>
          )}
        </div>
      )}
    </ChartFrame>
  );
}
