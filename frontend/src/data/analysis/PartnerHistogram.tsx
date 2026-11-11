import { useRef, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react';
import { Text } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { valueAxis } from '@/components/charts/axes';
import classes from '@/components/charts/Chart.module.css';
import { ChartFrame } from '@/components/charts/ChartFrame';
import { seriesColor } from '@/components/charts/colors';
import { formatCount } from '@/components/charts/format';
import { columnPath, textWidth } from '@/components/charts/marks';
import { linearScale } from '@/components/chartScale';

export interface PartnerHistogramProps {
  title: string;
  subtitle?: ReactNode;
  caption?: ReactNode;
  /** Bins of the strongest-partner |ρ| from 0 to 1. */
  edges: number[];
  counts: number[];
  threshold: number;
  onThresholdChange: (threshold: number) => void;
  /** |ρ| unrelated columns reach by chance (median), drawn as a dashed line. */
  chance: number | null;
  /** Columns below the threshold and all columns (exact, not from the bins). */
  below: number;
  total: number;
  height?: number;
  width?: number;
}

const MARGIN = { top: 14, right: 12, bottom: 26 };
export const BELOW_COLOR = seriesColor(1);
export const ABOVE_COLOR = seriesColor(0);

/** Rounds to 0.01 and keeps the threshold in [0, 1]. */
export function clampThreshold(value: number): number {
  return Math.min(1, Math.max(0, Math.round(value * 100) / 100));
}

/**
 * The histogram of every column's strongest-partner |ρ| with a threshold the user drags (or
 * moves with the arrow keys): bars below it in one colour, above in another. Noise columns
 * form a hump near 0, related columns one further right.
 */
export function PartnerHistogram({
  title,
  subtitle,
  caption,
  edges,
  counts,
  threshold,
  onThresholdChange,
  chance,
  below,
  total,
  height = 180,
  width: fixedWidth,
}: PartnerHistogramProps) {
  const { ref, width: measured } = useElementSize();
  const width = fixedWidth ?? measured;
  const dragging = useRef(false);
  const max = Math.max(1, ...counts);
  const base = height - MARGIN.bottom;
  const yAxis = valueAxis(0, max, Math.max(3, Math.floor((base - MARGIN.top) / 36)), [0, Infinity]);
  const yLabels = yAxis.ticks.map(yAxis.format);
  const left = Math.max(28, Math.max(...yLabels.map((l) => textWidth(l))) + 10);
  const right = Math.max(left + 1, width - MARGIN.right);
  const sx = linearScale([0, 1], [left, right]);
  const sy = linearScale(yAxis.domain, [base, MARGIN.top]);
  const xTicks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1].filter(
    (_, i) => width > 420 || i % 2 === 0
  );

  const fromPointer = (event: PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const value = (event.clientX - rect.left - left) / Math.max(1, right - left);
    onThresholdChange(clampThreshold(value));
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 0.05 : 0.01;
    const moves: Record<string, number> = {
      ArrowLeft: -step,
      ArrowDown: -step,
      ArrowRight: step,
      ArrowUp: step,
    };
    if (event.key in moves) {
      event.preventDefault();
      onThresholdChange(clampThreshold(threshold + moves[event.key]));
    } else if (event.key === 'Home' || event.key === 'End') {
      event.preventDefault();
      onThresholdChange(event.key === 'Home' ? 0 : 1);
    }
  };
  const valueText = `Threshold |ρ| ${threshold.toFixed(2)}: ${formatCount(below)} of ${formatCount(total)} columns have no partner above it`;

  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      caption={caption}
      legend={[
        { key: 'below', label: 'No partner above the threshold', color: BELOW_COLOR },
        { key: 'above', label: 'With a partner above it', color: ABOVE_COLOR },
      ]}
      table={() => ({
        head: ['Strongest |ρ|', 'Columns'],
        rows: counts.map((count, i) => [
          `${edges[i].toFixed(2)} – ${edges[i + 1].toFixed(2)}`,
          formatCount(count),
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
            aria-label={`${title}. ${valueText}.`}
          >
            {yAxis.ticks.map((tick, i) => (
              <g key={tick}>
                <line className={classes.grid} x1={left} x2={right} y1={sy(tick)} y2={sy(tick)} />
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
            {counts.map((count, i) => {
              const x0 = sx(edges[i]);
              const w = sx(edges[i + 1]) - x0;
              const gap = w >= 6 ? 1 : 0;
              const isBelow = edges[i + 1] <= threshold + 1e-9;
              return count > 0 ? (
                <path
                  key={i}
                  d={columnPath(
                    x0 + gap / 2,
                    Math.max(0.5, w - gap),
                    sy(count),
                    base,
                    w > 6 ? 2 : 1
                  )}
                  style={{ fill: isBelow ? BELOW_COLOR : ABOVE_COLOR }}
                />
              ) : null;
            })}
            {chance !== null && (
              <g>
                <line
                  x1={sx(chance)}
                  x2={sx(chance)}
                  y1={MARGIN.top}
                  y2={base}
                  stroke="var(--mantine-color-dimmed)"
                  strokeDasharray="3 3"
                />
                <text className={classes.tick} x={sx(chance) + 4} y={MARGIN.top + 2} dy="0.7em">
                  chance
                </text>
              </g>
            )}
            <line className={classes.axis} x1={left} x2={right} y1={base} y2={base} />
            {xTicks.map((tick) => (
              <text
                key={tick}
                className={classes.tick}
                x={sx(tick)}
                y={base + 16}
                textAnchor="middle"
              >
                {tick.toFixed(1)}
              </text>
            ))}
            <line
              x1={sx(threshold)}
              x2={sx(threshold)}
              y1={MARGIN.top - 8}
              y2={base}
              stroke="var(--mantine-color-text)"
              strokeWidth={2}
            />
            <circle cx={sx(threshold)} cy={MARGIN.top - 6} r={5} fill="var(--mantine-color-text)" />
          </svg>
        )}
        <div
          className={classes.overlay}
          data-clickable
          role="slider"
          tabIndex={0}
          aria-label={`${title}: threshold (drag, or use the arrow keys)`}
          aria-valuemin={0}
          aria-valuemax={1}
          aria-valuenow={threshold}
          aria-valuetext={valueText}
          style={{ cursor: 'ew-resize', touchAction: 'none' }}
          onPointerDown={(event) => {
            dragging.current = true;
            event.currentTarget.setPointerCapture?.(event.pointerId);
            fromPointer(event);
          }}
          onPointerMove={(event) => {
            if (dragging.current) {
              fromPointer(event);
            }
          }}
          onPointerUp={() => {
            dragging.current = false;
          }}
          onPointerCancel={() => {
            dragging.current = false;
          }}
          onKeyDown={onKeyDown}
        />
      </div>
      <Text size="xs" c="dimmed" mt={2} aria-live="polite">
        {formatCount(below)} of {formatCount(total)} columns have no partner at |ρ| ≥{' '}
        {threshold.toFixed(2)}
      </Text>
    </ChartFrame>
  );
}
