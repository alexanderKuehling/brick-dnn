import { useMemo, useRef, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react';
import { Group, NumberInput, Paper, Stack, Text } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { formatCount, formatNumber } from '@/components/charts/format';
import { countBelow, histogramCounts, histogramEdges } from './rowsData';
import classes from './RowsView.module.css';

export interface ThresholdHistogramProps {
  title: string;
  description?: ReactNode;
  /** One value per row (null: none). */
  values: (number | null)[];
  /** Values of a baseline, drawn as an outline scaled to the same number of rows. */
  baseline?: (number | null)[];
  baselineLabel?: string;
  threshold: number;
  onThresholdChange: (value: number) => void;
  /** A suggested threshold (marked on the axis), or null. */
  suggested: number | null;
  /** Largest value on the axis. */
  cap: number;
  /** "rows below" text, e.g. (n) => `${n} rows are unusually typical`. */
  belowText: (count: number) => ReactNode;
  xLabel: string;
  actions?: ReactNode;
  bins?: number;
  height?: number;
}

const MARGIN = { top: 8, right: 10, bottom: 22, left: 34 };

/**
 * A histogram of per-row values with a threshold the user drags (pointer or arrow keys) or
 * types; rows below it are coloured and counted. An optional baseline (e.g. random pairs) is
 * drawn as an outline for comparison.
 */
export function ThresholdHistogram({
  title,
  description,
  values,
  baseline,
  baselineLabel,
  threshold,
  onThresholdChange,
  suggested,
  cap,
  belowText,
  xLabel,
  actions,
  bins = 40,
  height = 170,
}: ThresholdHistogramProps) {
  const { ref, width: measured } = useElementSize();
  const width = measured || 480;
  const edges = useMemo(
    () => histogramEdges([...values, ...(baseline ?? [])], bins, cap),
    [values, baseline, bins, cap]
  );
  const counts = useMemo(() => histogramCounts(values, edges), [values, edges]);
  const baseCounts = useMemo(() => {
    if (!baseline || baseline.length === 0) {
      return null;
    }
    const raw = histogramCounts(baseline, edges);
    const known = values.filter((v) => v !== null).length;
    const total = raw.reduce((a, b) => a + b, 0) || 1;
    return raw.map((c) => (c * known) / total);
  }, [baseline, edges, values]);
  const below = useMemo(() => countBelow(values, threshold), [values, threshold]);

  const top = edges[edges.length - 1];
  const innerW = Math.max(10, width - MARGIN.left - MARGIN.right);
  const innerH = height - MARGIN.top - MARGIN.bottom;
  const maxCount = Math.max(1, ...counts, ...(baseCounts ?? []));
  const sx = (x: number) => MARGIN.left + (Math.min(Math.max(x, 0), top) / top) * innerW;
  const sy = (c: number) => MARGIN.top + innerH - (c / maxCount) * innerH;
  const barW = innerW / counts.length;
  const svgRef = useRef<SVGSVGElement>(null);
  const dragging = useRef(false);
  const step = top / 200;

  const fromPointer = (event: PointerEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left - MARGIN.left) / innerW) * top;
    const value = Math.min(top, Math.max(0, x));
    onThresholdChange(Number(value.toPrecision(2)));
  };
  const onKeyDown = (event: KeyboardEvent<SVGSVGElement>) => {
    const moves: Record<string, number> = {
      ArrowLeft: -step,
      ArrowDown: -step,
      ArrowRight: step,
      ArrowUp: step,
      PageDown: -step * 10,
      PageUp: step * 10,
    };
    if (event.key in moves) {
      event.preventDefault();
      const next = Math.min(top, Math.max(0, threshold + moves[event.key]));
      onThresholdChange(Number(next.toPrecision(3)));
    }
  };

  const ticks = [0, top / 4, top / 2, (3 * top) / 4, top];
  const outline = baseCounts
    ? baseCounts
        .map((c, i) => {
          const x0 = MARGIN.left + i * barW;
          return `${i === 0 ? 'M' : 'L'}${x0},${sy(c)} L${x0 + barW},${sy(c)}`;
        })
        .join(' ')
    : null;

  return (
    <Paper withBorder radius="md" p="sm" className={classes.card}>
      <Stack gap={6}>
        <div>
          <Text fw={600} size="sm">
            {title}
          </Text>
          {description && (
            <Text size="xs" c="dimmed">
              {description}
            </Text>
          )}
        </div>
        <div ref={ref} className={classes.chart}>
          <svg
            ref={svgRef}
            width={width}
            height={height}
            role="slider"
            tabIndex={0}
            aria-label={`${title}: threshold`}
            aria-valuemin={0}
            aria-valuemax={top}
            aria-valuenow={threshold}
            aria-valuetext={`${formatNumber(threshold, 3)}: ${formatCount(below)} rows below`}
            className={classes.histogram}
            onPointerDown={(event) => {
              dragging.current = true;
              event.currentTarget.setPointerCapture?.(event.pointerId);
              fromPointer(event);
            }}
            onPointerMove={(event) => dragging.current && fromPointer(event)}
            onPointerUp={() => {
              dragging.current = false;
            }}
            onKeyDown={onKeyDown}
          >
            {counts.map((count, i) => {
              const x0 = MARGIN.left + i * barW;
              const lowBin = edges[i + 1] <= threshold;
              return (
                <rect
                  key={i}
                  x={x0 + 0.5}
                  y={sy(count)}
                  width={Math.max(0.5, barW - 1)}
                  height={Math.max(0, MARGIN.top + innerH - sy(count))}
                  className={lowBin ? classes.barBelow : classes.bar}
                  data-testid={lowBin ? 'bar-below' : 'bar'}
                />
              );
            })}
            {outline && <path d={outline} className={classes.baseline} />}
            <line
              x1={MARGIN.left}
              x2={MARGIN.left + innerW}
              y1={MARGIN.top + innerH}
              y2={MARGIN.top + innerH}
              className={classes.axis}
            />
            {ticks.map((t) => (
              <text key={t} x={sx(t)} y={height - 6} textAnchor="middle" className={classes.tick}>
                {formatNumber(t, 2)}
              </text>
            ))}
            <text x={4} y={MARGIN.top + 8} className={classes.tick}>
              {formatCount(maxCount)}
            </text>
            {suggested !== null && suggested <= top && (
              <g>
                <line
                  x1={sx(suggested)}
                  x2={sx(suggested)}
                  y1={MARGIN.top}
                  y2={MARGIN.top + innerH}
                  className={classes.suggested}
                />
              </g>
            )}
            <line
              x1={sx(threshold)}
              x2={sx(threshold)}
              y1={MARGIN.top - 4}
              y2={MARGIN.top + innerH}
              className={classes.threshold}
            />
            <circle cx={sx(threshold)} cy={MARGIN.top} r={5} className={classes.handle} />
          </svg>
        </div>
        <Group gap="sm" wrap="wrap" align="flex-end" justify="space-between">
          <Group gap="xs" align="flex-end" wrap="nowrap">
            <NumberInput
              size="xs"
              w={110}
              label="Threshold"
              aria-label={`Threshold (${xLabel})`}
              value={threshold}
              min={0}
              step={step * 5}
              decimalScale={3}
              allowNegative={false}
              onChange={(value) => typeof value === 'number' && onThresholdChange(value)}
            />
            <Text size="sm" data-testid="below-count">
              {belowText(below)}
            </Text>
          </Group>
          {actions}
        </Group>
        <Group gap="md">
          <Text size="xs" c="dimmed">
            <span className={classes.keyBelow} aria-hidden /> below the threshold
          </Text>
          {baseline && baselineLabel && (
            <Text size="xs" c="dimmed">
              <span className={classes.keyBaseline} aria-hidden /> {baselineLabel}
            </Text>
          )}
          {suggested !== null && (
            <Text size="xs" c="dimmed">
              <span className={classes.keySuggested} aria-hidden /> clear gap at{' '}
              {formatNumber(suggested, 3)}
            </Text>
          )}
        </Group>
      </Stack>
    </Paper>
  );
}
