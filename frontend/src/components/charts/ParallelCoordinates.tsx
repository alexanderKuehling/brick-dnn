import { useMemo, useState, type ReactNode } from 'react';
import { Text } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { i18n, useTranslation } from '@/i18n';
import { ChartEmpty, ChartFrame, ChartTooltip, TooltipRow, type ChartTable } from './ChartFrame';
import { formatNumber } from './format';
import { truncate } from './marks';
import { overlayStyle } from './OverlayChart';
import classes from './Chart.module.css';

/** One vertical axis: a number (or a word, e.g. the optimizer) per item. */
export interface ParallelAxis {
  key: string;
  label: string;
  values: (number | string | null | undefined)[];
  format?: (value: number) => string;
}

export interface ParallelItem {
  key: string;
  label: string;
}

const MARGIN = { top: 34, right: 24, bottom: 26, left: 24 };

interface Scaled {
  /** 0 (bottom) to 1 (top) per item, null without a value. */
  positions: (number | null)[];
  /** Labels at the bottom and the top of the axis. */
  low: string;
  high: string;
  text: (index: number) => string;
}

function scaleAxis(axis: ParallelAxis): Scaled {
  const format = axis.format ?? ((v: number) => formatNumber(v));
  const numbers = axis.values.filter((v): v is number => typeof v === 'number' && isFinite(v));
  const words = Array.from(
    new Set(axis.values.filter((v): v is string => typeof v === 'string'))
  ).sort();
  const text = (index: number) => {
    const value = axis.values[index];
    if (typeof value === 'number') {
      return format(value);
    }
    return typeof value === 'string' ? value : '–';
  };
  if (words.length > 0 && numbers.length === 0) {
    const step = words.length > 1 ? 1 / (words.length - 1) : 0;
    return {
      positions: axis.values.map((v) =>
        typeof v === 'string' ? (words.length > 1 ? words.indexOf(v) * step : 0.5) : null
      ),
      low: words[0],
      high: words[words.length - 1],
      text,
    };
  }
  if (numbers.length === 0) {
    return { positions: axis.values.map(() => null), low: '', high: '', text };
  }
  const lo = Math.min(...numbers);
  const hi = Math.max(...numbers);
  return {
    positions: axis.values.map((v) =>
      typeof v === 'number' && isFinite(v) ? (hi === lo ? 0.5 : (v - lo) / (hi - lo)) : null
    ),
    low: format(lo),
    high: format(hi),
    text,
  };
}

/**
 * Parallel coordinates in plain SVG: one vertical axis per setting or metric (each from its
 * smallest to its largest value), one line per item (a run) in its colour of the overlay
 * chart. Arrow keys walk the items and show the values of one; the table view lists them all.
 */
export function ParallelCoordinates({
  title,
  subtitle,
  axes,
  items,
  caption,
  controls,
  height = 300,
  width: fixedWidth,
}: {
  title: string;
  subtitle?: ReactNode;
  axes: ParallelAxis[];
  items: ParallelItem[];
  caption?: ReactNode;
  controls?: ReactNode;
  height?: number;
  width?: number;
}) {
  useTranslation();
  const { ref, width: measured } = useElementSize();
  const width = fixedWidth ?? measured;
  const [active, setActive] = useState<number | null>(null);
  const scaled = useMemo(() => axes.map(scaleAxis), [axes]);
  const plotLeft = MARGIN.left;
  const plotRight = Math.max(plotLeft + 10, width - MARGIN.right);
  const top = MARGIN.top;
  const bottom = height - MARGIN.bottom;
  const axisX = (i: number) =>
    axes.length > 1 ? plotLeft + (i * (plotRight - plotLeft)) / (axes.length - 1) : plotLeft;
  const y = (position: number) => bottom - position * (bottom - top);
  const maxLabel = Math.max(6, Math.floor((plotRight - plotLeft) / Math.max(1, axes.length) / 7));

  const itemText = (index: number) =>
    `${items[index].label}: ${axes
      .map((axis, a) => `${axis.label} ${scaled[a].text(index)}`)
      .join(', ')}`;
  const summary = i18n.t('ui.components.ParallelCoordinates.summary', {
    title,
    count: items.length,
    axes: axes.map((a) => a.label).join(', '),
  });
  const table = (): ChartTable => ({
    head: ['', ...axes.map((a) => a.label)],
    rows: items.map((item, i) => [item.label, ...scaled.map((s) => s.text(i))]),
  });
  // The line nearest to the pointer, at the axis nearest to it.
  const pick = (event: React.PointerEvent<HTMLElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    let axis = 0;
    axes.forEach((_, a) => {
      if (Math.abs(axisX(a) - px) < Math.abs(axisX(axis) - px)) {
        axis = a;
      }
    });
    let best: number | null = null;
    let distance = 24;
    scaled[axis]?.positions.forEach((position, i) => {
      if (position !== null && Math.abs(y(position) - py) < distance) {
        distance = Math.abs(y(position) - py);
        best = i;
      }
    });
    setActive(best);
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (items.length === 0) {
      return;
    }
    const last = items.length - 1;
    const current = active ?? 0;
    const moves: Record<string, number | null> = {
      ArrowDown: Math.min(last, current + 1),
      ArrowRight: Math.min(last, current + 1),
      ArrowUp: Math.max(0, current - 1),
      ArrowLeft: Math.max(0, current - 1),
      Home: 0,
      End: last,
      Escape: null,
    };
    if (event.key in moves) {
      event.preventDefault();
      setActive(moves[event.key]);
    }
  };

  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      legend={items.map((item, i) => ({
        key: item.key,
        label: item.label,
        color: overlayStyle(i).color,
        shape: 'line',
        dashed: overlayStyle(i).dashed,
      }))}
      caption={caption}
      controls={controls}
      table={table}
    >
      {axes.length < 2 || items.length === 0 ? (
        <ChartEmpty>{i18n.t('ui.components.ParallelCoordinates.needTwo')}</ChartEmpty>
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
              {axes.map((axis, a) => (
                <g key={axis.key}>
                  <line className={classes.axis} x1={axisX(a)} x2={axisX(a)} y1={top} y2={bottom} />
                  <text
                    className={classes.label}
                    x={axisX(a)}
                    y={12}
                    textAnchor={a === 0 ? 'start' : a === axes.length - 1 ? 'end' : 'middle'}
                  >
                    {truncate(axis.label, maxLabel)}
                  </text>
                  <text
                    className={classes.tick}
                    x={axisX(a)}
                    y={top - 6}
                    textAnchor={a === 0 ? 'start' : a === axes.length - 1 ? 'end' : 'middle'}
                  >
                    {truncate(scaled[a].high, maxLabel)}
                  </text>
                  <text
                    className={classes.tick}
                    x={axisX(a)}
                    y={bottom + 14}
                    textAnchor={a === 0 ? 'start' : a === axes.length - 1 ? 'end' : 'middle'}
                  >
                    {truncate(scaled[a].low, maxLabel)}
                  </text>
                </g>
              ))}
              {items.map((item, i) => {
                const style = overlayStyle(i);
                const points = scaled
                  .map((s, a) => (s.positions[i] === null ? null : [axisX(a), y(s.positions[i]!)]))
                  .filter((p): p is number[] => p !== null);
                const dimmed = active !== null && active !== i;
                return (
                  <polyline
                    key={item.key}
                    data-testid={`parallel-${item.key}`}
                    points={points.map((p) => p.join(',')).join(' ')}
                    fill="none"
                    stroke={style.color}
                    strokeWidth={active === i ? 3.5 : 2}
                    strokeOpacity={dimmed ? 0.25 : 1}
                    strokeDasharray={style.dashed ? '6 4' : undefined}
                    strokeLinejoin="round"
                  />
                );
              })}
            </svg>
          )}
          <div
            className={classes.overlay}
            role="slider"
            tabIndex={0}
            aria-label={i18n.t('ui.components.ParallelCoordinates.useTheArrowKeys', { title })}
            aria-valuemin={1}
            aria-valuemax={items.length}
            aria-valuenow={(active ?? 0) + 1}
            aria-valuetext={itemText(active ?? 0)}
            onKeyDown={onKeyDown}
            onPointerMove={pick}
            onPointerLeave={() => setActive(null)}
            onFocus={() => setActive((current) => current ?? 0)}
            onBlur={() => setActive(null)}
          />
          {active !== null && width > 0 && (
            <ChartTooltip x={plotLeft} y={top} width={width}>
              <Text size="xs" fw={600} mb={2}>
                {items[active].label}
              </Text>
              {axes.map((axis, a) => (
                <TooltipRow key={axis.key} value={scaled[a].text(active)} label={axis.label} />
              ))}
            </ChartTooltip>
          )}
        </div>
      )}
    </ChartFrame>
  );
}
