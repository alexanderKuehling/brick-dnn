/**
 * A tiny bar sparkline for column headers and lists: the shape of a distribution, no axes.
 * Decorative by default (the numbers are shown elsewhere); pass `label` to describe it.
 */
export interface MiniHistogramProps {
  counts: number[];
  width?: number;
  height?: number;
  /** A CSS colour; defaults to the first chart series colour. */
  color?: string;
  /** Accessible description; without it the chart is hidden from screen readers. */
  label?: string;
}

export function MiniHistogram({
  counts,
  width = 72,
  height = 20,
  color = 'var(--chart-series-1, var(--mantine-color-blue-6))',
  label,
}: MiniHistogramProps) {
  const max = Math.max(0, ...counts);
  const gap = counts.length > 24 ? 0 : 1;
  const barWidth = counts.length > 0 ? (width - gap * (counts.length - 1)) / counts.length : 0;
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role={label ? 'img' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      style={{ display: 'block', flexShrink: 0 }}
    >
      {counts.map((count, index) => {
        // Non-empty bins stay visible even next to a very tall bin.
        const h = max > 0 ? Math.max(count > 0 ? 1.5 : 0, (count / max) * height) : 0;
        return (
          <rect
            key={index}
            x={index * (barWidth + gap)}
            y={height - h}
            width={Math.max(barWidth, 0.5)}
            height={h}
            rx={barWidth > 3 ? 1 : 0}
            fill={color}
          />
        );
      })}
    </svg>
  );
}
