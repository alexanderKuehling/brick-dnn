/** Numbers in explanations: a few significant digits, no exponent for everyday sizes. */
export function formatValue(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '–';
  }
  if (value === 0) {
    return '0';
  }
  const size = Math.abs(value);
  if (size >= 1e6 || size < 1e-4) {
    return value.toExponential(Math.max(0, digits - 1));
  }
  return Number(value.toPrecision(digits)).toLocaleString('en-US', { maximumFractionDigits: 6 });
}

export function formatSigned(value: number | null | undefined, digits = 3): string {
  const text = formatValue(value, digits);
  return value !== null && value !== undefined && value > 0 ? `+${text}` : text;
}

/** "1 × 28 × 28" */
export function formatShape(shape: readonly number[] | null | undefined): string {
  return shape && shape.length > 0 ? shape.join(' × ') : 'scalar';
}

export function percent(value: number | null | undefined): string {
  return value === null || value === undefined ? '–' : `${formatValue(value * 100, 3)} %`;
}

/** The heat colours of the server's pictures (render.HEAT_STOPS), for t in [0, 1]. */
const HEAT_STOPS = [
  [0, 0, 4],
  [40, 11, 84],
  [101, 21, 110],
  [159, 42, 99],
  [212, 72, 66],
  [245, 125, 21],
  [250, 193, 39],
  [252, 255, 164],
];

export function heatColor(t: number): string {
  const position = Math.max(0, Math.min(1, t)) * (HEAT_STOPS.length - 1);
  const low = Math.floor(position);
  const high = Math.min(low + 1, HEAT_STOPS.length - 1);
  const share = position - low;
  const [r, g, b] = HEAT_STOPS[low].map((c, i) =>
    Math.round(c * (1 - share) + HEAT_STOPS[high][i] * share)
  );
  return `rgb(${r}, ${g}, ${b})`;
}
