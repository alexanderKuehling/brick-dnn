import { useState } from 'react';
import { Badge, Group, Paper, SegmentedControl, Select, Stack, Table, Text } from '@mantine/core';
import { formatSigned, formatValue, heatColor, percent } from './format';
import { METHOD_TEXT } from './texts';
import type { BoundaryAnswer, PermutationAnswer, TabularAnswer } from './types';
import classes from './Explain.module.css';

const POSITIVE = 'var(--mantine-color-red-6)';
const NEGATIVE = 'var(--mantine-color-blue-6)';

/** A bar from the zero line in the middle, red to the right, blue to the left. */
function SignedBar({
  value,
  scale,
  label,
}: {
  value: number | null;
  scale: number;
  label: string;
}) {
  const share = value === null || scale <= 0 ? 0 : Math.min(1, Math.abs(value) / scale) * 50;
  return (
    <div className={classes.barTrack} role="img" aria-label={label}>
      <div className={classes.zero} style={{ left: '50%' }} />
      <div
        className={classes.bar}
        style={{
          left: (value ?? 0) >= 0 ? '50%' : `${50 - share}%`,
          width: `${share}%`,
          background: (value ?? 0) >= 0 ? POSITIVE : NEGATIVE,
        }}
      />
    </div>
  );
}

interface TabularExplanationProps {
  answer: TabularAnswer;
  classNames: string[] | null;
  onTarget: (index: number | undefined) => void;
}

/** Shapley values (and integrated gradients) of one row as bars around zero. */
export function TabularExplanation({ answer, classNames, onTarget }: TabularExplanationProps) {
  const hasIg = answer.features.some((f) => f.integrated_gradients !== undefined);
  const [method, setMethod] = useState<'shapley' | 'integrated_gradients'>('shapley');
  const shown = hasIg ? method : 'shapley';
  const values = answer.features.map(
    (f) => (shown === 'shapley' ? f.shapley : f.integrated_gradients) ?? 0
  );
  const scale = Math.max(1e-12, ...values.map((v) => Math.abs(v)));
  const prediction = answer.prediction;
  return (
    <Stack gap="sm">
      <Group gap="xs" wrap="wrap">
        <Text size="sm">
          Model:{' '}
          <b>
            {prediction.class !== undefined
              ? `${prediction.class} (${percent(prediction.probability)})`
              : formatValue(prediction.value, 4)}
          </b>
          {prediction.std !== undefined && prediction.std !== null && (
            <> ± {formatValue(prediction.std)} (std of the Gaussian process)</>
          )}
        </Text>
        {answer.sample.label !== undefined && (
          <Badge color={answer.sample.label === prediction.class ? 'teal' : 'red'} variant="light">
            true class: {answer.sample.label}
          </Badge>
        )}
        {answer.sample.value !== undefined && (
          <Badge variant="light">true value: {formatValue(answer.sample.value, 4)}</Badge>
        )}
      </Group>
      <Group gap="md" align="flex-end" wrap="wrap">
        {classNames && answer.target && (
          <Select
            label="Explain the probability of"
            size="xs"
            w={200}
            value={String(answer.target.index)}
            data={classNames.map((name, index) => ({
              value: String(index),
              label: index === prediction.index ? `${name} (predicted)` : name,
            }))}
            onChange={(value) =>
              onTarget(
                value === null || Number(value) === prediction.index ? undefined : Number(value)
              )
            }
            allowDeselect={false}
          />
        )}
        {hasIg && (
          <SegmentedControl
            size="xs"
            value={method}
            onChange={(value) => setMethod(value as typeof method)}
            data={[
              { value: 'shapley', label: 'Shapley values' },
              { value: 'integrated_gradients', label: 'Integrated gradients' },
            ]}
          />
        )}
      </Group>
      <Text size="sm">
        {answer.explains.charAt(0).toUpperCase() + answer.explains.slice(1)}:{' '}
        <b>{formatValue(answer.base_value, 4)}</b> for an average row (the mean of the training
        rows), <b>{formatValue(answer.value, 4)}</b> for this one.
      </Text>
      <Table fz="xs" verticalSpacing={4} withRowBorders={false}>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Feature</Table.Th>
            <Table.Th>Value</Table.Th>
            <Table.Th w="45%">
              <Group justify="space-between" gap={4} wrap="nowrap">
                <Text size="xs" c="blue">
                  lowers it
                </Text>
                <Text size="xs" c="red">
                  raises it
                </Text>
              </Group>
            </Table.Th>
            <Table.Th ta="right">Effect</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {answer.features.map((feature, i) => (
            <Table.Tr key={feature.name}>
              <Table.Td>{feature.name}</Table.Td>
              <Table.Td>{formatValue(feature.value, 4)}</Table.Td>
              <Table.Td>
                <SignedBar
                  value={values[i]}
                  scale={scale}
                  label={`${feature.name}: ${formatSigned(values[i])}`}
                />
              </Table.Td>
              <Table.Td ta="right" ff="monospace">
                {formatSigned(values[i])}
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
      <Text size="xs" c="dimmed">
        {shown === 'shapley' ? METHOD_TEXT.shapley : METHOD_TEXT.integrated}
        {shown === 'shapley' &&
          (answer.exact
            ? ' Computed exactly (all combinations of features).'
            : ` Estimated from ${answer.orders} random orders of the features.`)}
      </Text>
    </Stack>
  );
}

/** Mean drop of the score per feature, with one standard deviation as whiskers. */
export function PermutationView({ answer }: { answer: PermutationAnswer }) {
  const top = Math.max(1e-12, ...answer.features.map((f) => Math.abs(f.mean ?? 0) + (f.std ?? 0)));
  const metric = answer.metric === 'accuracy' ? 'accuracy' : 'R²';
  // Drops are differences of scores: round away the floating-point noise.
  const show = (value: number | null) =>
    value === null
      ? '–'
      : answer.metric === 'accuracy'
        ? `${(value * 100).toFixed(1)} %`
        : Math.abs(value) < 5e-4
          ? '0'
          : value.toFixed(3);
  return (
    <Stack gap="sm">
      <Text size="sm">
        Validation {metric} {show(answer.baseline)} on {answer.rows} rows; drop when a feature is
        shuffled ({answer.repeats} {answer.repeats === 1 ? 'repeat' : 'repeats'}):
      </Text>
      <Table fz="xs" verticalSpacing={4} withRowBorders={false}>
        <Table.Tbody>
          {answer.features.map((feature) => {
            const mean = feature.mean ?? 0;
            const std = feature.std ?? 0;
            const width = (Math.max(0, mean) / top) * 100;
            return (
              <Table.Tr key={feature.name}>
                <Table.Td w={140}>{feature.name}</Table.Td>
                <Table.Td>
                  <div
                    className={classes.barTrack}
                    role="img"
                    aria-label={`${feature.name}: ${show(mean)} ± ${show(std)}`}
                  >
                    <div
                      className={classes.bar}
                      style={{
                        left: 0,
                        width: `${width}%`,
                        background: 'var(--mantine-color-blue-6)',
                      }}
                    />
                    {std > 0 && (
                      <div
                        className={classes.whisker}
                        style={{
                          left: `${(Math.max(0, mean - std) / top) * 100}%`,
                          width: `${((Math.max(0, mean + std) - Math.max(0, mean - std)) / top) * 100}%`,
                        }}
                      />
                    )}
                  </div>
                </Table.Td>
                <Table.Td w={150} ta="right" ff="monospace" style={{ whiteSpace: 'nowrap' }}>
                  {show(mean)} ± {show(std)}
                </Table.Td>
              </Table.Tr>
            );
          })}
        </Table.Tbody>
      </Table>
      <Text size="xs" c="dimmed">
        {METHOD_TEXT.permutation}
      </Text>
    </Stack>
  );
}

const PLOT = { width: 430, height: 420, left: 62, bottom: 36, top: 8, right: 8 };

function ticks(low: number, high: number, count = 5): number[] {
  return Array.from({ length: count }, (_, i) => low + ((high - low) * i) / (count - 1));
}

/** The predictions on a grid over the two features, with the training points. */
export function BoundaryView({ answer }: { answer: BoundaryAnswer }) {
  const [view, setView] = useState<'prediction' | 'std'>('prediction');
  const { width, height, left, bottom, top, right } = PLOT;
  const innerW = width - left - right;
  const innerH = height - top - bottom;
  const [x0, x1] = answer.x;
  const [y0, y1] = answer.y;
  const px = (x: number) => left + ((x - x0) / (x1 - x0)) * innerW;
  const py = (y: number) => top + (1 - (y - y0) / (y1 - y0)) * innerH;
  const classification = Boolean(answer.classes);
  const [v0, v1] = answer.value_range ?? [0, 1];
  const background = view === 'std' && answer.std ? answer.std.png : answer.png;
  // Regression points: coloured like the background (the same scale).
  const pointColor = (label: number) => {
    if (classification) {
      return answer.colors?.[label] ?? 'gray';
    }
    return heatColor(v1 > v0 ? (label - v0) / (v1 - v0) : 0.5);
  };
  return (
    <Stack gap="sm">
      {answer.std && (
        <SegmentedControl
          size="xs"
          value={view}
          onChange={(value) => setView(value as typeof view)}
          data={[
            { value: 'prediction', label: 'Prediction' },
            { value: 'std', label: 'Uncertainty (std)' },
          ]}
          w="fit-content"
        />
      )}
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className={classes.plot}
        role="img"
        aria-label={`Decision boundary over ${answer.features[0]} and ${answer.features[1]}`}
      >
        <image
          href={background}
          x={left}
          y={top}
          width={innerW}
          height={innerH}
          preserveAspectRatio="none"
          style={{ imageRendering: classification ? 'pixelated' : 'auto' }}
        />
        <rect
          x={left}
          y={top}
          width={innerW}
          height={innerH}
          fill="none"
          stroke="currentColor"
          opacity={0.4}
        />
        {answer.points.map(([x, y, label], i) => (
          <circle
            key={i}
            cx={px(x)}
            cy={py(y)}
            r={3.2}
            fill={view === 'std' ? 'white' : pointColor(label)}
            stroke="#1a1b1e"
            strokeWidth={0.8}
          />
        ))}
        {ticks(x0, x1).map((t) => (
          <text
            key={`x${t}`}
            x={px(t)}
            y={height - bottom + 14}
            fontSize={10}
            textAnchor="middle"
            fill="currentColor"
          >
            {formatValue(t, 2)}
          </text>
        ))}
        {ticks(y0, y1).map((t) => (
          <text
            key={`y${t}`}
            x={left - 6}
            y={py(t) + 3}
            fontSize={10}
            textAnchor="end"
            fill="currentColor"
          >
            {formatValue(t, 2)}
          </text>
        ))}
        <text
          x={left + innerW / 2}
          y={height - 4}
          fontSize={11}
          textAnchor="middle"
          fill="currentColor"
        >
          {answer.features[0]}
        </text>
        <text
          x={10}
          y={top + innerH / 2}
          fontSize={11}
          textAnchor="middle"
          fill="currentColor"
          transform={`rotate(-90 10 ${top + innerH / 2})`}
        >
          {answer.features[1]}
        </text>
      </svg>
      {classification ? (
        <Group gap="sm">
          {answer.classes?.map((name, i) => (
            <Group key={name} gap={4}>
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: 10,
                  display: 'inline-block',
                  background: answer.colors?.[i],
                }}
              />
              <Text size="xs">{name}</Text>
            </Group>
          ))}
        </Group>
      ) : (
        <Text size="xs">
          {view === 'std' && answer.std
            ? `std from ${formatValue(answer.std.range[0])} (dark) to ${formatValue(answer.std.range[1])} (bright)`
            : `${answer.target ?? 'Prediction'} from ${formatValue(v0)} (dark) to ${formatValue(v1)} (bright)`}
        </Text>
      )}
      <Paper withBorder p="xs">
        <Text size="xs" c="dimmed">
          {view === 'std' ? METHOD_TEXT.boundaryStd : METHOD_TEXT.boundary} Points: up to{' '}
          {answer.points.length} training rows.
        </Text>
      </Paper>
    </Stack>
  );
}
