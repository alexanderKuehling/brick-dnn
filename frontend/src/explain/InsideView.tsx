import { useState } from 'react';
import {
  Badge,
  Box,
  Group,
  Paper,
  ScrollArea,
  SimpleGrid,
  Stack,
  Text,
  Title,
  UnstyledButton,
} from '@mantine/core';
import { MiniHistogram } from '@/components/charts/MiniHistogram';
import { formatShape, formatValue, percent } from './format';
import { METHOD_TEXT } from './texts';
import type {
  ActivationsAnswer,
  ActivationView,
  Histogram,
  NodeActivation,
  WeightsAnswer,
} from './types';
import classes from './Explain.module.css';

function HistogramLine({ histogram, label }: { histogram: Histogram; label: string }) {
  if (histogram.counts.length === 0) {
    return null;
  }
  return (
    <Group gap={6} wrap="nowrap" align="flex-end">
      <Text size="10px" c="dimmed" ff="monospace">
        {formatValue(histogram.min)}
      </Text>
      <MiniHistogram counts={histogram.counts} width={160} height={28} label={label} />
      <Text size="10px" c="dimmed" ff="monospace">
        {formatValue(histogram.max)}
      </Text>
    </Group>
  );
}

function VectorBars({ values, total }: { values: (number | null)[]; total: number }) {
  const top = Math.max(1e-12, ...values.map((v) => Math.abs(v ?? 0)));
  const width = 360;
  const height = 60;
  const bar = width / values.length;
  return (
    <Stack gap={4}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className={classes.plot}
        style={{ maxWidth: width }}
        role="img"
        aria-label={`${values.length} values from ${formatValue(Math.min(...values.map((v) => v ?? 0)))} to ${formatValue(Math.max(...values.map((v) => v ?? 0)))}`}
      >
        <line
          x1={0}
          x2={width}
          y1={height / 2}
          y2={height / 2}
          stroke="currentColor"
          opacity={0.3}
        />
        {values.map((value, i) => {
          const h = (Math.abs(value ?? 0) / top) * (height / 2 - 1);
          return (
            <rect
              key={i}
              x={i * bar + (bar > 3 ? 0.5 : 0)}
              width={Math.max(0.5, bar - (bar > 3 ? 1 : 0))}
              y={(value ?? 0) >= 0 ? height / 2 - h : height / 2}
              height={h}
              fill={
                (value ?? 0) >= 0 ? 'var(--mantine-color-red-6)' : 'var(--mantine-color-blue-6)'
              }
            />
          );
        })}
      </svg>
      {total > values.length && (
        <Text size="xs" c="dimmed">
          The first {values.length} of {total} values.
        </Text>
      )}
    </Stack>
  );
}

function ActivationPicture({ view }: { view: ActivationView }) {
  if (view.kind === 'maps') {
    return (
      <Stack gap={4}>
        <img src={view.png} alt="Feature maps" className={classes.maps} />
        <Text size="xs" c="dimmed">
          {view.shown === view.channels
            ? `All ${view.channels} feature maps`
            : `The first ${view.shown} of ${view.channels} feature maps`}{' '}
          ({view.tile[0]}×{view.tile[1]} each; bright = high, each map scaled on its own).
        </Text>
      </Stack>
    );
  }
  if (view.kind === 'matrix' || view.kind === 'image') {
    return (
      <Stack gap={4}>
        <img src={view.png} alt="Heatmap of the output" className={classes.maps} />
        <Text size="xs" c="dimmed">
          {view.kind === 'matrix'
            ? `${view.rows} rows (positions) × ${view.cols} columns (features)`
            : `${view.rows}×${view.cols}`}
          {view.shown[0] < view.rows || view.shown[1] < view.cols
            ? `, the first ${view.shown[0]}×${view.shown[1]} shown`
            : ''}
          ; bright = high.
        </Text>
      </Stack>
    );
  }
  if (view.kind === 'vector') {
    return <VectorBars values={view.values} total={view.total} />;
  }
  if (view.kind === 'scalar') {
    return (
      <Text size="xl" fw={600} ff="monospace">
        {formatValue(view.value, 5)}
      </Text>
    );
  }
  return null;
}

function NodeDetails({ node, description }: { node: NodeActivation; description?: string }) {
  return (
    <Stack gap="xs">
      <Group gap="xs">
        <Title order={5}>{node.name}</Title>
        {node.name !== node.type && <Badge variant="light">{node.type}</Badge>}
        <Badge variant="outline" color="gray">
          {formatShape(node.shape)}
        </Badge>
      </Group>
      {description && (
        <Text size="sm" c="dimmed">
          {description}
        </Text>
      )}
      <ActivationPicture view={node.view} />
      <Group gap="lg">
        <Text size="xs">
          mean <b>{formatValue(node.stats.mean)}</b>
        </Text>
        <Text size="xs">
          std <b>{formatValue(node.stats.std)}</b>
        </Text>
        <Text size="xs">
          zeros <b>{percent(node.stats.zeros)}</b>
        </Text>
      </Group>
      <HistogramLine
        histogram={node.histogram}
        label={`Distribution of the outputs of ${node.name}`}
      />
    </Stack>
  );
}

interface InsideProps {
  activations: ActivationsAnswer | null;
  weights: WeightsAnswer | null;
  descriptions: Record<string, string>;
}

/** Every brick's output for the sample (click a brick), then the learned weights. */
export function InsideView({ activations, weights, descriptions }: InsideProps) {
  const nodes = activations?.nodes ?? [];
  const [selected, setSelected] = useState<string | null>(null);
  const current =
    nodes.find((node) => node.id === selected) ?? nodes[Math.min(1, nodes.length - 1)];
  return (
    <Stack gap="md">
      {activations && (
        <>
          {activations.prediction?.class !== undefined && (
            <Text size="sm">
              Model: <b>{activations.prediction.class}</b> (
              {percent(activations.prediction.probability)})
              {activations.sample.label !== undefined && (
                <> · true class: {activations.sample.label}</>
              )}
            </Text>
          )}
          <SimpleGrid
            cols={{ base: 1, sm: 2 }}
            spacing="md"
            style={{ gridTemplateColumns: undefined }}
          >
            <Box>
              <Text size="xs" c="dimmed" mb={4}>
                The bricks in the order the data flows through them:
              </Text>
              <ScrollArea.Autosize mah={420} type="auto">
                <Stack gap={2}>
                  {nodes.map((node, index) => (
                    <UnstyledButton
                      key={node.id}
                      className={classes.brick}
                      data-active={node.id === current?.id || undefined}
                      onClick={() => setSelected(node.id)}
                      aria-pressed={node.id === current?.id}
                    >
                      <Group justify="space-between" wrap="nowrap" gap="xs">
                        <Text size="sm" truncate>
                          {index + 1}. {node.name}
                        </Text>
                        <Text size="xs" c="dimmed" ff="monospace" style={{ whiteSpace: 'nowrap' }}>
                          {formatShape(node.shape)}
                        </Text>
                      </Group>
                    </UnstyledButton>
                  ))}
                </Stack>
              </ScrollArea.Autosize>
            </Box>
            {current && <NodeDetails node={current} description={descriptions[current.type]} />}
          </SimpleGrid>
          <Text size="xs" c="dimmed">
            {METHOD_TEXT.activations}
          </Text>
        </>
      )}
      {weights && (
        <Stack gap="sm">
          <Title order={5}>Learned weights</Title>
          {weights.filters && (
            <Paper withBorder p="xs">
              <Stack gap={4}>
                <Text size="sm" fw={600}>
                  Filters of {weights.filters.name} ({weights.filters.shape[2]}×
                  {weights.filters.shape[3]}, {weights.filters.count} of {weights.filters.shape[0]})
                </Text>
                <img src={weights.filters.png} alt="Learned filters" className={classes.filters} />
                {weights.filters.note && (
                  <Text size="xs" c="dimmed">
                    {weights.filters.note}
                  </Text>
                )}
                <Text size="xs" c="dimmed">
                  {METHOD_TEXT.filters}
                </Text>
              </Stack>
            </Paper>
          )}
          <SimpleGrid cols={{ base: 1, xs: 2, md: 3 }} spacing="xs">
            {weights.layers.flatMap((layer) =>
              layer.params.map((param) => (
                <Paper key={`${layer.id}.${param.name}`} withBorder p="xs">
                  <Text size="xs" fw={600} truncate>
                    {layer.name}.{param.name}{' '}
                    <Text span size="xs" c="dimmed" ff="monospace">
                      {formatShape(param.shape)}
                    </Text>
                  </Text>
                  <HistogramLine
                    histogram={param.histogram}
                    label={`Distribution of ${layer.name}.${param.name}`}
                  />
                  <Text size="10px" c="dimmed">
                    mean {formatValue(param.histogram.mean)} · std{' '}
                    {formatValue(param.histogram.std)} · norm {formatValue(param.norm)}
                  </Text>
                </Paper>
              ))
            )}
          </SimpleGrid>
          <Text size="xs" c="dimmed">
            {METHOD_TEXT.weights}
          </Text>
        </Stack>
      )}
    </Stack>
  );
}
