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
import { i18n, Trans, useTranslation } from '@/i18n';
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
  useTranslation();
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
        aria-label={i18n.t('ui.explain.InsideView.valuesFromTo', {
          length: values.length,
          min: formatValue(Math.min(...values.map((v) => v ?? 0))),
          max: formatValue(Math.max(...values.map((v) => v ?? 0))),
        })}
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
          {i18n.t('ui.explain.InsideView.theFirstOfValues', { length: values.length, total })}
        </Text>
      )}
    </Stack>
  );
}

function ActivationPicture({ view }: { view: ActivationView }) {
  useTranslation();
  if (view.kind === 'maps') {
    return (
      <Stack gap={4}>
        <img
          src={view.png}
          alt={i18n.t('ui.explain.InsideView.featureMaps')}
          className={classes.maps}
        />
        <Text size="xs" c="dimmed">
          {i18n.t('ui.explain.InsideView.eachBrightHighEachMap', {
            value:
              view.shown === view.channels
                ? i18n.t('ui.explain.InsideView.allFeatureMaps', { channels: view.channels })
                : i18n.t('ui.explain.InsideView.theFirstOfFeatureMaps', {
                    shown: view.shown,
                    channels: view.channels,
                  }),
            value2: view.tile[0],
            value3: view.tile[1],
          })}
        </Text>
      </Stack>
    );
  }
  if (view.kind === 'matrix' || view.kind === 'image') {
    return (
      <Stack gap={4}>
        <img
          src={view.png}
          alt={i18n.t('ui.explain.InsideView.heatmapOfTheOutput')}
          className={classes.maps}
        />
        <Text size="xs" c="dimmed">
          {i18n.t('ui.explain.InsideView.brightHigh', {
            value:
              view.kind === 'matrix'
                ? i18n.t('ui.explain.InsideView.rowsPositionsColumnsFeatures', {
                    rows: view.rows,
                    cols: view.cols,
                  })
                : `${view.rows}×${view.cols}`,
            value2:
              view.shown[0] < view.rows || view.shown[1] < view.cols
                ? i18n.t('ui.explain.InsideView.theFirstShown', {
                    value: view.shown[0],
                    value2: view.shown[1],
                  })
                : '',
          })}
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
  useTranslation();
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
          <Trans
            i18nKey="ui.explain.InsideView.mean"
            values={{ mean: formatValue(node.stats.mean) }}
            components={[<b />]}
          />
        </Text>
        <Text size="xs">
          <Trans
            i18nKey="ui.explain.InsideView.std"
            values={{ std: formatValue(node.stats.std) }}
            components={[<b />]}
          />
        </Text>
        <Text size="xs">
          <Trans
            i18nKey="ui.explain.InsideView.zeros"
            values={{ percent: percent(node.stats.zeros) }}
            components={[<b />]}
          />
        </Text>
      </Group>
      <HistogramLine
        histogram={node.histogram}
        label={i18n.t('ui.explain.InsideView.distributionOfTheOutputsOf', { name: node.name })}
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
  useTranslation();
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
              {i18n.t('ui.explain.InsideView.model')} <b>{activations.prediction.class}</b> (
              {percent(activations.prediction.probability)})
              {activations.sample.label !== undefined && (
                <>
                  {' '}
                  {i18n.t('ui.explain.InsideView.trueClass', { label: activations.sample.label })}
                </>
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
                {i18n.t('ui.explain.InsideView.theBricksInTheOrder')}
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
          <Title order={5}>{i18n.t('ui.explain.InsideView.learnedWeights')}</Title>
          {weights.filters && (
            <Paper withBorder p="xs">
              <Stack gap={4}>
                <Text size="sm" fw={600}>
                  {i18n.t('ui.explain.InsideView.filtersOfOf', {
                    name: weights.filters.name,
                    value: weights.filters.shape[2],
                    value2: weights.filters.shape[3],
                    count: weights.filters.count,
                    value3: weights.filters.shape[0],
                  })}
                </Text>
                <img
                  src={weights.filters.png}
                  alt={i18n.t('ui.explain.InsideView.learnedFilters')}
                  className={classes.filters}
                />
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
                    label={i18n.t('ui.explain.InsideView.distributionOf', {
                      name: layer.name,
                      name2: param.name,
                    })}
                  />
                  <Text size="10px" c="dimmed">
                    {i18n.t('ui.explain.InsideView.meanStdNorm', {
                      mean: formatValue(param.histogram.mean),
                      std: formatValue(param.histogram.std),
                      norm: formatValue(param.norm),
                    })}
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
