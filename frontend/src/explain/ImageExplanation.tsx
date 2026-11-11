import { useState } from 'react';
import {
  Alert,
  Badge,
  Group,
  Paper,
  Select,
  SimpleGrid,
  Slider,
  Stack,
  Text,
  Title,
} from '@mantine/core';
import { formatShape, percent } from './format';
import { METHOD_TEXT } from './texts';
import type { HeatMap, ImageAnswer } from './types';
import classes from './Explain.module.css';

/** The image with a heatmap on top (the heatmap's opacity grows with its value). */
export function Overlay({
  image,
  heat,
  opacity,
  aspect,
  label,
}: {
  image: string;
  heat?: string;
  opacity: number;
  aspect: number;
  label: string;
}) {
  return (
    <div className={classes.overlay} style={{ '--aspect': aspect } as React.CSSProperties}>
      <img src={image} alt={label} />
      {heat && <img src={heat} alt="" style={{ opacity }} />}
    </div>
  );
}

function MapCard({
  title,
  text,
  map,
  image,
  opacity,
  aspect,
  children,
}: {
  title: string;
  text: string;
  map: HeatMap;
  image: string;
  opacity: number;
  aspect: number;
  children?: React.ReactNode;
}) {
  return (
    <Paper withBorder p="xs">
      <Stack gap={6}>
        <Title order={6}>{title}</Title>
        {map.unavailable ? (
          <Text size="sm" c="dimmed">
            {map.unavailable}
          </Text>
        ) : (
          <Overlay image={image} heat={map.png} opacity={opacity} aspect={aspect} label={title} />
        )}
        {children}
        {map.note && (
          <Text size="xs" c="orange">
            {map.note}
          </Text>
        )}
        <Text size="xs" c="dimmed">
          {text}
        </Text>
      </Stack>
    </Paper>
  );
}

interface ImageExplanationProps {
  answer: ImageAnswer;
  classNames: string[];
  onTarget: (index: number | undefined) => void;
  onLayer: (layer: string | undefined) => void;
}

export function ImageExplanation({ answer, classNames, onTarget, onLayer }: ImageExplanationProps) {
  const [opacity, setOpacity] = useState(0.75);
  const [height, width] = answer.size;
  const aspect = width / height;
  const prediction = answer.prediction;
  const truth = answer.sample.label;
  const gradcam = answer.gradcam;
  return (
    <Stack gap="sm">
      <Group gap="xs" wrap="wrap">
        <Text size="sm">
          Model: <b>{prediction.class}</b> ({percent(prediction.probability)})
        </Text>
        {truth !== undefined && (
          <Badge color={truth === prediction.class ? 'teal' : 'red'} variant="light">
            true class: {truth}
          </Badge>
        )}
        {prediction.top && prediction.top.length > 1 && (
          <Text size="xs" c="dimmed">
            then{' '}
            {prediction.top
              .slice(1)
              .map((top) => `${top.class} ${percent(top.probability)}`)
              .join(', ')}
          </Text>
        )}
      </Group>
      <Group gap="md" align="flex-end" wrap="wrap">
        <Select
          label="Explain the class"
          size="xs"
          w={180}
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
          searchable={classNames.length > 10}
        />
        {gradcam.layers && gradcam.layers.length > 1 && (
          <Select
            label="Grad-CAM brick"
            size="xs"
            w={220}
            value={gradcam.layer?.id ?? null}
            data={gradcam.layers.map((layer) => ({
              value: layer.id,
              label: `${layer.name} (${formatShape(layer.shape)})`,
            }))}
            onChange={(value) => onLayer(value ?? undefined)}
            allowDeselect={false}
          />
        )}
        <Stack gap={2} w={160}>
          <Text size="xs">Heatmap opacity</Text>
          <Slider
            value={opacity}
            onChange={setOpacity}
            min={0}
            max={1}
            step={0.05}
            label={(value) => `${Math.round(value * 100)} %`}
            aria-label="Heatmap opacity"
            size="sm"
          />
        </Stack>
      </Group>
      <SimpleGrid cols={{ base: 1, xs: 2, md: 4 }} spacing="xs">
        <Paper withBorder p="xs">
          <Stack gap={6}>
            <Title order={6}>The image</Title>
            <Overlay image={answer.image} opacity={0} aspect={aspect} label="The image" />
            <Text size="xs" c="dimmed">
              {width}×{height} pixels, as the network sees it.
            </Text>
          </Stack>
        </Paper>
        <MapCard
          title="Saliency"
          text={METHOD_TEXT.saliency}
          map={answer.saliency}
          image={answer.image}
          opacity={opacity}
          aspect={aspect}
        />
        <MapCard
          title="Grad-CAM"
          text={METHOD_TEXT.gradcam}
          map={gradcam}
          image={answer.image}
          opacity={opacity}
          aspect={aspect}
        >
          {gradcam.layer && (
            <Text size="xs">
              at {gradcam.layer.name} ({formatShape(gradcam.layer.shape)})
            </Text>
          )}
        </MapCard>
        <MapCard
          title="Occlusion"
          text={METHOD_TEXT.occlusion}
          map={answer.occlusion}
          image={answer.image}
          opacity={opacity}
          aspect={aspect}
        >
          {answer.occlusion.patch && (
            <Text size="xs">
              {answer.occlusion.patch}×{answer.occlusion.patch} patches; largest drop{' '}
              {percent(answer.occlusion.largest_drop)}
            </Text>
          )}
        </MapCard>
      </SimpleGrid>
      {answer.target.index !== prediction.index && (
        <Alert variant="light" color="blue">
          The maps show what speaks for “{answer.target.class}”, not for the predicted class.
        </Alert>
      )}
    </Stack>
  );
}
