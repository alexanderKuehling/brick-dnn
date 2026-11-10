import { useEffect, useMemo, useState } from 'react';
import {
  IconArrowRight,
  IconChevronLeft,
  IconChevronRight,
  IconPlayerPause,
  IconPlayerPlay,
  IconRoute,
  IconX,
} from '@tabler/icons-react';
import { Panel } from '@xyflow/react';
import { ActionIcon, Button, Group, Paper, Stack, Text, Tooltip } from '@mantine/core';
import type { Analysis, LayerSpec } from '@/api/types';
import { formatShape } from './format';
import { flowSteps, paramFormula, shapeChange, type FlowStep, type Link } from './learning';

const STEP_MS = 2400;

/** A small picture of a tensor: stacked maps (C×H×W), a grid (L×F) or a row (F). */
export function TensorGlyph({ shape }: { shape: number[] | null }) {
  const size = 56;
  if (!shape || shape.length === 0) {
    return <svg width={size} height={size} aria-hidden />;
  }
  const scale = (value: number, max: number) =>
    Math.max(6, Math.min(max, 8 * Math.log2(value + 1)));
  let body: React.ReactNode;
  if (shape.length >= 3) {
    const [c, h, w] = shape.slice(-3);
    const layers = Math.min(6, Math.max(1, Math.round(Math.log2(c + 1))));
    const height = scale(h, 38);
    const width = scale(w, 38);
    body = Array.from({ length: layers }, (_, i) => (
      <rect
        key={i}
        x={4 + (layers - 1 - i) * 3}
        y={4 + i * 3}
        width={width}
        height={height}
        rx={1.5}
        fill="var(--mantine-color-blue-filled)"
        fillOpacity={0.25 + (0.6 * (i + 1)) / layers}
        stroke="var(--mantine-color-body)"
        strokeWidth={0.8}
      />
    ));
  } else if (shape.length === 2) {
    const [rows, cols] = shape;
    body = (
      <rect
        x={4}
        y={4}
        width={scale(cols, 48)}
        height={scale(rows, 48)}
        rx={1.5}
        fill="var(--mantine-color-grape-filled)"
        fillOpacity={0.6}
      />
    );
  } else {
    const cells = Math.min(10, Math.max(1, Math.round(Math.log2(shape[0] + 1))));
    body = Array.from({ length: cells }, (_, i) => (
      <rect
        key={i}
        x={4 + i * 5}
        y={22}
        width={4}
        height={12}
        rx={1}
        fill="var(--mantine-color-teal-filled)"
        fillOpacity={0.75}
      />
    ));
  }
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden>
      {body}
    </svg>
  );
}

function StepCard({
  step,
  spec,
  analysis,
}: {
  step: FlowStep;
  spec?: LayerSpec;
  analysis: Analysis;
}) {
  const formula = paramFormula(analysis.nodes[step.id]);
  const change = shapeChange(step);
  return (
    <Stack gap={6}>
      <Group gap="xs" wrap="nowrap" align="center">
        {step.type !== 'Input' && (
          <>
            <Stack gap={0} align="center">
              <TensorGlyph shape={step.inputs[0] ?? null} />
              <Text size="10px" ff="monospace">
                {formatShape(step.inputs[0])}
              </Text>
            </Stack>
            <IconArrowRight size={16} />
          </>
        )}
        <Stack gap={0} align="center">
          <TensorGlyph shape={step.output} />
          <Text size="10px" ff="monospace">
            {formatShape(step.output)}
          </Text>
        </Stack>
        <Stack gap={2} style={{ minWidth: 0, flex: 1 }}>
          <Text size="sm" fw={600}>
            {spec?.label ?? step.type}
          </Text>
          {change && (
            <Text size="xs" c="dimmed">
              {change}
            </Text>
          )}
        </Stack>
      </Group>
      <Text size="xs">{spec?.description}</Text>
      {formula ? (
        <Text size="xs" ff="monospace">
          {formula.formula} parameters
        </Text>
      ) : (
        step.params !== null &&
        step.params > 0 && (
          <Text size="xs" ff="monospace">
            {step.params.toLocaleString('en-US')} parameters
          </Text>
        )
      )}
    </Stack>
  );
}

interface DataFlowTourProps {
  analysis: Analysis | null;
  specs: Record<string, LayerSpec>;
  links: Link[];
  focusNode: (id: string) => void;
}

/**
 * "How data flows": steps a sample (one tensor, batch dimension left out) through the
 * network brick by brick, with the shape before and after and what each brick computes.
 * Shown on the canvas; selecting a step centres its brick. Needs no server.
 */
export function DataFlowTour({ analysis, specs, links, focusNode }: DataFlowTourProps) {
  const [open, setOpen] = useState(false);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const steps = useMemo(
    () => (analysis?.valid && analysis.family !== 'classical' ? flowSteps(analysis, links) : []),
    [analysis, links]
  );
  const current = steps[Math.min(index, steps.length - 1)];

  useEffect(() => {
    if (!playing) {
      return undefined;
    }
    const timer = window.setTimeout(() => {
      if (index + 1 >= steps.length) {
        setPlaying(false);
      } else {
        setIndex(index + 1);
      }
    }, STEP_MS);
    return () => window.clearTimeout(timer);
  }, [playing, index, steps.length]);

  useEffect(() => {
    if (open && current) {
      focusNode(current.id);
    }
    // Only when the step changes (focusNode changes with every edit of the network).
  }, [open, current?.id]);

  if (steps.length < 2 || !analysis) {
    return null;
  }
  if (!open) {
    return (
      <Panel position="bottom-center">
        <Tooltip label="Follow one sample through the network, brick by brick">
          <Button
            size="xs"
            variant="default"
            leftSection={<IconRoute size={14} />}
            onClick={() => {
              setIndex(0);
              setOpen(true);
            }}
          >
            How data flows
          </Button>
        </Tooltip>
      </Panel>
    );
  }
  const go = (next: number) => {
    setPlaying(false);
    setIndex(Math.max(0, Math.min(steps.length - 1, next)));
  };
  return (
    <Panel position="bottom-center">
      <Paper
        withBorder
        shadow="md"
        p="sm"
        w={{ base: 'calc(100vw - 32px)', xs: 420 }}
        role="region"
        aria-label="How data flows"
      >
        <Stack gap="xs">
          <Group justify="space-between" wrap="nowrap">
            <Text size="xs" c="dimmed">
              Step {index + 1} of {steps.length} · one sample, without the batch dimension
            </Text>
            <ActionIcon
              variant="subtle"
              color="gray"
              size="sm"
              aria-label="Close"
              onClick={() => setOpen(false)}
            >
              <IconX size={14} />
            </ActionIcon>
          </Group>
          {current && <StepCard step={current} spec={specs[current.type]} analysis={analysis} />}
          <Group justify="space-between">
            <Button
              size="compact-sm"
              variant="default"
              leftSection={<IconChevronLeft size={14} />}
              disabled={index === 0}
              onClick={() => go(index - 1)}
            >
              Back
            </Button>
            <ActionIcon
              variant="light"
              aria-label={playing ? 'Pause' : 'Play'}
              onClick={() => {
                if (!playing && index + 1 >= steps.length) {
                  setIndex(0);
                }
                setPlaying(!playing);
              }}
            >
              {playing ? <IconPlayerPause size={16} /> : <IconPlayerPlay size={16} />}
            </ActionIcon>
            <Button
              size="compact-sm"
              rightSection={<IconChevronRight size={14} />}
              disabled={index + 1 >= steps.length}
              onClick={() => go(index + 1)}
            >
              Next
            </Button>
          </Group>
        </Stack>
      </Paper>
    </Panel>
  );
}
