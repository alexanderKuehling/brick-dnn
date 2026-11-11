import { useMemo } from 'react';
import { Code, Stack, Text } from '@mantine/core';
import type { Analysis } from '@/api/types';
import { coversInput, describeField, paramFormula, receptiveFields, type Link } from './learning';

/**
 * For learners, in the inspector of a brick: how its parameters add up and how much of the
 * input one of its output values "sees" (convolution stacks).
 */
export function LayerLearning({
  nodeId,
  analysis,
  links,
}: {
  nodeId: string;
  analysis: Analysis | null;
  links: Link[] | undefined;
}) {
  const info = analysis?.nodes[nodeId];
  const formula = paramFormula(info);
  const fields = useMemo(
    () => (analysis && links ? receptiveFields(analysis, links) : {}),
    [analysis, links]
  );
  const field = info && info.type !== 'Input' ? fields[nodeId] : undefined;
  if (!formula && !field) {
    return null;
  }
  return (
    <Stack gap={6}>
      {formula && (
        <div>
          <Text size="xs" c="dimmed">
            How the parameters add up
          </Text>
          <Code aria-label="Parameter formula">{formula.formula}</Code>
          <Text size="xs" c="dimmed" mt={2}>
            {formula.parts}.
          </Text>
        </div>
      )}
      {field && (
        <div>
          <Text size="xs" c="dimmed">
            Receptive field
          </Text>
          <Text size="sm" fw={600}>
            {field.approximate ? 'about ' : ''}
            {describeField(field)}
            {coversInput(field) && ' (the whole input)'}
          </Text>
          <Text size="xs" c="dimmed">
            Each output value depends on {field.approximate ? 'about ' : ''}this much of the input{' '}
            {field.input.join('×')}
            {field.jump.some((j) => j !== 1)
              ? `; neighbouring outputs are ${field.jump.map((j) => Number(j.toFixed(2))).join('×')} input ${field.size.length === 2 ? 'pixels' : 'steps'} apart`
              : ''}
            . Deeper convolutions and pooling see larger parts.
          </Text>
        </div>
      )}
    </Stack>
  );
}
