import { useMemo } from 'react';
import { Code, Stack, Text } from '@mantine/core';
import type { Analysis } from '@/api/types';
import { i18n, useTranslation } from '@/i18n';
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
  useTranslation();
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
            {i18n.t('ui.explain.LayerLearning.howTheParametersAddUp')}
          </Text>
          <Code aria-label={i18n.t('ui.explain.LayerLearning.parameterFormula')}>
            {formula.formula}
          </Code>
          <Text size="xs" c="dimmed" mt={2}>
            {formula.parts}.
          </Text>
        </div>
      )}
      {field && (
        <div>
          <Text size="xs" c="dimmed">
            {i18n.t('ui.explain.LayerLearning.receptiveField')}
          </Text>
          <Text size="sm" fw={600}>
            {field.approximate ? i18n.t('ui.explain.LayerLearning.about') : ''}
            {describeField(field)}
            {coversInput(field) && i18n.t('ui.explain.LayerLearning.theWholeInput')}
          </Text>
          <Text size="xs" c="dimmed">
            {i18n.t('ui.explain.LayerLearning.eachOutputValueDependsOn', {
              value: field.approximate ? i18n.t('ui.explain.LayerLearning.about') : '',
              input: field.input.join('×'),
              value2: field.jump.some((j) => j !== 1)
                ? i18n.t('ui.explain.LayerLearning.neighbouringOutputsAreInputApart', {
                    map: field.jump.map((j) => Number(j.toFixed(2))).join('×'),
                    value: field.size.length === 2 ? 'pixels' : 'steps',
                  })
                : '',
            })}
          </Text>
        </div>
      )}
    </Stack>
  );
}
