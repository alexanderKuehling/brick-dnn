import { useEffect, useState } from 'react';
import { Alert, Loader, Paper, Stack, Text } from '@mantine/core';
import { api, type PredictionForm } from '@/api';
import { slugify } from '@/builder/graph';
import { i18n, useTranslation } from '@/i18n';
import { formatThreshold } from '../threshold';
import { ImagePredict } from './ImagePredict';
import { TabularPredict } from './TabularPredict';

/**
 * Try a trained run: predict a row typed in, pasted CSV rows, an uploaded or drawn image.
 * Loads what the run takes when it is shown (GET …/predict-form/). A binary classification
 * says which decision threshold its predictions use.
 */
export function PredictionPanel({
  runId,
  runName,
  decisionThreshold,
  positiveClass,
  ensemble = false,
}: {
  runId: string;
  runName: string;
  /** An ensemble: it can show what each member predicts too. */
  ensemble?: boolean;
  /** Binary classification: the run's decision threshold (null for other runs). */
  decisionThreshold?: number | null;
  /** The class the threshold decides for. */
  positiveClass?: string;
}) {
  useTranslation();
  const [form, setForm] = useState<PredictionForm | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setForm(null);
    setError(null);
    api.training.runs
      .predictForm(runId, controller.signal)
      .then(setForm)
      .catch((reason: Error) => {
        if (reason.name !== 'AbortError') {
          setError(reason.message);
        }
      });
    return () => controller.abort();
  }, [runId]);

  let body: React.ReactNode;
  if (error) {
    body = (
      <Text size="sm" c="red">
        {error}
      </Text>
    );
  } else if (!form) {
    body = <Loader size="sm" aria-label={i18n.t('ui.training.PredictionPanel.loading')} />;
  } else if (!form.available) {
    body = (
      <Text size="sm" c="dimmed">
        {form.reason ?? i18n.t('ui.training.PredictionPanel.thisRunCannotPredict')}
      </Text>
    );
  } else if (form.kind === 'demo') {
    body = (
      <Text size="sm" c="dimmed">
        {i18n.t('ui.training.PredictionPanel.thisRunWasTrainedOn')}
      </Text>
    );
  } else if (form.kind === 'image' && form.image) {
    body = <ImagePredict runId={runId} form={form} />;
  } else if (form.kind === 'tabular' && form.prepared) {
    body = (
      <TabularPredict
        runId={runId}
        form={form}
        fileBase={slugify(runName) || 'run'}
        ensemble={ensemble}
      />
    );
  } else {
    body = (
      <Text size="sm" c="dimmed">
        {i18n.t('ui.training.PredictionPanel.predictionsForThisKindOf')}
      </Text>
    );
  }

  return (
    <Paper withBorder p="sm" radius="sm">
      <Stack gap="sm">
        {form?.budget_note && (
          <Alert
            color="yellow"
            variant="light"
            p="xs"
            title={i18n.t('ui.training.PredictionPanel.noComputeLeft')}
          >
            <Text size="sm">{form.budget_note}</Text>
            <Text size="xs" c="dimmed">
              {i18n.t('ui.training.PredictionPanel.predictionsUseComputeCpuTime')}
            </Text>
          </Alert>
        )}
        {typeof decisionThreshold === 'number' && form?.available && (
          <Text size="xs" c="dimmed" data-testid="prediction-threshold">
            {i18n.t('ui.training.PredictionPanel.decisionThresholdThePredictionIs', {
              decisionThreshold: formatThreshold(decisionThreshold),
              value: positiveClass ?? i18n.t('ui.training.PredictionPanel.theSecondClass'),
              decisionThreshold2: formatThreshold(decisionThreshold),
            })}
          </Text>
        )}
        {form?.raw_note && (
          <Text size="xs" c="dimmed">
            {form.raw_note}
          </Text>
        )}
        {body}
      </Stack>
    </Paper>
  );
}
