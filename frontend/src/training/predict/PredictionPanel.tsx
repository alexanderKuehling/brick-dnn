import { useEffect, useState } from 'react';
import { Alert, Loader, Paper, Stack, Text } from '@mantine/core';
import { api, type PredictionForm } from '@/api';
import { slugify } from '@/builder/graph';
import { ImagePredict } from './ImagePredict';
import { TabularPredict } from './TabularPredict';

/**
 * Try a trained run: predict a row typed in, pasted CSV rows, an uploaded or drawn image.
 * Loads what the run takes when it is shown (GET …/predict-form/).
 */
export function PredictionPanel({ runId, runName }: { runId: string; runName: string }) {
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
    body = <Loader size="sm" aria-label="Loading" />;
  } else if (!form.available) {
    body = (
      <Text size="sm" c="dimmed">
        {form.reason ?? 'This run cannot predict.'}
      </Text>
    );
  } else if (form.kind === 'demo') {
    body = (
      <Text size="sm" c="dimmed">
        This run was trained on random demo data, so its predictions mean nothing. Train it on a
        dataset to try it here.
      </Text>
    );
  } else if (form.kind === 'image' && form.image) {
    body = <ImagePredict runId={runId} form={form} />;
  } else if (form.kind === 'tabular' && form.prepared) {
    body = <TabularPredict runId={runId} form={form} fileBase={slugify(runName) || 'run'} />;
  } else {
    body = (
      <Text size="sm" c="dimmed">
        Predictions for this kind of run are not available in the web app.
      </Text>
    );
  }

  return (
    <Paper withBorder p="sm" radius="sm">
      <Stack gap="sm">
        {form?.budget_note && (
          <Alert color="yellow" variant="light" p="xs" title="No compute left">
            <Text size="sm">{form.budget_note}</Text>
            <Text size="xs" c="dimmed">
              Predictions use compute (CPU time) too, so they are not possible until it resets.
            </Text>
          </Alert>
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
