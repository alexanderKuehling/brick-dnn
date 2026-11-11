import { useEffect, useMemo, useState } from 'react';
import { IconAlertTriangle, IconRefresh } from '@tabler/icons-react';
import { Alert, Button, Group, Loader, Modal, Stack, Tabs, Text } from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { api, type LayerCatalog, type RunDetail } from '@/api';
import { explainApi, sampleFields } from './api';
import { ImageExplanation } from './ImageExplanation';
import { InsideView } from './InsideView';
import { imageFormat } from './pixels';
import { ImageSamplePicker, TabularSamplePicker } from './SamplePicker';
import { BoundaryView, PermutationView, TabularExplanation } from './TabularViews';
import type {
  ActivationsAnswer,
  BoundaryAnswer,
  ExplainMethodId,
  ExplainMethods,
  ExplainRequest,
  ImageAnswer,
  PermutationAnswer,
  PickedSample,
  SamplesAnswer,
  TabularAnswer,
  WeightsAnswer,
} from './types';
import { useExplanation, type Explanation } from './useExplanation';

type TabId = 'prediction' | 'importance' | 'boundary' | 'inside';

/** Loading, error (with a retry) or the content. */
function Pending<T>({
  result,
  children,
  what,
}: {
  result: Explanation<T>;
  children: (data: T) => React.ReactNode;
  what: string;
}) {
  if (result.error) {
    return (
      <Alert color="red" variant="light" icon={<IconAlertTriangle size={18} />}>
        <Stack gap={6} align="flex-start">
          <Text size="sm">{result.error}</Text>
          <Button
            size="compact-xs"
            variant="light"
            leftSection={<IconRefresh size={14} />}
            onClick={result.retry}
          >
            Try again
          </Button>
        </Stack>
      </Alert>
    );
  }
  if (result.loading || !result.data) {
    return (
      <Group gap="xs">
        <Loader size="sm" />
        <Text size="sm" c="dimmed">
          Computing {what} on the server …
        </Text>
      </Group>
    );
  }
  return (
    <>
      {children(result.data)}
      <Notes notes={(result.data as { notes?: string[] }).notes} />
    </>
  );
}

function Notes({ notes }: { notes?: string[] }) {
  if (!notes || notes.length === 0) {
    return null;
  }
  return (
    <Stack gap={2} mt="xs">
      {notes.map((note) => (
        <Text key={note} size="xs" c="orange">
          {note}
        </Text>
      ))}
    </Stack>
  );
}

interface ExplainModalProps {
  run: RunDetail;
  opened: boolean;
  onClose: () => void;
}

export function ExplainModal({ run, opened, onClose }: ExplainModalProps) {
  const small = useMediaQuery('(max-width: 48em)');
  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title={`Explain “${run.name}”`}
      size="80rem"
      fullScreen={small}
      zIndex={400}
    >
      {opened && <ExplainContent run={run} />}
    </Modal>
  );
}

export function ExplainContent({ run }: { run: RunDetail }) {
  const [methods, setMethods] = useState<ExplainMethods | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [catalog, setCatalog] = useState<LayerCatalog | null>(null);
  const [tab, setTab] = useState<TabId | null>(null);
  const [picked, setPicked] = useState<PickedSample | null>(null);
  const [target, setTarget] = useState<number | undefined>(undefined);
  const [layer, setLayer] = useState<string | undefined>(undefined);

  useEffect(() => {
    const controller = new AbortController();
    explainApi
      .methods(run.id, controller.signal)
      .then(setMethods)
      .catch((exc: Error) => exc.name !== 'AbortError' && setError(exc.message));
    api
      .layers()
      .then(setCatalog)
      .catch(() => undefined);
    return () => controller.abort();
  }, [run.id]);

  const can = (id: ExplainMethodId) =>
    Boolean(methods?.methods.find((m) => m.id === id)?.available);
  const reason = (id: ExplainMethodId) => methods?.methods.find((m) => m.id === id)?.reason ?? null;
  const kind = methods?.kind ?? null;
  const explainMethod: ExplainMethodId | null = can('image')
    ? 'image'
    : can('tabular')
      ? 'tabular'
      : null;
  const tabs: { id: TabId; label: string; enabled: boolean }[] = [
    { id: 'prediction', label: 'Explain a prediction', enabled: explainMethod !== null },
    { id: 'importance', label: 'Feature importance', enabled: can('permutation') },
    { id: 'boundary', label: 'Decision boundary', enabled: can('boundary') },
    { id: 'inside', label: 'Inside the network', enabled: can('activations') || can('weights') },
  ];
  const shownTab = tab ?? tabs.find((t) => t.enabled)?.id ?? null;

  // The validation samples (to pick one), once for the dialog.
  const samples = useExplanation<SamplesAnswer>(
    run.id,
    can('samples') ? { method: 'samples' } : null
  );
  const sample: PickedSample | null =
    picked ??
    (samples.data && samples.data.samples.length > 0
      ? {
          kind: 'validation',
          index: samples.data.samples[0].index,
          label: samples.data.samples[0].label,
        }
      : kind === 'demo' && !can('samples')
        ? { kind: 'random' }
        : null);
  const fields = sampleFields(sample);
  const hasSample = sample !== null;

  const request = (
    method: ExplainMethodId,
    extra: Partial<ExplainRequest> = {}
  ): ExplainRequest => ({
    method,
    ...extra,
  });
  const explanation = useExplanation<ImageAnswer | TabularAnswer>(
    run.id,
    shownTab === 'prediction' && explainMethod && hasSample && sample?.kind !== 'random'
      ? request(explainMethod, {
          ...fields,
          target,
          ...(explainMethod === 'image' ? { layer } : {}),
        })
      : null
  );
  const permutation = useExplanation<PermutationAnswer>(
    run.id,
    shownTab === 'importance' ? request('permutation') : null
  );
  const boundary = useExplanation<BoundaryAnswer>(
    run.id,
    shownTab === 'boundary' ? request('boundary') : null
  );
  const activations = useExplanation<ActivationsAnswer>(
    run.id,
    shownTab === 'inside' && can('activations') && hasSample ? request('activations', fields) : null
  );
  const weights = useExplanation<WeightsAnswer>(
    run.id,
    shownTab === 'inside' && can('weights') ? request('weights') : null
  );

  const descriptions = useMemo(
    () => Object.fromEntries((catalog?.layers ?? []).map((spec) => [spec.type, spec.description])),
    [catalog]
  );
  const classNames = useMemo(() => {
    const plan = run.plan;
    const names =
      (run.state?.preprocessing?.class_names as string[] | undefined) ?? plan?.class_names ?? [];
    const count = plan?.num_classes ?? names.length;
    return Array.from({ length: count }, (_, i) => names[i] ?? `class ${i}`);
  }, [run.plan, run.state?.preprocessing]);

  const pick = (next: PickedSample) => {
    setPicked(next);
    setTarget(undefined);
  };

  if (error) {
    return <Alert color="red">{error}</Alert>;
  }
  if (!methods) {
    return <Loader size="sm" />;
  }
  if (methods.unavailable) {
    return <Alert color="yellow">{methods.unavailable}</Alert>;
  }
  if (!shownTab) {
    return (
      <Alert color="yellow">
        {methods.methods.find((m) => m.reason)?.reason ?? 'Nothing can be explained for this run.'}
      </Alert>
    );
  }

  const picker =
    kind === 'image' ? (
      <ImageSamplePicker
        samples={samples.data}
        loading={samples.loading}
        error={samples.error}
        picked={sample}
        onPick={pick}
        format={imageFormat(run.state?.preprocessing)}
      />
    ) : kind === 'tabular' || (kind === 'demo' && can('samples')) ? (
      <TabularSamplePicker
        samples={samples.data}
        loading={samples.loading}
        error={samples.error}
        picked={sample}
        onPick={pick}
      />
    ) : null;

  return (
    <Stack gap="md">
      {methods.demo && (
        <Alert variant="light" color="yellow">
          Trained on random demo data: the network has learned nothing meaningful, but you can still
          look inside it.
        </Alert>
      )}
      <Tabs value={shownTab} onChange={(value) => setTab(value as TabId)} keepMounted={false}>
        <Tabs.List>
          {tabs.map((t) => (
            <Tabs.Tab
              key={t.id}
              value={t.id}
              disabled={!t.enabled}
              title={t.enabled ? undefined : (tabReason(t.id, reason) ?? undefined)}
            >
              {t.label}
            </Tabs.Tab>
          ))}
        </Tabs.List>

        <Tabs.Panel value="prediction" pt="md">
          <Stack gap="md">
            {picker}
            {explainMethod === 'image' && (
              <Pending
                result={explanation as Explanation<ImageAnswer>}
                what="saliency, Grad-CAM and occlusion"
              >
                {(data) => (
                  <ImageExplanation
                    answer={data}
                    classNames={classNames}
                    onTarget={setTarget}
                    onLayer={setLayer}
                  />
                )}
              </Pending>
            )}
            {explainMethod === 'tabular' && hasSample && (
              <Pending result={explanation as Explanation<TabularAnswer>} what="the attributions">
                {(data) => (
                  <TabularExplanation
                    answer={data}
                    classNames={classNames.length > 0 ? classNames : null}
                    onTarget={setTarget}
                  />
                )}
              </Pending>
            )}
          </Stack>
        </Tabs.Panel>

        <Tabs.Panel value="importance" pt="md">
          <Pending result={permutation} what="the permutation importance">
            {(data) => <PermutationView answer={data} />}
          </Pending>
        </Tabs.Panel>

        <Tabs.Panel value="boundary" pt="md">
          <Pending result={boundary} what="the decision boundary">
            {(data) => <BoundaryView answer={data} />}
          </Pending>
        </Tabs.Panel>

        <Tabs.Panel value="inside" pt="md">
          <Stack gap="md">
            {can('activations') && picker}
            {can('activations') && sample?.kind === 'random' && (
              <Text size="sm" c="dimmed">
                A random input (the run trained on demo data, which is not kept).
              </Text>
            )}
            {can('activations') && hasSample && (
              <Pending result={activations} what="the output of every brick">
                {(data) => (
                  <InsideView activations={data} weights={null} descriptions={descriptions} />
                )}
              </Pending>
            )}
            {can('weights') && (
              <Pending result={weights} what="the weights">
                {(data) => (
                  <InsideView activations={null} weights={data} descriptions={descriptions} />
                )}
              </Pending>
            )}
          </Stack>
        </Tabs.Panel>
      </Tabs>
      <Text size="xs" c="dimmed">
        Explanations run on the server like predictions and count as predictions in your compute
        usage; asking the same again is free. See the documentation for what each method can and
        cannot tell.
      </Text>
    </Stack>
  );
}

function tabReason(id: TabId, reason: (id: ExplainMethodId) => string | null): string | null {
  if (id === 'prediction') {
    return reason('image') ?? reason('tabular');
  }
  if (id === 'importance') {
    return reason('permutation');
  }
  if (id === 'boundary') {
    return reason('boundary');
  }
  return reason('activations');
}
