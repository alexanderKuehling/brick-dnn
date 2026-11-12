/**
 * "Why the model decides": previews of the explanations right in the run view, so they are
 * not hidden behind the Explain button. Tables: the permutation importance (and the decision
 * boundary for two features); images: Grad-CAM and saliency of a few validation images.
 * Each opens the full dialog on its tab. Computing them costs a few seconds of server CPU and
 * counts as a prediction, so they load on request; runs whose explanations were shown in this
 * browser before load them at once (the server has them cached, which costs nothing).
 */
import { useEffect, useId, useState } from 'react';
import { IconArrowUpRight, IconBulb, IconRefresh, IconSparkles } from '@tabler/icons-react';
import {
  Button,
  Group,
  Loader,
  Paper,
  SegmentedControl,
  Stack,
  Text,
  ThemeIcon,
  UnstyledButton,
} from '@mantine/core';
import type { RunDetail } from '@/api';
import { explainApi } from './api';
import { ExplainDialog } from './ExplainButton';
import type { ExplainStart } from './ExplainModal';
import { percent } from './format';
import { Overlay } from './ImageExplanation';
import { Pending } from './Pending';
import { BoundaryView, PermutationView } from './TabularViews';
import type {
  BoundaryAnswer,
  ExplainMethodId,
  ExplainMethods,
  ExplainTab,
  ImageAnswer,
  ListedSample,
  PermutationAnswer,
  SamplesAnswer,
} from './types';
import { useExplanation, type Explanation } from './useExplanation';
import classes from './RunExplanations.module.css';

/** Runs whose explanations were shown in this browser (newest first). */
export const EXPLAINED_RUNS_KEY = 'brickdnn:explained-runs';
const REMEMBERED_RUNS = 50;
/** Features shown before "Show all". */
export const TOP_FEATURES = 10;
/** Validation images in the strip. */
const IMAGES = 3;

function wasExplained(runId: string): boolean {
  try {
    const list: unknown = JSON.parse(localStorage.getItem(EXPLAINED_RUNS_KEY) ?? '[]');
    return Array.isArray(list) && list.includes(runId);
  } catch {
    return false;
  }
}

function rememberExplained(runId: string): void {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(EXPLAINED_RUNS_KEY) ?? '[]');
    const list = Array.isArray(raw)
      ? raw.filter((id) => typeof id === 'string' && id !== runId)
      : [];
    localStorage.setItem(
      EXPLAINED_RUNS_KEY,
      JSON.stringify([runId, ...list].slice(0, REMEMBERED_RUNS))
    );
  } catch {
    // Storage unavailable: the previews just wait for the button next time.
  }
}

/** A few validation samples: right ones and the first mistake (often the most telling). */
export function pickSamples(samples: ListedSample[], count = IMAGES): ListedSample[] {
  const wrong = samples.find((sample) => sample.correct === false);
  const right = samples.filter((sample) => sample !== wrong).slice(0, wrong ? count - 1 : count);
  return wrong ? [...right, wrong] : right;
}

/** The section for a finished run with weights (the caller checks that). */
export function RunExplanations({ run }: { run: RunDetail }) {
  // A new run starts from scratch (its own methods, answers and button).
  return <Explanations key={run.id} run={run} />;
}

function Explanations({ run }: { run: RunDetail }) {
  const titleId = useId();
  const [methods, setMethods] = useState<ExplainMethods | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [requested, setRequested] = useState(() => wasExplained(run.id));
  const [dialog, setDialog] = useState<ExplainStart | null>(null);

  // Which methods work: cheap, no computation on the server.
  useEffect(() => {
    const controller = new AbortController();
    setError(null);
    explainApi
      .methods(run.id, controller.signal)
      .then(setMethods)
      .catch((exc: Error) => exc.name !== 'AbortError' && setError(exc.message));
    return () => controller.abort();
  }, [run.id, attempt]);

  const can = (id: ExplainMethodId) =>
    Boolean(methods && !methods.unavailable && methods.methods.find((m) => m.id === id)?.available);
  const reason = (id: ExplainMethodId) => methods?.methods.find((m) => m.id === id)?.reason ?? null;
  const demo = Boolean(methods?.demo);
  const importance = !demo && can('permutation');
  const boundary = !demo && can('boundary');
  const images = !demo && methods?.kind === 'image' && can('samples') && can('image');
  const previews = importance || boundary || images;
  const load = requested && previews;

  const permutation = useExplanation<PermutationAnswer>(
    run.id,
    load && importance ? { method: 'permutation' } : null
  );
  const plane = useExplanation<BoundaryAnswer>(
    run.id,
    load && boundary ? { method: 'boundary' } : null
  );
  const samples = useExplanation<SamplesAnswer>(
    run.id,
    load && images ? { method: 'samples' } : null
  );
  const picked = pickSamples(samples.data?.samples ?? []);
  // A fixed number of hooks: one per image of the strip.
  const image0 = useExplanation<ImageAnswer>(
    run.id,
    picked[0] ? { method: 'image', sample: picked[0].index } : null
  );
  const image1 = useExplanation<ImageAnswer>(
    run.id,
    picked[1] ? { method: 'image', sample: picked[1].index } : null
  );
  const image2 = useExplanation<ImageAnswer>(
    run.id,
    picked[2] ? { method: 'image', sample: picked[2].index } : null
  );

  const arrived = Boolean(permutation.data || plane.data || samples.data);
  useEffect(() => {
    if (arrived) {
      rememberExplained(run.id);
    }
  }, [arrived, run.id]);

  const open = (start: ExplainStart) => setDialog(start);
  const anyTab =
    can('image') || can('tabular') || can('permutation') || can('boundary') || inside(can);

  let body: React.ReactNode;
  if (error) {
    body = (
      <Group gap="xs">
        <Text size="sm" c="dimmed">
          Cannot check which explanations work for this run: {error}
        </Text>
        <Button
          size="compact-xs"
          variant="light"
          leftSection={<IconRefresh size={14} />}
          onClick={() => setAttempt((n) => n + 1)}
        >
          Try again
        </Button>
      </Group>
    );
  } else if (!methods) {
    body = <Loader size="xs" aria-label="Checking which explanations work" />;
  } else if (methods.unavailable) {
    body = (
      <Text size="sm" c="dimmed">
        {methods.unavailable}
      </Text>
    );
  } else if (demo) {
    body = (
      <Stack gap="xs" align="flex-start">
        <Text size="sm" c="dimmed">
          Trained on random demo data, so there is nothing meaningful to explain: train on a dataset
          to see what the model relies on. You can still look inside the network.
        </Text>
        {inside(can) && (
          <Button
            size="compact-sm"
            variant="light"
            color="grape"
            onClick={() => open({ tab: 'inside' })}
          >
            Look inside the network
          </Button>
        )}
      </Stack>
    );
  } else if (!previews) {
    body = (
      <Stack gap="xs" align="flex-start">
        <Text size="sm" c="dimmed">
          {(methods.kind === 'image'
            ? (reason('samples') ?? reason('image'))
            : (reason('permutation') ?? reason('tabular'))) ?? 'There is no preview for this run.'}
        </Text>
        {anyTab && <OpenFull onClick={() => open({})} />}
      </Stack>
    );
  } else if (!requested) {
    body = (
      <Stack gap="xs" align="flex-start">
        <Text size="sm">
          {images
            ? 'See where the network looks in a few validation images (Grad-CAM and saliency).'
            : importance
              ? `See which features the model relies on most${boundary ? ' and what it predicts across its two features' : ''}.`
              : 'See what the model predicts across the plane of its two input features.'}
        </Text>
        <Button
          variant="light"
          color="grape"
          leftSection={<IconSparkles size={16} />}
          onClick={() => setRequested(true)}
        >
          {images ? 'Show where it looks' : 'Show what matters'}
        </Button>
        <Text size="xs" c="dimmed">
          Computed on the server in a few seconds; it counts as a prediction. Afterwards it is shown
          right away.
        </Text>
      </Stack>
    );
  } else {
    body = (
      <div className={classes.blocks}>
        {importance && <ImportanceBlock result={permutation} onOpen={open} />}
        {boundary && <BoundaryBlock result={plane} onOpen={open} />}
        {images && (
          <ImagesBlock
            samples={samples}
            picked={picked}
            answers={[image0, image1, image2]}
            onOpen={open}
          />
        )}
      </div>
    );
  }

  return (
    <Paper
      component="section"
      aria-labelledby={titleId}
      withBorder
      radius="md"
      p="sm"
      className={classes.section}
      data-run-side
    >
      <Stack gap="sm">
        <Group gap={8} wrap="nowrap">
          <ThemeIcon variant="light" color="grape" radius="xl" size="md" aria-hidden>
            <IconBulb size={16} />
          </ThemeIcon>
          <Text fw={600} id={titleId}>
            Why the model decides
          </Text>
        </Group>
        {body}
      </Stack>
      <ExplainDialog run={run} start={dialog} onClose={() => setDialog(null)} />
    </Paper>
  );
}

function inside(can: (id: ExplainMethodId) => boolean): boolean {
  return can('activations') || can('weights');
}

const TAB_NAMES: Record<ExplainTab, string> = {
  prediction: 'Explain a prediction',
  importance: 'Feature importance',
  boundary: 'Decision boundary',
  inside: 'Inside the network',
};

/** "Open the full explanation" (on a tab of the dialog). */
function OpenFull({ tab, onClick }: { tab?: ExplainTab; onClick: () => void }) {
  return (
    <Button
      variant="subtle"
      size="compact-xs"
      rightSection={<IconArrowUpRight size={14} />}
      onClick={onClick}
      aria-label={tab ? `Open the full explanation: ${TAB_NAMES[tab]}` : undefined}
    >
      Open the full explanation
    </Button>
  );
}

function Block({
  title,
  text,
  children,
  tab,
  onOpen,
}: {
  title: string;
  text: string;
  children: React.ReactNode;
  tab: ExplainTab;
  onOpen: (start: ExplainStart) => void;
}) {
  return (
    <Stack gap={6} align="stretch" style={{ minWidth: 0 }}>
      <Text size="sm" fw={500}>
        {title}
      </Text>
      <Text size="xs" c="dimmed">
        {text}
      </Text>
      {children}
      <div>
        <OpenFull tab={tab} onClick={() => onOpen({ tab })} />
      </div>
    </Stack>
  );
}

function ImportanceBlock({
  result,
  onOpen,
}: {
  result: Explanation<PermutationAnswer>;
  onOpen: (start: ExplainStart) => void;
}) {
  const [all, setAll] = useState(false);
  const count = result.data?.features.length ?? 0;
  return (
    <Block
      title="What matters most"
      text="How much the validation score drops when the values of one feature are shuffled: the longer the bar, the more the model relies on that feature."
      tab="importance"
      onOpen={onOpen}
    >
      <Pending result={result} what="the feature importance">
        {(data) => (
          <Stack gap={4} align="flex-start">
            <div style={{ width: '100%' }}>
              <PermutationView answer={data} limit={all ? undefined : TOP_FEATURES} compact />
            </div>
            {count > TOP_FEATURES && (
              <Button
                variant="subtle"
                color="gray"
                size="compact-xs"
                onClick={() => setAll((open) => !open)}
                aria-expanded={all}
              >
                {all ? `Show the top ${TOP_FEATURES}` : `Show all ${count} features`}
              </Button>
            )}
          </Stack>
        )}
      </Pending>
    </Block>
  );
}

function BoundaryBlock({
  result,
  onOpen,
}: {
  result: Explanation<BoundaryAnswer>;
  onOpen: (start: ExplainStart) => void;
}) {
  return (
    <Block
      title="Decision boundary"
      text="What the model predicts for every combination of its two input features, with training points on top. Pale areas are where it hesitates."
      tab="boundary"
      onOpen={onOpen}
    >
      <Pending result={result} what="the decision boundary">
        {(data) => <BoundaryView answer={data} compact />}
      </Pending>
    </Block>
  );
}

type MapKind = 'gradcam' | 'saliency';

function ImagesBlock({
  samples,
  picked,
  answers,
  onOpen,
}: {
  samples: Explanation<SamplesAnswer>;
  picked: ListedSample[];
  answers: Explanation<ImageAnswer>[];
  onOpen: (start: ExplainStart) => void;
}) {
  const [chosen, setChosen] = useState<MapKind>('gradcam');
  const loaded = answers.map((answer) => answer.data).filter((a): a is ImageAnswer => !!a);
  // Grad-CAM needs a convolution: without one, saliency it is.
  const noGradcam = loaded.length > 0 && loaded.every((answer) => answer.gradcam.unavailable);
  const kind: MapKind = noGradcam ? 'saliency' : chosen;
  return (
    <Block
      title="Where it looks"
      text={
        kind === 'gradcam'
          ? 'Grad-CAM: the regions where the network found evidence for its answer (bright = more). Click an image to explain it in full.'
          : 'Saliency: the pixels that change the answer most when they change a little. Click an image to explain it in full.'
      }
      tab="prediction"
      onOpen={onOpen}
    >
      <Pending result={samples} what="the validation images">
        {() =>
          picked.length === 0 ? (
            <Text size="sm" c="dimmed">
              No validation images to show.
            </Text>
          ) : (
            <Stack gap="xs" align="flex-start">
              <SegmentedControl
                size="xs"
                value={kind}
                onChange={(value) => setChosen(value as MapKind)}
                data={[
                  { value: 'gradcam', label: 'Grad-CAM', disabled: noGradcam },
                  { value: 'saliency', label: 'Saliency' },
                ]}
                aria-label="Heatmap"
              />
              <div className={classes.strip}>
                {picked.map((sample, i) => (
                  <ImageTile
                    key={sample.index}
                    sample={sample}
                    answer={answers[i]}
                    kind={kind}
                    onOpen={() =>
                      onOpen({
                        tab: 'prediction',
                        sample: { kind: 'validation', index: sample.index, label: sample.label },
                      })
                    }
                  />
                ))}
              </div>
            </Stack>
          )
        }
      </Pending>
    </Block>
  );
}

function ImageTile({
  sample,
  answer,
  kind,
  onOpen,
}: {
  sample: ListedSample;
  answer: Explanation<ImageAnswer>;
  kind: MapKind;
  onOpen: () => void;
}) {
  const data = answer.data;
  const map = data ? data[kind] : null;
  const aspect = data ? data.size[1] / data.size[0] : 1;
  const caption = `true ${sample.label ?? '–'} · model ${sample.predicted ?? '–'}`;
  return (
    <UnstyledButton
      className={classes.tile}
      onClick={onOpen}
      aria-label={`Explain sample ${sample.index} in full: ${caption}`}
    >
      <Overlay
        image={data?.image ?? sample.png ?? ''}
        heat={map && !map.unavailable ? map.png : undefined}
        opacity={0.7}
        aspect={aspect}
        label={`Sample ${sample.index}`}
      />
      <Text
        size="xs"
        className={classes.caption}
        c={sample.correct === false ? 'red' : undefined}
        title={`${caption} (${percent(sample.probability)})`}
      >
        {caption}
      </Text>
      {answer.loading && <Loader size={12} aria-label="Computing" />}
      {answer.error && (
        <Text size="xs" c="red">
          {answer.error}
        </Text>
      )}
    </UnstyledButton>
  );
}
