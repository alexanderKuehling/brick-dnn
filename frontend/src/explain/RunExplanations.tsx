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
import { i18n, useTranslation } from '@/i18n';
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
  useTranslation();
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
          {i18n.t('ui.explain.RunExplanations.cannotCheckWhichExplanationsWork', { error })}
        </Text>
        <Button
          size="compact-xs"
          variant="light"
          leftSection={<IconRefresh size={14} />}
          onClick={() => setAttempt((n) => n + 1)}
        >
          {i18n.t('ui.explain.RunExplanations.tryAgain')}
        </Button>
      </Group>
    );
  } else if (!methods) {
    body = (
      <Loader
        size="xs"
        aria-label={i18n.t('ui.explain.RunExplanations.checkingWhichExplanationsWork')}
      />
    );
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
          {i18n.t('ui.explain.RunExplanations.trainedOnRandomDemoData')}
        </Text>
        {inside(can) && (
          <Button
            size="compact-sm"
            variant="light"
            color="grape"
            onClick={() => open({ tab: 'inside' })}
          >
            {i18n.t('ui.explain.RunExplanations.lookInsideTheNetwork')}
          </Button>
        )}
      </Stack>
    );
  } else if (!previews) {
    body = (
      <Stack gap="xs" align="flex-start">
        <Text size="sm" c="dimmed">
          {methods.kind === 'tokens'
            ? // Texts: no attributions or importance, but the weights are on the Inside tab.
              can('weights')
              ? i18n.t('ui.explain.RunExplanations.textsWeightsOnly')
              : (reason('weights') ?? i18n.t('ui.explain.RunExplanations.thereIsNoPreviewFor'))
            : ((methods.kind === 'image'
                ? (reason('samples') ?? reason('image'))
                : (reason('permutation') ?? reason('tabular'))) ??
              i18n.t('ui.explain.RunExplanations.thereIsNoPreviewFor'))}
        </Text>
        {anyTab && (
          <OpenFull
            tab={methods.kind === 'tokens' ? 'inside' : undefined}
            onClick={() => open(methods.kind === 'tokens' ? { tab: 'inside' } : {})}
          />
        )}
      </Stack>
    );
  } else if (!requested) {
    body = (
      <Stack gap="xs" align="flex-start">
        <Text size="sm">
          {images
            ? i18n.t('ui.explain.RunExplanations.seeWhereTheNetworkLooks')
            : importance
              ? i18n.t('ui.explain.RunExplanations.seeWhichFeaturesTheModel', {
                  value: boundary
                    ? i18n.t('ui.explain.RunExplanations.andWhatItPredictsAcross')
                    : '',
                })
              : i18n.t('ui.explain.RunExplanations.seeWhatTheModelPredicts')}
        </Text>
        <Button
          variant="light"
          color="grape"
          leftSection={<IconSparkles size={16} />}
          onClick={() => setRequested(true)}
        >
          {images
            ? i18n.t('ui.explain.RunExplanations.showWhereItLooks')
            : i18n.t('ui.explain.RunExplanations.showWhatMatters')}
        </Button>
        <Text size="xs" c="dimmed">
          {i18n.t('ui.explain.RunExplanations.computedOnTheServerIn')}
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
            {i18n.t('ui.explain.RunExplanations.whyTheModelDecides')}
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
  get prediction() {
    return i18n.t('ui.explain.RunExplanations.explainAPrediction');
  },
  get importance() {
    return i18n.t('ui.explain.RunExplanations.featureImportance');
  },
  get boundary() {
    return i18n.t('ui.explain.RunExplanations.decisionBoundary');
  },
  get inside() {
    return i18n.t('ui.explain.RunExplanations.insideTheNetwork');
  },
};

/** "Open the full explanation" (on a tab of the dialog). */
function OpenFull({ tab, onClick }: { tab?: ExplainTab; onClick: () => void }) {
  useTranslation();
  return (
    <Button
      variant="subtle"
      size="compact-xs"
      rightSection={<IconArrowUpRight size={14} />}
      onClick={onClick}
      aria-label={
        tab
          ? i18n.t('ui.explain.RunExplanations.openTheFullExplanation2', { value: TAB_NAMES[tab] })
          : undefined
      }
    >
      {i18n.t('ui.explain.RunExplanations.openTheFullExplanation')}
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
  useTranslation();
  const [all, setAll] = useState(false);
  const count = result.data?.features.length ?? 0;
  return (
    <Block
      title={i18n.t('ui.explain.RunExplanations.whatMattersMost')}
      text={i18n.t('ui.explain.RunExplanations.howMuchTheValidationScore')}
      tab="importance"
      onOpen={onOpen}
    >
      <Pending result={result} what={i18n.t('ui.explain.RunExplanations.theFeatureImportance')}>
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
                {all
                  ? i18n.t('ui.explain.RunExplanations.showTheTop', { TOP_FEATURES })
                  : i18n.t('ui.explain.RunExplanations.showAllFeatures', { count })}
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
  useTranslation();
  return (
    <Block
      title={i18n.t('ui.explain.RunExplanations.decisionBoundary')}
      text={i18n.t('ui.explain.RunExplanations.whatTheModelPredictsFor')}
      tab="boundary"
      onOpen={onOpen}
    >
      <Pending result={result} what={i18n.t('ui.explain.RunExplanations.theDecisionBoundary')}>
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
  useTranslation();
  const [chosen, setChosen] = useState<MapKind>('gradcam');
  const loaded = answers.map((answer) => answer.data).filter((a): a is ImageAnswer => !!a);
  // Grad-CAM needs a convolution: without one, saliency it is.
  const noGradcam = loaded.length > 0 && loaded.every((answer) => answer.gradcam.unavailable);
  const kind: MapKind = noGradcam ? 'saliency' : chosen;
  return (
    <Block
      title={i18n.t('ui.explain.RunExplanations.whereItLooks')}
      text={
        kind === 'gradcam'
          ? i18n.t('ui.explain.RunExplanations.gradCamTheRegionsWhere')
          : i18n.t('ui.explain.RunExplanations.saliencyThePixelsThatChange')
      }
      tab="prediction"
      onOpen={onOpen}
    >
      <Pending result={samples} what={i18n.t('ui.explain.RunExplanations.theValidationImages')}>
        {() =>
          picked.length === 0 ? (
            <Text size="sm" c="dimmed">
              {i18n.t('ui.explain.RunExplanations.noValidationImagesToShow')}
            </Text>
          ) : (
            <Stack gap="xs" align="flex-start">
              <SegmentedControl
                size="xs"
                value={kind}
                onChange={(value) => setChosen(value as MapKind)}
                data={[
                  { value: 'gradcam', label: 'Grad-CAM', disabled: noGradcam },
                  { value: 'saliency', label: i18n.t('ui.explain.RunExplanations.saliency') },
                ]}
                aria-label={i18n.t('ui.explain.RunExplanations.heatmap')}
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
  useTranslation();
  const data = answer.data;
  const map = data ? data[kind] : null;
  const aspect = data ? data.size[1] / data.size[0] : 1;
  const caption = i18n.t('ui.explain.RunExplanations.trueModel', {
    value: sample.label ?? '–',
    value2: sample.predicted ?? '–',
  });
  return (
    <UnstyledButton
      className={classes.tile}
      onClick={onOpen}
      aria-label={i18n.t('ui.explain.RunExplanations.explainSampleInFull', {
        index: sample.index,
        caption,
      })}
    >
      <Overlay
        image={data?.image ?? sample.png ?? ''}
        heat={map && !map.unavailable ? map.png : undefined}
        opacity={0.7}
        aspect={aspect}
        label={i18n.t('ui.explain.RunExplanations.sample', { index: sample.index })}
      />
      <Text
        size="xs"
        className={classes.caption}
        c={sample.correct === false ? 'red' : undefined}
        title={`${caption} (${percent(sample.probability)})`}
      >
        {caption}
      </Text>
      {answer.loading && (
        <Loader size={12} aria-label={i18n.t('ui.explain.RunExplanations.computing')} />
      )}
      {answer.error && (
        <Text size="xs" c="red">
          {answer.error}
        </Text>
      )}
    </UnstyledButton>
  );
}
