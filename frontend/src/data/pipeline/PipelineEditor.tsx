import { useEffect, useId, useRef, useState } from 'react';
import { IconArrowBackUp, IconEye } from '@tabler/icons-react';
import { Button, Group, Loader, Paper, Stack, Text, Title, VisuallyHidden } from '@mantine/core';
import type {
  ColumnInfo,
  DataSource,
  PipelineStep,
  PrepSettings,
  StepCatalog,
  StepReport,
  StepSpec,
} from '@/api/types';
import { usePrepAccess } from '../prepAccess';
import { columnsBefore, finalColumns, insertStep, stepSpec } from '../steps';
import { AddStepMenu } from './AddStepMenu';
import { StepCard } from './StepCard';
import { canMoveStep, duplicateStep, MAX_STEPS, moveStep, restoreStep } from './stepList';
import { TargetSplitCard } from './TargetSplitCard';
import classes from './PipelineEditor.module.css';

export interface PipelineEditorProps {
  catalog: StepCatalog;
  steps: PipelineStep[];
  onStepsChange: (steps: PipelineStep[]) => void;
  settings: PrepSettings;
  onSettingsChange: (settings: PrepSettings) => void;
  /** Reports of the last preview, by step id. */
  reports: Record<string, StepReport>;
  /** Columns of the main source (input of the first step). */
  sourceColumns: ColumnInfo[];
  /** All data sources of the user (for join / append). */
  sources: DataSource[];
  currentSourceId: number;
  /** Step whose output is previewed (null: the end of the pipeline). */
  upto: string | null;
  onUptoChange: (id: string | null) => void;
  previewing: boolean;
  /** Resolved task from the preview. */
  task: 'classification' | 'regression' | null;
  splitCounts: number[] | null;
}

/** How long "Undo" stays available after deleting a step. */
const UNDO_MS = 10_000;

/** Target & split settings and the list of preparation steps with their reports. */
export function PipelineEditor({
  catalog,
  steps,
  onStepsChange,
  settings,
  onSettingsChange,
  reports,
  sourceColumns,
  sources,
  currentSourceId,
  upto,
  onUptoChange,
  previewing,
  task,
  splitCounts,
}: PipelineEditorProps) {
  const headingId = useId();
  // Viewers of a shared preparation look at the steps; others only use the main table.
  const { readOnly, sharedBy } = usePrepAccess();
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(() => new Set());
  const [added, setAdded] = useState<string | null>(null);
  const [deleted, setDeleted] = useState<{ step: PipelineStep; index: number } | null>(null);
  // Every step id used so far: new steps never get the id of a deleted one.
  const usedIds = useRef(new Set<string>());

  useEffect(() => {
    for (const step of steps) {
      usedIds.current.add(step.id);
    }
  }, [steps]);

  /** Ids new steps must not get (deleted steps, steps with a report). */
  const reservedIds = () => [...usedIds.current, ...Object.keys(reports)];

  useEffect(() => {
    if (!deleted) {
      return undefined;
    }
    const timer = window.setTimeout(() => setDeleted(null), UNDO_MS);
    return () => window.clearTimeout(timer);
  }, [deleted]);

  const hasOtherSources = sources.some((source) => source.id !== currentSourceId);
  const limit =
    steps.length >= MAX_STEPS ? `A preparation can have at most ${MAX_STEPS} steps.` : null;
  const uptoIndex = upto === null ? -1 : steps.findIndex((step) => step.id === upto);

  const toggle = (id: string) => {
    const next = new Set(expanded);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    setExpanded(next);
  };

  const addStep = (spec: StepSpec) => {
    const next = insertStep(catalog, steps, { type: spec.type, params: {} }, null, {
      reserved: reservedIds(),
    });
    const created = next.find((step) => !steps.some((old) => old.id === step.id));
    onStepsChange(next);
    if (created) {
      // New steps open their settings, which usually need a choice (columns …).
      setExpanded(new Set(expanded).add(created.id));
      setAdded(created.id);
    }
  };

  const replace = (index: number, step: PipelineStep) =>
    onStepsChange(steps.map((s, i) => (i === index ? step : s)));

  const remove = (index: number) => {
    const step = steps[index];
    onStepsChange(steps.filter((_, i) => i !== index));
    setDeleted({ step, index });
    if (upto === step.id) {
      onUptoChange(null);
    }
  };

  const undo = () => {
    if (deleted) {
      onStepsChange(restoreStep(catalog, steps, deleted.step, deleted.index, reservedIds()));
      setDeleted(null);
    }
  };

  const addMenu = (fullWidth: boolean) => (
    <AddStepMenu
      catalog={catalog}
      onAdd={addStep}
      combineDisabled={
        sharedBy
          ? `Combining tables needs other tables of ${sharedBy}: only they can add these steps.`
          : hasOtherSources
            ? null
            : 'Import another table first to combine it with this one.'
      }
      disabled={limit}
      buttonProps={
        fullWidth ? { variant: 'light', fullWidth: true } : { size: 'compact-sm', variant: 'light' }
      }
    />
  );

  return (
    <Stack gap="md" className={classes.editor}>
      <TargetSplitCard
        settings={settings}
        onSettingsChange={onSettingsChange}
        columns={finalColumns(steps, reports, sourceColumns)}
        task={task}
        splitCounts={splitCounts}
        readOnly={readOnly}
      />

      <section aria-labelledby={headingId} aria-busy={previewing}>
        <Group justify="space-between" mb="xs" wrap="nowrap">
          <Group gap={8} wrap="nowrap">
            <Title order={3} size="h5" id={headingId}>
              Steps
            </Title>
            {steps.length > 0 && (
              <Text size="sm" c="dimmed">
                {steps.length}
              </Text>
            )}
            <span aria-live="polite" className={classes.live}>
              {previewing && (
                <>
                  <Loader size={12} aria-hidden />
                  <VisuallyHidden>Updating the preview</VisuallyHidden>
                </>
              )}
            </span>
          </Group>
          {!readOnly && addMenu(false)}
        </Group>

        {uptoIndex >= 0 && (
          <Paper withBorder radius="md" p="xs" mb="xs" className={classes.uptoBanner}>
            <Group gap="xs" justify="space-between" wrap="nowrap">
              <Text size="sm" className={classes.reportLine}>
                <IconEye size={16} aria-hidden />
                <span>
                  The preview shows the data after step {uptoIndex + 1}
                  {uptoIndex < steps.length - 1 ? '; later steps do not run.' : '.'}
                </span>
              </Text>
              <Button size="compact-xs" variant="subtle" onClick={() => onUptoChange(null)}>
                Show all steps
              </Button>
            </Group>
          </Paper>
        )}

        {deleted && (
          <Paper withBorder radius="md" p="xs" mb="xs" role="status">
            <Group gap="xs" justify="space-between" wrap="nowrap">
              <Text size="sm" truncate>
                Deleted “{stepSpec(catalog, deleted.step.type)?.label ?? deleted.step.type}”.
              </Text>
              <Button
                size="compact-xs"
                variant="subtle"
                leftSection={<IconArrowBackUp size={14} />}
                onClick={undo}
              >
                Undo
              </Button>
            </Group>
          </Paper>
        )}

        {steps.length === 0 ? (
          <Paper withBorder radius="md" p="md" className={classes.empty}>
            <Text size="sm" fw={500}>
              No steps yet
            </Text>
            {!readOnly && (
              <Text size="sm" c="dimmed" mt={4}>
                Add steps to clean and transform the data, e.g. fill missing values, clip outliers
                or encode categories. The column menus of the data table and the Readiness tab
                suggest steps, too.
              </Text>
            )}
          </Paper>
        ) : (
          <ol className={classes.stepList}>
            {steps.map((step, index) => (
              <StepCard
                key={step.id}
                step={step}
                index={index}
                spec={stepSpec(catalog, step.type)}
                report={reports[step.id]}
                columns={columnsBefore(steps, index, reports, sourceColumns)}
                sources={sources}
                currentSourceId={currentSourceId}
                expanded={expanded.has(step.id)}
                onToggle={() => toggle(step.id)}
                onChange={(next) => replace(index, next)}
                onMove={(delta) => onStepsChange(moveStep(catalog, steps, index, delta))}
                canMoveUp={canMoveStep(catalog, steps, index, -1)}
                canMoveDown={canMoveStep(catalog, steps, index, 1)}
                onDuplicate={() => onStepsChange(duplicateStep(steps, index, reservedIds()))}
                onDelete={() => remove(index)}
                previewed={upto === step.id}
                notRun={uptoIndex >= 0 && index > uptoIndex}
                onPreviewToggle={() => onUptoChange(upto === step.id ? null : step.id)}
                previewing={previewing}
                scrollIntoView={added === step.id}
                readOnly={readOnly}
              />
            ))}
          </ol>
        )}

        {steps.length >= 3 && !readOnly && <div className={classes.addBottom}>{addMenu(true)}</div>}
      </section>
    </Stack>
  );
}
