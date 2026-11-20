import { useEffect, useMemo, useState } from 'react';
import {
  IconAlertTriangle,
  IconArrowLeft,
  IconDeviceFloppy,
  IconPlayerStop,
  IconStack2,
  IconTrash,
} from '@tabler/icons-react';
import {
  Alert,
  Anchor,
  Badge,
  Button,
  Collapse,
  Group,
  Loader,
  NumberInput,
  SegmentedControl,
  Stack,
  Table,
  Text,
  Title,
} from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { notifications } from '@mantine/notifications';
import {
  api,
  type EnsembleComparisonRow,
  type EnsembleSplit,
  type EnsembleState,
  type GraphDocument,
  type RunDetail,
} from '@/api';
import { currentLocale, i18n, lowerCase, useTranslation } from '@/i18n';
import {
  canDeleteRun,
  canSetThreshold,
  canStopRun,
  NO_SCOPE,
  otherStarter,
  type TrainingScope,
} from './access';
import {
  comparisonColumns,
  METHOD_LABELS,
  methodText,
  shares,
  SPLIT_LABELS,
  type ComparisonColumn,
} from './ensemble';
import { EvaluationResults, type ThresholdProps } from './EvaluationResults';
import { formatDate, formatMetric, formatPercent, formatR2, isActive } from './format';
import { PredictionPanel } from './predict/PredictionPanel';
import { confirmDeleteRun, confirmStopRun, StatusBadge } from './runActions';
import { cpuUsage, QueueNotice, ToggleButton } from './runParts';

interface EnsembleRunViewProps {
  run: RunDetail;
  doc?: GraphDocument;
  scope?: TrainingScope;
  onBack: () => void;
  onUpdate: (run: RunDetail) => void;
  onDeleted: (id: string) => void;
  /** Opens a member's run. */
  onOpenRun?: (id: string) => void;
}

/**
 * The run view of an ensemble: its members (links to their runs) with their weights, how it
 * combines them, every member next to the ensemble per split, the ensemble's metrics with the
 * threshold analysis of two classes, and predictions.
 */
export function EnsembleRunView({
  run,
  scope = NO_SCOPE,
  onBack,
  onUpdate,
  onDeleted,
  onOpenRun,
}: EnsembleRunViewProps) {
  useTranslation();
  const state = run.state;
  const info = run.ensemble ?? null;
  const combined = state?.ensemble ?? null;
  const active = isActive(run.status);
  const queued = run.status === 'queued';
  const method = info?.method ?? combined?.method ?? 'average';
  const task = run.task ?? state?.task ?? null;
  const by = otherStarter(run, scope);
  const [predictOpened, predict] = useDisclosure(false);
  const threshold: ThresholdProps = {
    decisionThreshold: run.decision_threshold,
    onSaveThreshold: canSetThreshold(run, scope)
      ? async (value) => {
          const saved = await api.training.runs.setDecisionThreshold(run.id, value);
          onUpdate({ ...run, decision_threshold: saved.decision_threshold });
        }
      : undefined,
  };
  const canPredict = run.has_weights && !active && !info?.broken;
  const positive = run.plan?.class_names?.[1] ?? state?.class_names?.[1];

  return (
    <Stack gap="md">
      <Group justify="space-between" wrap="nowrap" gap="xs">
        <Button
          variant="subtle"
          color="gray"
          size="compact-sm"
          leftSection={<IconArrowLeft size={14} />}
          onClick={onBack}
        >
          {i18n.t('ui.training.EnsembleRunView.allRuns')}
        </Button>
      </Group>

      <div>
        <Group gap="xs" wrap="nowrap" justify="space-between" align="flex-start">
          <Group gap="xs" wrap="nowrap" style={{ minWidth: 0 }}>
            <IconStack2 size={20} style={{ flexShrink: 0 }} aria-hidden />
            <Title order={4} style={{ minWidth: 0, overflowWrap: 'anywhere' }}>
              {run.name}
            </Title>
          </Group>
          <StatusBadge status={run.status} />
        </Group>
        <Text size="xs" c="dimmed" mt={4}>
          {[
            i18n.t('ui.training.EnsembleRunView.ensemble'),
            lowerCase(METHOD_LABELS[method]),
            by && i18n.t('ui.training.DataSourcePicker.byOwner', { owner: by }),
            run.dataset.name,
            run.started_via?.name &&
              i18n.t('ui.training.EnsembleRunView.via', { name: run.started_via.name }),
            i18n.t('ui.training.EnsembleRunView.created', { date: formatDate(run.created_at) }),
            cpuUsage(run),
          ]
            .filter(Boolean)
            .join(' · ')}
        </Text>
      </div>

      {queued ? (
        <QueueNotice run={run} />
      ) : active ? (
        <Group gap="xs" wrap="nowrap">
          <Loader size="xs" />
          <Text size="sm">{run.message || i18n.t('ui.training.EnsembleRunView.combining')}</Text>
        </Group>
      ) : run.status === 'failed' ? (
        <Alert
          color="red"
          icon={<IconAlertTriangle size={18} />}
          title={i18n.t('ui.training.EnsembleRunView.theEnsembleFailed')}
        >
          {run.message}
        </Alert>
      ) : (
        run.message && (
          <Text size="sm" c="dimmed">
            {run.message}
          </Text>
        )
      )}
      {info?.broken && (
        <Alert
          color="yellow"
          icon={<IconAlertTriangle size={18} />}
          title={i18n.t('ui.training.EnsembleRunView.aMemberIsGone')}
        >
          {info.broken}
        </Alert>
      )}

      <Members run={run} scope={scope} onOpenRun={onOpenRun} onUpdate={onUpdate} />

      {combined && <Comparison combined={combined} task={task} />}

      {!active && (
        <EvaluationResults
          evaluation={state?.evaluation}
          threshold={threshold}
          subject={i18n.t('ui.training.EnsembleRunView.theEnsemble')}
        />
      )}

      <Group gap="xs">
        {active && canStopRun(run, scope) && (
          <Button
            color="red"
            variant="light"
            leftSection={<IconPlayerStop size={16} />}
            disabled={run.stop_requested}
            onClick={() => confirmStopRun(run, onUpdate)}
          >
            {queued
              ? i18n.t('ui.training.EnsembleRunView.cancel')
              : run.stop_requested
                ? i18n.t('ui.training.EnsembleRunView.stopping')
                : i18n.t('ui.training.EnsembleRunView.stop')}
          </Button>
        )}
        {canDeleteRun(run, scope) && (
          <Button
            variant="subtle"
            color="red"
            leftSection={<IconTrash size={16} />}
            onClick={() => confirmDeleteRun(run, onDeleted)}
          >
            {i18n.t('ui.training.EnsembleRunView.delete')}
          </Button>
        )}
      </Group>

      {canPredict && (
        <div>
          <ToggleButton opened={predictOpened} onClick={predict.toggle}>
            {i18n.t('ui.training.EnsembleRunView.tryTheEnsemblePredict')}
          </ToggleButton>
          <Collapse expanded={predictOpened}>
            {predictOpened && (
              <PredictionPanel
                runId={run.id}
                runName={run.name}
                decisionThreshold={run.decision_threshold}
                positiveClass={positive}
                ensemble
              />
            )}
          </Collapse>
        </div>
      )}
    </Stack>
  );
}

/** The runs it combines, with links, models and weights (editable by its owner). */
function Members({
  run,
  scope,
  onOpenRun,
  onUpdate,
}: {
  run: RunDetail;
  scope: TrainingScope;
  onOpenRun?: (id: string) => void;
  onUpdate: (run: RunDetail) => void;
}) {
  useTranslation();
  const info = run.ensemble;
  const members = useMemo(() => info?.members ?? [], [info]);
  const method = info?.method ?? 'average';
  const saved = useMemo(() => members.map((m) => m.weight ?? 0), [members]);
  const [weights, setWeights] = useState<number[]>(saved);
  const [saving, setSaving] = useState(false);
  useEffect(() => setWeights(saved), [saved]);
  const editable =
    method === 'average' &&
    canSetThreshold(run, scope) &&
    run.status === 'completed' &&
    !info?.broken &&
    members.length > 0;
  const changed = weights.some((w, i) => Math.abs(w - (saved[i] ?? 0)) > 1e-9);
  const shown = shares(weights);
  const valid = weights.some((w) => w > 0);

  const save = async () => {
    setSaving(true);
    try {
      onUpdate(await api.training.runs.setEnsembleWeights(run.id, weights));
      notifications.show({
        color: 'green',
        title: i18n.t('ui.training.EnsembleRunView.weightsSaved'),
        message: i18n.t('ui.training.EnsembleRunView.theEnsembleWasMeasuredAgain'),
      });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: i18n.t('ui.training.EnsembleRunView.cannotSaveTheWeights'),
        message: (error as Error).message,
      });
    } finally {
      setSaving(false);
    }
  };

  return (
    <Stack gap="xs" component="section" aria-label={i18n.t('ui.training.EnsembleRunView.members')}>
      <Group justify="space-between" gap="xs" align="baseline">
        <Text size="sm" fw={600}>
          {i18n.t('ui.training.EnsembleRunView.members')}
        </Text>
        <Text size="xs" c="dimmed">
          {METHOD_LABELS[method]}
        </Text>
      </Group>
      <Text size="xs" c="dimmed">
        {methodText(method, run.task)}
      </Text>
      <Table
        fz="sm"
        withTableBorder
        aria-label={i18n.t('ui.training.EnsembleRunView.membersOfTheEnsemble')}
      >
        <Table.Thead>
          <Table.Tr>
            <Table.Th>{i18n.t('ui.training.EnsembleRunView.run')}</Table.Th>
            <Table.Th>{i18n.t('ui.training.EnsembleRunView.model')}</Table.Th>
            {method === 'average' && (
              <Table.Th w={editable ? 130 : 80}>
                {i18n.t('ui.training.EnsembleRunView.weight')}
              </Table.Th>
            )}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {members.map((member, index) => (
            <Table.Tr key={member.id ?? `deleted-${index}`}>
              <Table.Td>
                {member.id && member.visible && onOpenRun ? (
                  <Anchor component="button" size="sm" onClick={() => onOpenRun(member.id!)}>
                    {member.name}
                  </Anchor>
                ) : (
                  <Text size="sm">{member.name}</Text>
                )}
                {member.deleted && (
                  <Badge color="red" variant="light" tt="none" ml={6}>
                    {i18n.t('ui.training.EnsembleRunView.deleted')}
                  </Badge>
                )}
              </Table.Td>
              <Table.Td>
                <Text size="sm" c="dimmed">
                  {member.model ?? '–'}
                </Text>
              </Table.Td>
              {method === 'average' && (
                <Table.Td>
                  {editable ? (
                    <NumberInput
                      size="xs"
                      min={0}
                      step={0.1}
                      decimalScale={3}
                      aria-label={i18n.t('ui.training.EnsembleRunView.weightOf', {
                        name: member.name,
                      })}
                      value={weights[index] ?? 0}
                      rightSection={
                        <Text size="xs" c="dimmed" pr={6}>
                          {formatPercent(shown[index], 0)}
                        </Text>
                      }
                      rightSectionWidth={44}
                      onChange={(value) =>
                        setWeights((current) =>
                          current.map((w, i) => (i === index ? Number(value) || 0 : w))
                        )
                      }
                    />
                  ) : (
                    <Text size="sm">{formatPercent(member.weight, 0)}</Text>
                  )}
                </Table.Td>
              )}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
      {editable && changed && (
        <Group gap="xs">
          <Button
            size="compact-sm"
            leftSection={<IconDeviceFloppy size={14} />}
            loading={saving}
            disabled={!valid}
            onClick={() => void save()}
          >
            {i18n.t('ui.training.EnsembleRunView.saveWeights')}
          </Button>
          <Button size="compact-sm" variant="subtle" color="gray" onClick={() => setWeights(saved)}>
            {i18n.t('ui.training.EnsembleRunView.reset')}
          </Button>
          <Text size="xs" c="dimmed">
            {i18n.t('ui.training.EnsembleRunView.measuredAgainOnTheMembers')}
          </Text>
        </Group>
      )}
    </Stack>
  );
}

function formatColumn(column: ComparisonColumn, value: number | null | undefined): string {
  if (column.format === 'r2') {
    return formatR2(value);
  }
  if (column.format === 'percent') {
    return formatPercent(value);
  }
  if (column.format === 'score') {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : '–';
  }
  return formatMetric(value);
}

/** Every member and the ensemble on the same rows of one split. */
function Comparison({ combined, task }: { combined: EnsembleState; task: string | null }) {
  useTranslation();
  const splits = combined.splits.length > 0 ? combined.splits : (['val'] as EnsembleSplit[]);
  const [split, setSplit] = useState<EnsembleSplit>(splits.includes('test') ? 'test' : splits[0]);
  const rows = combined.comparison;
  const binary = rows.some((row) => typeof row[split]?.roc_auc === 'number');
  const columns = comparisonColumns(
    task === 'regression' ? 'regression' : 'classification',
    binary
  );
  const best = (column: ComparisonColumn): number | null => {
    const values = rows
      .map((row) => row[split]?.[column.key])
      .filter((v): v is number => typeof v === 'number');
    if (values.length === 0) {
      return null;
    }
    return column.higher ? Math.max(...values) : Math.min(...values);
  };
  const stackingFit = combined.method === 'stacking' && split === combined.fitted_on;
  const size = combined.rows[split];
  let note: string | null = null;
  if (stackingFit) {
    note = combined.optimistic
      ? i18n.t('ui.training.EnsembleRunView.theStackingModelWasFitted')
      : i18n.t('ui.training.EnsembleRunView.theStackingModelWasFitted2');
  } else if (combined.method === 'average' && combined.optimistic) {
    note = i18n.t('ui.training.EnsembleRunView.noValidationOrTestRows');
  }

  return (
    <Stack
      gap="xs"
      component="section"
      aria-label={i18n.t('ui.training.EnsembleRunView.comparison')}
    >
      <Group justify="space-between" gap="xs" align="center">
        <Text size="sm" fw={600}>
          {i18n.t('ui.training.EnsembleRunView.membersAndEnsemble')}
        </Text>
        {splits.length > 1 && (
          <SegmentedControl
            size="xs"
            aria-label={i18n.t('ui.training.EnsembleRunView.split')}
            value={split}
            onChange={(value) => setSplit(value as EnsembleSplit)}
            data={splits.map((key) => ({ value: key, label: SPLIT_LABELS[key] }))}
          />
        )}
      </Group>
      <Table
        fz="xs"
        striped
        withTableBorder
        aria-label={i18n.t('ui.training.EnsembleRunView.membersAndEnsemble')}
      >
        <Table.Thead>
          <Table.Tr>
            <Table.Th>{SPLIT_LABELS[split]}</Table.Th>
            {columns.map((column) => (
              <Table.Th key={column.key}>{column.label}</Table.Th>
            ))}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {rows.map((row: EnsembleComparisonRow) => {
            const ensemble = row.member === null;
            return (
              <Table.Tr key={`${row.member ?? 'ensemble'}`} fw={ensemble ? 700 : undefined}>
                <Table.Td>
                  {ensemble ? i18n.t('ui.training.EnsembleRunView.ensemble') : row.name}
                </Table.Td>
                {columns.map((column) => {
                  const value = row[split]?.[column.key];
                  const top = typeof value === 'number' && value === best(column);
                  return (
                    <Table.Td key={column.key} c={top ? 'teal' : undefined}>
                      {formatColumn(column, value)}
                    </Table.Td>
                  );
                })}
              </Table.Tr>
            );
          })}
        </Table.Tbody>
      </Table>
      <Text size="xs" c="dimmed">
        {[
          typeof size === 'number' &&
            i18n.t('ui.training.EnsembleRunView.rowsTheSameForAll', {
              size: size.toLocaleString(currentLocale()),
            }),
          i18n.t('ui.training.EnsembleRunView.theBestValueOfEach'),
        ]
          .filter(Boolean)
          .join(' ')}
      </Text>
      {note && (
        <Text size="xs" c="orange" data-testid="ensemble-optimistic">
          {note}
        </Text>
      )}
    </Stack>
  );
}
