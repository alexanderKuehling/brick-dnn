import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import {
  IconAlertCircle,
  IconAlertTriangle,
  IconArrowDown,
  IconArrowMerge,
  IconArrowUp,
  IconChevronRight,
  IconCopy,
  IconDots,
  IconEye,
  IconSchool,
  IconTrash,
} from '@tabler/icons-react';
import {
  ActionIcon,
  Anchor,
  Badge,
  Group,
  Menu,
  Paper,
  Stack,
  Switch,
  Text,
  Tooltip,
  UnstyledButton,
  VisuallyHidden,
} from '@mantine/core';
import type { ColumnInfo, DataSource, PipelineStep, StepReport, StepSpec } from '@/api/types';
import { formatCount } from '../table/format';
import { GroupIcon } from './groups';
import { errorField, isParamVisible, paramSummary, paramValue } from './params';
import {
  STATUS_COLORS,
  STATUS_HINTS,
  STATUS_LABELS,
  stepStatus,
  type StepDisplayStatus,
} from './stepList';
import { StepParamField } from './StepParamField';
import classes from './PipelineEditor.module.css';

export interface StepCardProps {
  step: PipelineStep;
  index: number;
  /** Undefined for step types the catalog does not know. */
  spec: StepSpec | undefined;
  report: StepReport | undefined;
  /** The columns this step works on. */
  columns: ColumnInfo[];
  sources: DataSource[];
  currentSourceId: number;
  expanded: boolean;
  onToggle: () => void;
  onChange: (step: PipelineStep) => void;
  onMove: (delta: -1 | 1) => void;
  canMoveUp: boolean;
  canMoveDown: boolean;
  onDuplicate: () => void;
  onDelete: () => void;
  /** The preview shows the data after this step. */
  previewed: boolean;
  /**
   * The preview shows the data of an earlier step, so this one does not run: `report` may be
   * from an earlier preview of all steps and is only used for the columns.
   */
  notRun?: boolean;
  onPreviewToggle: () => void;
  /** A new preview is loading (the report may be out of date). */
  previewing: boolean;
  /** Scroll the card into view once (after it was added). */
  scrollIntoView?: boolean;
  /** Shown, not changed: no actions, the settings are disabled. */
  readOnly?: boolean;
}

const DETAILS_SHOWN = 4;

const FITTED_HINT =
  'Means, bounds, categories … are computed on the training rows only and then applied to all rows, so validation and test data stay unseen.';

/** One step of the pipeline: header with status and actions, report, and its settings. */
export function StepCard({
  step,
  index,
  spec,
  report,
  columns,
  sources,
  currentSourceId,
  expanded,
  onToggle,
  onChange,
  onMove,
  canMoveUp,
  canMoveDown,
  onDuplicate,
  onDelete,
  previewed,
  notRun = false,
  onPreviewToggle,
  previewing,
  scrollIntoView = false,
  readOnly = false,
}: StepCardProps) {
  const formId = useId();
  const ref = useRef<HTMLLIElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const refocus = useRef(false);
  const [allDetails, setAllDetails] = useState(false);

  useLayoutEffect(() => {
    // Reordering moves the card in the DOM, which can drop the focus: put it back.
    if (refocus.current) {
      refocus.current = false;
      toggleRef.current?.focus();
    }
  }, [index]);

  useEffect(() => {
    if (scrollIntoView) {
      ref.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }, [scrollIntoView]);

  const label = spec?.label ?? step.type;
  const status: StepDisplayStatus = !spec
    ? 'error'
    : notRun && step.enabled
      ? 'not_run'
      : stepStatus(step, report);
  const ranReport =
    report && report.type === step.type && step.enabled && !notRun ? report : undefined;
  const fieldError = spec && status === 'error' ? errorField(spec, ranReport) : null;

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (readOnly || !event.altKey || (event.key !== 'ArrowUp' && event.key !== 'ArrowDown')) {
      return;
    }
    event.preventDefault();
    if (event.key === 'ArrowUp' && canMoveUp) {
      refocus.current = true;
      onMove(-1);
    } else if (event.key === 'ArrowDown' && canMoveDown) {
      refocus.current = true;
      onMove(1);
    }
  };

  const actionsMenu = readOnly ? null : (
    <Menu position="bottom-end" shadow="md" withinPortal>
      <Menu.Target>
        <ActionIcon variant="subtle" color="gray" aria-label={`Actions for step ${index + 1}`}>
          <IconDots size={16} />
        </ActionIcon>
      </Menu.Target>
      <Menu.Dropdown>
        <Menu.Item
          leftSection={<IconArrowUp size={14} />}
          disabled={!canMoveUp}
          onClick={() => onMove(-1)}
          rightSection={
            <Text size="xs" c="dimmed">
              Alt ↑
            </Text>
          }
        >
          Move up
        </Menu.Item>
        <Menu.Item
          leftSection={<IconArrowDown size={14} />}
          disabled={!canMoveDown}
          onClick={() => onMove(1)}
          rightSection={
            <Text size="xs" c="dimmed">
              Alt ↓
            </Text>
          }
        >
          Move down
        </Menu.Item>
        {spec && (
          <Menu.Item leftSection={<IconCopy size={14} />} onClick={onDuplicate}>
            Duplicate
          </Menu.Item>
        )}
        <Menu.Divider />
        <Menu.Item color="red" leftSection={<IconTrash size={14} />} onClick={onDelete}>
          Delete
        </Menu.Item>
      </Menu.Dropdown>
    </Menu>
  );

  if (!spec) {
    return (
      <Paper
        component="li"
        ref={ref}
        withBorder
        radius="md"
        className={classes.step}
        data-status="error"
        aria-label={`Step ${index + 1}: unknown step type ${step.type}`}
      >
        <div className={classes.stepHeader}>
          <div className={classes.stepToggle}>
            <span className={classes.chevronSpace} />
            <GroupIcon group={undefined} />
            <span className={classes.stepTitle}>
              <Text component="span" size="xs" c="dimmed" fw={600} className={classes.stepNumber}>
                {index + 1}
              </Text>
              <Text component="span" size="sm" fw={600} truncate>
                {step.type}
              </Text>
            </span>
          </div>
          <Badge size="sm" variant="light" color="red">
            Unknown
          </Badge>
          {actionsMenu}
        </div>
        <Text size="sm" c="red" mt={6} className={classes.reportLine}>
          <IconAlertCircle size={16} aria-hidden />
          <span>
            Unknown step type “{step.type}”. The server does not offer it (any more); delete this
            step.
          </span>
        </Text>
      </Paper>
    );
  }

  const visibleParams = spec.params.filter((param) => isParamVisible(param, step.params, spec));
  const summary = paramSummary(spec, step.params, sources);
  // Older servers do not say what mapping keys are: without a column param they are columns.
  const keysAreColumns = !spec.params.some((p) => p.kind === 'column' || p.kind === 'columns');
  const details = ranReport?.details ?? [];
  const shownDetails = allDetails ? details : details.slice(0, DETAILS_SHOWN);

  return (
    <Paper
      component="li"
      ref={ref}
      withBorder
      radius="md"
      className={classes.step}
      data-status={status}
      data-off={!step.enabled || undefined}
      data-previewed={previewed || undefined}
      aria-label={`Step ${index + 1}: ${label}`}
    >
      <div className={classes.stepHeader}>
        <UnstyledButton
          ref={toggleRef}
          className={classes.stepToggle}
          onClick={onToggle}
          onKeyDown={onKeyDown}
          aria-expanded={expanded}
          aria-controls={expanded ? formId : undefined}
          aria-keyshortcuts={readOnly ? undefined : 'Alt+ArrowUp Alt+ArrowDown'}
        >
          <IconChevronRight
            size={14}
            className={classes.chevron}
            data-expanded={expanded || undefined}
            aria-hidden
          />
          <GroupIcon group={spec.group} />
          <span className={classes.stepTitle}>
            <Text component="span" size="xs" c="dimmed" fw={600} className={classes.stepNumber}>
              {index + 1}
            </Text>
            <Text component="span" size="sm" fw={600} truncate>
              {label}
            </Text>
          </span>
        </UnstyledButton>
        <Tooltip label={STATUS_HINTS[status]} multiline w={220} withArrow>
          <Badge size="sm" variant="light" color={STATUS_COLORS[status]} className={classes.badge}>
            {STATUS_LABELS[status]}
          </Badge>
        </Tooltip>
        <Tooltip
          label={previewed ? 'Show the data after all steps' : 'Preview the data after this step'}
          withArrow
        >
          <ActionIcon
            variant={previewed ? 'filled' : 'subtle'}
            color={previewed ? undefined : 'gray'}
            aria-label="Preview after this step"
            aria-pressed={previewed}
            onClick={onPreviewToggle}
          >
            <IconEye size={16} />
          </ActionIcon>
        </Tooltip>
        <Switch
          size="xs"
          checked={step.enabled}
          disabled={readOnly}
          aria-label={`Use step ${index + 1} (${label})`}
          onChange={(event) => onChange({ ...step, enabled: event.currentTarget.checked })}
        />
        {actionsMenu}
      </div>

      {!expanded && summary && (
        <Text size="xs" c="dimmed" lineClamp={2} className={classes.summary}>
          {summary}
        </Text>
      )}

      <div className={classes.report} data-stale={previewing || undefined}>
        {status === 'error' && ranReport?.error && (
          <Text size="sm" c="red" className={classes.reportLine}>
            <IconAlertCircle size={16} aria-hidden />
            <span>{ranReport.error}</span>
          </Text>
        )}
        {status === 'ok' && ranReport && (
          <>
            {ranReport.summary && <Text size="sm">{ranReport.summary}</Text>}
            <Changes report={ranReport} />
          </>
        )}
        {status === 'skipped' && (
          <Text size="xs" c="dimmed">
            {STATUS_HINTS.skipped}
          </Text>
        )}
        {(status === 'ok' || status === 'error') &&
          ranReport?.warnings.map((warning) => (
            <Text key={warning} size="sm" className={classes.reportLine} data-tone="warning">
              <IconAlertTriangle size={16} aria-hidden />
              <span>{warning}</span>
            </Text>
          ))}
        {status === 'ok' && details.length > 0 && (
          <ul className={classes.details}>
            {shownDetails.map((detail, i) => (
              <li key={`${detail.column ?? ''}-${i}`}>
                {detail.column && <span className={classes.detailColumn}>{detail.column}</span>}
                {detail.column ? ': ' : ''}
                {detail.text}
              </li>
            ))}
            {details.length > DETAILS_SHOWN && (
              <li>
                <Anchor
                  component="button"
                  type="button"
                  size="xs"
                  onClick={() => setAllDetails((v) => !v)}
                >
                  {allDetails ? 'Show less' : `Show all ${details.length}`}
                </Anchor>
              </li>
            )}
          </ul>
        )}
      </div>

      {(spec.fitted || spec.training_only || spec.combine || spec.pre_split) && (
        <Group gap={12} mt={4} className={classes.hints}>
          {spec.fitted && (
            <Tooltip label={FITTED_HINT} multiline w={260} withArrow>
              <Text size="xs" c="dimmed" className={classes.hint}>
                <IconSchool size={14} aria-hidden />
                Learns from training rows
                <VisuallyHidden>. {FITTED_HINT}</VisuallyHidden>
              </Text>
            </Tooltip>
          )}
          {spec.training_only && (
            <Text size="xs" c="dimmed" className={classes.hint}>
              <IconSchool size={14} aria-hidden />
              Changes training rows only
            </Text>
          )}
          {(spec.combine || spec.pre_split) && (
            <Text size="xs" c="dimmed" className={classes.hint}>
              <IconArrowMerge size={14} aria-hidden />
              Runs before the split
            </Text>
          )}
        </Group>
      )}

      {expanded && (
        // A disabled fieldset keeps every kind of field from changing (read-only views).
        <fieldset disabled={readOnly} className={classes.form} id={formId}>
          <Stack gap="sm">
            {visibleParams.map((param) => (
              <StepParamField
                key={param.name}
                param={param}
                value={paramValue(param, step.params)}
                params={step.params}
                columns={columns}
                sources={sources}
                currentSourceId={currentSourceId}
                mappingKeysAreColumns={
                  param.key_kind ? param.key_kind === 'column' : keysAreColumns
                }
                error={fieldError?.param === param.name ? fieldError.message : null}
                onChange={(value) =>
                  onChange({ ...step, params: { ...step.params, [param.name]: value } })
                }
              />
            ))}
            {visibleParams.length === 0 && (
              <Text size="sm" c="dimmed">
                This step has no settings.
              </Text>
            )}
          </Stack>
        </fieldset>
      )}
    </Paper>
  );
}

function plural(count: number, word: string): string {
  return `${formatCount(count)} ${count === 1 ? word : `${word}s`}`;
}

function names(list: string[], max = 3): string {
  return list.length <= max
    ? list.join(', ')
    : `${list.slice(0, max).join(', ')} and ${list.length - max} more`;
}

/** Rows and columns the step changed, e.g. "1,200 → 1,180 rows · +2 columns (a, b)". */
function Changes({ report }: { report: StepReport }) {
  const parts: string[] = [];
  if (report.rows_after !== report.rows_before) {
    parts.push(`${formatCount(report.rows_before)} → ${plural(report.rows_after, 'row')}`);
  }
  if (report.added.length > 0) {
    parts.push(`+${plural(report.added.length, 'column')} (${names(report.added)})`);
  }
  if (report.removed.length > 0) {
    parts.push(`−${plural(report.removed.length, 'column')} (${names(report.removed)})`);
  }
  if (parts.length === 0) {
    return null;
  }
  return (
    <Text size="xs" c="dimmed" className={classes.changes}>
      {parts.join(' · ')}
    </Text>
  );
}
