import { useState } from 'react';
import {
  IconAdjustments,
  IconAdjustmentsHorizontal,
  IconAlertTriangle,
  IconApi,
  IconBooks,
  IconChartLine,
  IconChecklist,
  IconChevronDown,
  IconCircleCheck,
  IconClockHour4,
  IconCode,
  IconDeviceFloppy,
  IconDownload,
  IconEye,
  IconFolder,
  IconFolderOpen,
  IconKey,
  IconList,
  IconPencil,
  IconPlayerPlay,
  IconPlayerStop,
  IconPlugConnected,
  IconSend,
  IconStack2,
  IconStethoscope,
  IconTrash,
  IconUpload,
  IconWand,
  IconX,
} from '@tabler/icons-react';
import { ActionIcon, Anchor, Badge, Collapse, Text, ThemeIcon, Tooltip } from '@mantine/core';
import type { AgentAction, AgentTarget } from '@/api';
import { fullDateTime, timeInDay } from '@/collab/time';
import { i18n, lowerCase, useTranslation } from '@/i18n';
import {
  actionLabel,
  argumentText,
  formatDuration,
  summaryParts,
  TARGET_LABELS,
  unnamedTargets,
  type EntryGroup,
} from './format';
import classes from './Activity.module.css';

type Icon = typeof IconKey;

const TOOL_ICONS: Record<string, Icon> = {
  list_layer_types: IconStack2,
  list_examples: IconBooks,
  analyze_network: IconStethoscope,
  generate_pytorch_code: IconCode,
  list_projects: IconFolder,
  get_project: IconFolderOpen,
  save_project: IconDeviceFloppy,
  delete_project: IconTrash,
  get_training_options: IconAdjustments,
  upload_dataset: IconUpload,
  check_training: IconChecklist,
  start_training: IconPlayerPlay,
  list_training_runs: IconList,
  get_training_run: IconChartLine,
  stop_training: IconPlayerStop,
  get_weights_download: IconDownload,
  predict: IconWand,
  set_decision_threshold: IconAdjustmentsHorizontal,
  create_ensemble: IconStack2,
  delete_training_run: IconTrash,
  delete_dataset: IconTrash,
};

const METHOD_ICONS: Record<string, Icon> = {
  GET: IconEye,
  POST: IconSend,
  PATCH: IconPencil,
  PUT: IconPencil,
  DELETE: IconTrash,
};

function iconOf(entry: AgentAction): Icon {
  if (entry.outcome === 'error') {
    return IconAlertTriangle;
  }
  if (entry.outcome === 'pending') {
    return IconClockHour4;
  }
  if (entry.outcome === 'approved') {
    return IconCircleCheck;
  }
  if (entry.outcome === 'denied') {
    return IconX;
  }
  if (entry.source === 'mcp') {
    return TOOL_ICONS[entry.tool] ?? IconApi;
  }
  return METHOD_ICONS[entry.method] ?? IconApi;
}

export type OpenTarget = (target: AgentTarget) => void;

function TargetLink({
  target,
  onOpen,
  children,
}: {
  target: AgentTarget;
  onOpen: OpenTarget;
  children: React.ReactNode;
}) {
  useTranslation();
  return (
    <Anchor
      component="button"
      type="button"
      className={classes.link}
      onClick={(event: React.MouseEvent) => {
        event.stopPropagation();
        onOpen(target);
      }}
      aria-label={i18n.t('ui.activity.ActivityItem.open', {
        value: lowerCase(TARGET_LABELS[target.kind]),
        name: target.name,
      })}
    >
      {children}
    </Anchor>
  );
}

/** The summary, with the names of networks, runs and datasets as links. */
function Summary({ entry, onOpen }: { entry: AgentAction; onOpen: OpenTarget }) {
  const extra = unnamedTargets(entry.summary, entry.targets);
  return (
    <Text className={classes.summary} fw={500}>
      {summaryParts(entry.summary, entry.targets).map((part, index) =>
        typeof part === 'string' ? (
          part
        ) : (
          <TargetLink key={index} target={part.target} onOpen={onOpen}>
            {part.text}
          </TargetLink>
        )
      )}
      {extra.map((target) => (
        <span key={`${target.kind}:${target.id}`}>
          {' · '}
          {TARGET_LABELS[target.kind]}{' '}
          <TargetLink target={target} onOpen={onOpen}>
            {target.name}
          </TargetLink>
        </span>
      ))}
    </Text>
  );
}

const OUTCOME_COLORS: Partial<Record<AgentAction['outcome'], string>> = {
  error: 'red',
  pending: 'yellow',
  approved: 'green',
  denied: 'gray',
};

/** "Waiting for approval", "Approved" or "Denied" for entries of an approval request. */
function ApprovalBadge({ entry, waiting }: { entry: AgentAction; waiting: boolean }) {
  useTranslation();
  if (entry.approval == null) {
    return null;
  }
  const label =
    entry.outcome === 'pending'
      ? waiting
        ? i18n.t('ui.activity.ActivityItem.waitingForApproval')
        : i18n.t('ui.activity.ActivityItem.askedForApproval')
      : entry.outcome === 'approved'
        ? i18n.t('ui.activity.ActivityItem.approved')
        : entry.outcome === 'denied'
          ? i18n.t('ui.activity.ActivityItem.denied')
          : i18n.t('ui.activity.ActivityItem.approvedFailed');
  return (
    <Badge
      size="sm"
      radius="sm"
      variant={waiting ? 'filled' : 'light'}
      color={OUTCOME_COLORS[entry.outcome] ?? 'gray'}
    >
      {label} #{entry.approval}
    </Badge>
  );
}

function Details({ group, now }: { group: EntryGroup; now: Date }) {
  useTranslation();
  const [entry] = group.entries;
  const args = Object.entries(entry.arguments);
  const source =
    entry.source === 'mcp'
      ? i18n.t('ui.activity.ActivityItem.mcpTool')
      : i18n.t('ui.activity.ActivityItem.restRequest');
  return (
    <div className={classes.details}>
      <dl className={classes.facts}>
        <dt>{i18n.t('ui.activity.ActivityItem.when')}</dt>
        <dd>
          {i18n.t('ui.activity.ActivityItem.took', {
            fullDateTime: fullDateTime(entry.created_at),
            duration_ms: formatDuration(entry.duration_ms),
          })}
        </dd>
        <dt>
          {entry.agent.kind === 'app'
            ? i18n.t('ui.activity.ActivityItem.connectedApp')
            : i18n.t('ui.activity.ActivityItem.apiToken')}
        </dt>
        <dd>
          {entry.agent.name}
          {entry.agent.prefix && (
            <Text span ff="monospace" size="xs" c="dimmed">
              {' '}
              {entry.agent.prefix}…
            </Text>
          )}
        </dd>
        {entry.user_agent && (
          <>
            <dt>{i18n.t('ui.activity.ActivityItem.client')}</dt>
            <dd>{entry.user_agent}</dd>
          </>
        )}
        <dt>{source}</dt>
        <dd>
          <Text span ff="monospace" size="sm">
            {actionLabel(entry)}
          </Text>
          {entry.status_code !== null &&
            i18n.t('ui.activity.ActivityItem.http', { status_code: entry.status_code })}
          {entry.writes
            ? i18n.t('ui.activity.ActivityItem.changesSomething')
            : i18n.t('ui.activity.ActivityItem.readsOnly')}
        </dd>
        <dt>{i18n.t('ui.activity.ActivityItem.arguments')}</dt>
        <dd>
          {args.length === 0
            ? i18n.t('ui.activity.ActivityItem.none')
            : args.map(([key, value]) => (
                <div key={key}>
                  <Text span ff="monospace" size="sm">
                    {key}
                  </Text>
                  : {argumentText(value)}
                </div>
              ))}
        </dd>
      </dl>
      <Text size="xs" c="dimmed" mt="xs">
        {i18n.t('ui.activity.ActivityItem.argumentsAreStoredSummarisedSizes')}
      </Text>
      {group.entries.length > 1 && (
        <>
          <Text size="sm" fw={600} mt="sm">
            {i18n.t('ui.activity.ActivityItem.callsInARow', { length: group.entries.length })}
          </Text>
          <ol className={classes.calls}>
            {group.entries.map((call) => (
              <li key={call.id}>
                <Text span size="sm" c="dimmed">
                  {timeInDay(call.created_at, now)}
                </Text>{' '}
                <Text span size="sm">
                  {call.summary} ({formatDuration(call.duration_ms)})
                </Text>
              </li>
            ))}
          </ol>
        </>
      )}
    </div>
  );
}

/** One entry of the log, or a run of the same read-only call (shown with "×n"). */
export function ActivityItem({
  group,
  fresh,
  now,
  onOpen,
  waiting = false,
}: {
  group: EntryGroup;
  fresh: boolean;
  now: Date;
  onOpen: OpenTarget;
  /** The entry's approval request still waits for the user. */
  waiting?: boolean;
}) {
  useTranslation();
  const [opened, setOpened] = useState(false);
  const [entry] = group.entries;
  const failed = entry.outcome === 'error';
  const denied = entry.outcome === 'denied';
  const ItemIcon = iconOf(entry);
  const AgentIcon = entry.agent.kind === 'app' ? IconPlugConnected : IconKey;
  const toggle = () => setOpened((open) => !open);
  const detailsId = `activity-${group.key}`;

  return (
    <li
      className={classes.item}
      data-error={failed || undefined}
      data-pending={waiting || undefined}
      data-fresh={fresh || undefined}
      aria-label={entry.summary}
    >
      {/* The whole row opens the details; the button is for keyboards and screen readers. */}
      {/* eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-static-element-interactions */}
      <div className={classes.row} onClick={toggle}>
        <ThemeIcon
          className={classes.icon}
          variant="light"
          color={OUTCOME_COLORS[entry.outcome] ?? (entry.writes ? 'brick' : 'gray')}
          radius="xl"
          size={32}
          aria-hidden
        >
          <ItemIcon size={17} />
        </ThemeIcon>
        <div className={classes.main}>
          <Summary entry={entry} onOpen={onOpen} />
          {failed && (
            <Text size="sm" c="red" className={classes.error}>
              {entry.error}
            </Text>
          )}
          {denied && entry.error && (
            <Text size="sm" c="dimmed" className={classes.error}>
              {i18n.t('ui.activity.ActivityItem.yourMessage', { error: entry.error })}
            </Text>
          )}
          <div className={classes.meta}>
            <Badge
              variant="light"
              color={entry.agent.kind === 'app' ? 'teal' : 'gray'}
              radius="sm"
              leftSection={<AgentIcon size={12} />}
              className={classes.agentBadge}
            >
              {entry.agent.name}
            </Badge>
            <ApprovalBadge entry={entry} waiting={waiting} />
            <Text span ff="monospace" size="xs" c="dimmed" className={classes.tool}>
              {actionLabel(entry)}
            </Text>
            <Text span size="xs" c="dimmed">
              {formatDuration(entry.duration_ms)}
            </Text>
            {group.entries.length > 1 && (
              <Badge
                key={entry.id}
                size="sm"
                variant="outline"
                color="gray"
                radius="sm"
                className={classes.count}
                aria-label={i18n.t('ui.activity.ActivityItem.callsInARow', {
                  length: group.entries.length,
                })}
              >
                ×{group.entries.length}
              </Badge>
            )}
          </div>
        </div>
        <div className={classes.side}>
          <Tooltip label={fullDateTime(entry.created_at)} openDelay={400}>
            <Text component="time" dateTime={entry.created_at} size="xs" c="dimmed">
              {timeInDay(entry.created_at, now)}
            </Text>
          </Tooltip>
          <ActionIcon
            variant="subtle"
            color="gray"
            size="sm"
            aria-label={
              opened
                ? i18n.t('ui.activity.ActivityItem.hideDetails')
                : i18n.t('ui.activity.ActivityItem.showDetails')
            }
            aria-expanded={opened}
            aria-controls={detailsId}
            onClick={(event) => {
              event.stopPropagation();
              toggle();
            }}
          >
            <IconChevronDown
              size={16}
              className={classes.chevron}
              data-open={opened || undefined}
            />
          </ActionIcon>
        </div>
      </div>
      <Collapse expanded={opened} id={detailsId}>
        {opened && <Details group={group} now={now} />}
      </Collapse>
    </li>
  );
}
