import {
  IconArrowDown,
  IconArrowUp,
  IconBinary,
  IconCalendarEvent,
  IconChartHistogram,
  IconChevronDown,
  IconCursorText,
  IconCut,
  IconDropletQuestion,
  IconMathFunction,
  IconRowRemove,
  IconTarget,
  IconTransform,
  IconTrash,
  type Icon,
} from '@tabler/icons-react';
import { ActionIcon, Badge, Menu, UnstyledButton, VisuallyHidden } from '@mantine/core';
import type { ColumnInfo, ColumnProfile, NewStep } from '@/api/types';
import { MiniHistogram } from '@/components/charts/MiniHistogram';
import { i18n, useTranslation } from '@/i18n';
import { usePrepAccess } from '../prepAccess';
import { columnStepActions, type ColumnActionKey } from './columnActions';
import { formatCount, formatShare } from './format';
import { KIND_LABELS, KindIcon } from './kinds';
import classes from './DataTable.module.css';

const ACTION_ICONS: Record<ColumnActionKey, Icon> = {
  impute: IconDropletQuestion,
  clip: IconCut,
  log: IconMathFunction,
  onehot: IconBinary,
  date_parts: IconCalendarEvent,
  drop: IconTrash,
  drop_rows: IconRowRemove,
};

export type SortDirection = 'ascending' | 'descending';

interface ColumnHeaderProps {
  column: ColumnInfo;
  profile: ColumnProfile | undefined;
  /** Rows of the whole table (for the missing share), from the profile or the sample. */
  rows: number;
  /** Missing values: whole table if profiled, else in the sample. */
  missing: number;
  width: number;
  colIndex: number;
  isTarget: boolean;
  sort: SortDirection | null;
  onSort: () => void;
  onAddStep: (step: NewStep) => void;
  onSetTarget: (column: string) => void;
  onShowColumn?: (column: string) => void;
  onRename: () => void;
  onCast: () => void;
}

/** Name, kind, missing share and distribution of a column, with its quick actions. */
export function ColumnHeader({
  column,
  profile,
  rows,
  missing,
  width,
  colIndex,
  isTarget,
  sort,
  onSort,
  onAddStep,
  onSetTarget,
  onShowColumn,
  onRename,
  onCast,
}: ColumnHeaderProps) {
  useTranslation();
  // Viewers of a shared preparation cannot add steps or change the target.
  const { readOnly } = usePrepAccess();
  const actions = columnStepActions(column, profile, missing, isTarget);
  const drop = actions.find((action) => action.key === 'drop');
  const others = actions.filter((action) => action.key !== 'drop');
  const share = missing / Math.max(rows, 1);
  const SortIcon = sort === 'descending' ? IconArrowDown : IconArrowUp;

  return (
    <th
      scope="col"
      aria-colindex={colIndex}
      aria-sort={sort ?? undefined}
      className={classes.head}
      data-target={isTarget || undefined}
      style={{ width, minWidth: width, maxWidth: width }}
    >
      <div className={classes.headTop}>
        <KindIcon kind={column.kind} />
        <UnstyledButton
          className={classes.headName}
          onClick={onSort}
          title={i18n.t('ui.data.ColumnHeader.clickToSortTheSample', {
            name: column.name,
            value: KIND_LABELS[column.kind],
          })}
        >
          <span className={classes.headNameText}>{column.name}</span>
          <VisuallyHidden>
            {i18n.t('ui.data.ColumnHeader.sortTheSample', {
              value: KIND_LABELS[column.kind],
              value2: sort ? i18n.t('ui.data.ColumnHeader.sorted', { sort }) : '',
            })}
          </VisuallyHidden>
          {sort && <SortIcon size={12} aria-hidden className={classes.sortIcon} />}
        </UnstyledButton>
        {(!readOnly || onShowColumn) && (
          <Menu position="bottom-end" shadow="md" width={270} withinPortal>
            <Menu.Target>
              <ActionIcon
                size="sm"
                variant="subtle"
                color="gray"
                aria-label={i18n.t('ui.data.ColumnHeader.actionsForColumn', { name: column.name })}
              >
                <IconChevronDown size={14} />
              </ActionIcon>
            </Menu.Target>
            <Menu.Dropdown>
              <Menu.Label>
                {column.name} · {KIND_LABELS[column.kind]}
              </Menu.Label>
              {isTarget && (
                <Menu.Label className={classes.menuNote}>
                  {i18n.t('ui.data.ColumnHeader.targetColumnWhatTheModel')}
                </Menu.Label>
              )}
              {!readOnly && (
                <Menu.Item
                  leftSection={<IconTarget size={14} />}
                  disabled={isTarget}
                  onClick={() => onSetTarget(column.name)}
                >
                  {isTarget
                    ? i18n.t('ui.data.ColumnHeader.thisIsTheTarget')
                    : i18n.t('ui.data.ColumnHeader.useAsTarget')}
                </Menu.Item>
              )}
              {onShowColumn && (
                <Menu.Item
                  leftSection={<IconChartHistogram size={14} />}
                  onClick={() => onShowColumn(column.name)}
                >
                  {i18n.t('ui.data.ColumnHeader.showInAnalysis')}
                </Menu.Item>
              )}
              {!readOnly && (
                <>
                  <Menu.Divider />
                  <Menu.Label>
                    {isTarget
                      ? i18n.t('ui.data.ColumnHeader.addAStepForThe')
                      : i18n.t('ui.data.ColumnHeader.addAStep')}
                  </Menu.Label>
                  {others.map((action) => {
                    const ActionIconComponent = ACTION_ICONS[action.key];
                    return (
                      <Menu.Item
                        key={action.key}
                        leftSection={<ActionIconComponent size={14} />}
                        onClick={() => onAddStep(action.step)}
                      >
                        {action.label}
                      </Menu.Item>
                    );
                  })}
                  <Menu.Item leftSection={<IconCursorText size={14} />} onClick={onRename}>
                    {i18n.t('ui.data.ColumnHeader.rename')}
                  </Menu.Item>
                  <Menu.Item leftSection={<IconTransform size={14} />} onClick={onCast}>
                    {i18n.t('ui.data.ColumnHeader.changeType')}
                  </Menu.Item>
                  {drop && (
                    <>
                      <Menu.Divider />
                      <Menu.Item
                        color="red"
                        leftSection={<IconTrash size={14} />}
                        onClick={() => onAddStep(drop.step)}
                      >
                        {drop.label}
                      </Menu.Item>
                    </>
                  )}
                </>
              )}
            </Menu.Dropdown>
          </Menu>
        )}
      </div>
      <div className={classes.headMeta}>
        {isTarget && (
          <Badge size="xs" variant="light" color="green" className={classes.targetBadge}>
            {i18n.t('ui.data.ColumnHeader.target')}
          </Badge>
        )}
        <span data-level={share > 0.4 ? 'high' : share > 0 ? 'some' : 'none'}>
          {missing > 0
            ? i18n.t('ui.data.ColumnHeader.missing', { missing: formatShare(missing, rows) })
            : i18n.t('ui.data.ColumnHeader.complete')}
        </span>
        {profile && (column.kind === 'categorical' || column.kind === 'text') && (
          <span>
            {' '}
            {i18n.t('ui.data.ColumnHeader.unique', { unique: formatCount(profile.unique) })}
          </span>
        )}
      </div>
      <div className={classes.headDist}>
        <Distribution column={column} profile={profile} width={width - 20} />
      </div>
    </th>
  );
}

function Distribution({
  column,
  profile,
  width,
}: {
  column: ColumnInfo;
  profile: ColumnProfile | undefined;
  width: number;
}) {
  useTranslation();
  if (!profile) {
    return null;
  }
  if ((column.kind === 'numeric' || column.kind === 'datetime') && profile.histogram) {
    return <MiniHistogram counts={profile.histogram.counts} width={width} height={18} />;
  }
  if (column.kind === 'boolean' && profile.top) {
    const ones = profile.top.find((entry) => entry.value === '1')?.count ?? 0;
    return (
      <span>{i18n.t('ui.data.ColumnHeader.true', { ones: formatShare(ones, profile.count) })}</span>
    );
  }
  if (column.kind === 'categorical' && profile.top && profile.top.length > 0) {
    const text = profile.top
      .slice(0, 3)
      .map((entry) => `${entry.value} ${formatShare(entry.count, profile.count)}`)
      .join(' · ');
    return <span title={text}>{text}</span>;
  }
  if (column.kind === 'text' && profile.mean_length !== null) {
    return (
      <span>
        {i18n.t('ui.data.ColumnHeader.avgCharacters', { round: Math.round(profile.mean_length) })}
      </span>
    );
  }
  return null;
}
