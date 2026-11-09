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
          title={`${column.name} (${KIND_LABELS[column.kind]}). Click to sort the sample.`}
        >
          <span className={classes.headNameText}>{column.name}</span>
          <VisuallyHidden>
            , {KIND_LABELS[column.kind]}
            {sort ? `, sorted ${sort}` : ''}. Sort the sample
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
                aria-label={`Actions for column ${column.name}`}
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
                  Target column: what the model learns to predict. It is not encoded, filled or
                  dropped like a feature.
                </Menu.Label>
              )}
              {!readOnly && (
                <Menu.Item
                  leftSection={<IconTarget size={14} />}
                  disabled={isTarget}
                  onClick={() => onSetTarget(column.name)}
                >
                  {isTarget ? 'This is the target' : 'Use as target'}
                </Menu.Item>
              )}
              {onShowColumn && (
                <Menu.Item
                  leftSection={<IconChartHistogram size={14} />}
                  onClick={() => onShowColumn(column.name)}
                >
                  Show in Analysis
                </Menu.Item>
              )}
              {!readOnly && (
                <>
                  <Menu.Divider />
                  <Menu.Label>{isTarget ? 'Add a step for the target' : 'Add a step'}</Menu.Label>
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
                    Rename…
                  </Menu.Item>
                  <Menu.Item leftSection={<IconTransform size={14} />} onClick={onCast}>
                    Change type…
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
            Target
          </Badge>
        )}
        <span data-level={share > 0.4 ? 'high' : share > 0 ? 'some' : 'none'}>
          {missing > 0 ? `${formatShare(missing, rows)} missing` : 'complete'}
        </span>
        {profile && (column.kind === 'categorical' || column.kind === 'text') && (
          <span> · {formatCount(profile.unique)} unique</span>
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
  if (!profile) {
    return null;
  }
  if ((column.kind === 'numeric' || column.kind === 'datetime') && profile.histogram) {
    return <MiniHistogram counts={profile.histogram.counts} width={width} height={18} />;
  }
  if (column.kind === 'boolean' && profile.top) {
    const ones = profile.top.find((entry) => entry.value === '1')?.count ?? 0;
    return <span>{formatShare(ones, profile.count)} true</span>;
  }
  if (column.kind === 'categorical' && profile.top && profile.top.length > 0) {
    const text = profile.top
      .slice(0, 3)
      .map((entry) => `${entry.value} ${formatShare(entry.count, profile.count)}`)
      .join(' · ');
    return <span title={text}>{text}</span>;
  }
  if (column.kind === 'text' && profile.mean_length !== null) {
    return <span>avg. {Math.round(profile.mean_length)} characters</span>;
  }
  return null;
}
