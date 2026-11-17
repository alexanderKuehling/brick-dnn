import { Fragment, useEffect, useState } from 'react';
import {
  IconArrowDown,
  IconArrowsSort,
  IconArrowUp,
  IconChevronDown,
  IconChevronRight,
  IconExternalLink,
  IconNote,
  IconPencil,
  IconStar,
  IconStarFilled,
} from '@tabler/icons-react';
import {
  ActionIcon,
  Alert,
  Badge,
  Center,
  Checkbox,
  Group,
  Loader,
  ScrollArea,
  Table,
  Text,
  Tooltip,
  UnstyledButton,
  VisuallyHidden,
} from '@mantine/core';
import { api, type RunDetail, type RunRow } from '@/api';
import { useTranslation } from '@/i18n';
import { StatusBadge } from '@/training/runActions';
import { TrialsTable } from '@/training/RunView';
import type { Column } from './columns';
import classes from './Experiments.module.css';

export interface RunsTableProps {
  rows: RunRow[];
  columns: Column[];
  sort: string;
  order: 'asc' | 'desc';
  onSort: (sort: string) => void;
  /** Selected for comparing (ids); without it the table has no checkboxes. */
  selected?: string[];
  maxSelected?: number;
  onToggle?: (id: string) => void;
  onOpen?: (row: RunRow) => void;
  onAnnotate?: (row: RunRow) => void;
  onStar?: (row: RunRow) => void;
  /** A caption for screen readers (the table's name). */
  label: string;
}

/** The badges after a run's name: ensemble, search. */
export function RunBadges({ row }: { row: RunRow }) {
  const { t } = useTranslation();
  return (
    <>
      {row.family === 'ensemble' && (
        <Badge size="xs" variant="outline" color="grape" tt="none">
          {t('experiments.family.ensemble')}
        </Badge>
      )}
      {row.search && (
        <Badge size="xs" variant="outline" color="cyan" tt="none">
          {t('experiments.table.search', { count: row.trials_done })}
        </Badge>
      )}
    </>
  );
}

/** The trials of a search, loaded when its row is expanded. */
function TrialsOfRun({ row }: { row: RunRow }) {
  const { t } = useTranslation();
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    api.training.runs
      .get(row.id, controller.signal)
      .then(setRun)
      .catch((err: Error) => {
        if (!controller.signal.aborted) {
          setError(err.message);
        }
      });
    return () => controller.abort();
  }, [row.id]);
  if (error) {
    return <Alert color="red">{error}</Alert>;
  }
  if (!run) {
    return (
      <Center py="sm">
        <Loader size="sm" aria-label={t('experiments.table.loadingTrials')} />
      </Center>
    );
  }
  const state = run.state;
  const tuning = run.tuning ?? state?.tuning ?? null;
  if (!tuning) {
    return null;
  }
  return (
    <TrialsTable
      tuning={tuning}
      trials={state?.trials ?? []}
      total={state?.trials_total ?? tuning.trials ?? 0}
      epochs={state?.epochs || run.epochs}
      best={state?.best_trial ?? run.best_trial?.trial ?? null}
      active={false}
      classification={row.task === 'classification'}
      segmentation={row.task === 'segmentation'}
      classical={row.family === 'classical'}
      features={row.family === 'hybrid' && !!tuning.neural_space}
    />
  );
}

/**
 * The runs table: sortable column headers (the server sorts), a checkbox per run to compare
 * it, its star, and the trials of a search below its row.
 */
export function RunsTable({
  rows,
  columns,
  sort,
  order,
  onSort,
  selected,
  maxSelected = 10,
  onToggle,
  onOpen,
  onAnnotate,
  onStar,
  label,
}: RunsTableProps) {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState<string[]>([]);
  const selecting = selected !== undefined && onToggle !== undefined;
  const span = columns.length + 2 + (selecting ? 1 : 0);

  const header = (column: { id: string; label: string; sort?: string; numeric?: boolean }) => {
    const sortKey = column.sort;
    const sorted = sortKey !== undefined && sort === sortKey;
    const ariaSort = sorted ? (order === 'asc' ? 'ascending' : 'descending') : undefined;
    return (
      <Table.Th
        key={column.id}
        aria-sort={ariaSort}
        ta={column.numeric ? 'right' : 'left'}
        className={classes.headerCell}
      >
        {sortKey ? (
          <UnstyledButton
            className={classes.sortButton}
            onClick={() => onSort(sortKey)}
            aria-label={t('experiments.table.sortBy', { label: column.label })}
          >
            <Group gap={4} wrap="nowrap" justify={column.numeric ? 'flex-end' : 'flex-start'}>
              <span>{column.label}</span>
              {sorted ? (
                order === 'asc' ? (
                  <IconArrowUp size={12} aria-hidden />
                ) : (
                  <IconArrowDown size={12} aria-hidden />
                )
              ) : (
                <IconArrowsSort size={12} aria-hidden className={classes.sortIdle} />
              )}
            </Group>
          </UnstyledButton>
        ) : (
          column.label
        )}
      </Table.Th>
    );
  };

  return (
    <ScrollArea type="auto" offsetScrollbars>
      <Table
        striped
        highlightOnHover
        withTableBorder
        fz="xs"
        verticalSpacing={6}
        className={classes.table}
        aria-label={label}
      >
        <Table.Thead>
          <Table.Tr>
            {selecting && (
              <Table.Th>
                <VisuallyHidden>{t('experiments.table.compare')}</VisuallyHidden>
              </Table.Th>
            )}
            <Table.Th>
              <VisuallyHidden>{t('experiments.table.star')}</VisuallyHidden>
            </Table.Th>
            {header({ id: 'name', label: t('experiments.column.name'), sort: 'name' })}
            {columns.map(header)}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {rows.map((row) => {
            const isSelected = selected?.includes(row.id) ?? false;
            const full = !isSelected && (selected?.length ?? 0) >= maxSelected;
            const open = expanded.includes(row.id);
            const detailsId = `trials-${row.id}`;
            return (
              <Fragment key={row.id}>
                <Table.Tr data-selected={isSelected || undefined}>
                  {selecting && (
                    <Table.Td>
                      <Tooltip
                        label={t('experiments.table.full', { count: maxSelected })}
                        disabled={!full}
                      >
                        <Checkbox
                          size="xs"
                          aria-label={t('experiments.table.select', { name: row.name })}
                          checked={isSelected}
                          disabled={full}
                          onChange={() => onToggle(row.id)}
                        />
                      </Tooltip>
                    </Table.Td>
                  )}
                  <Table.Td>
                    {onStar && row.can_annotate ? (
                      <ActionIcon
                        size="sm"
                        variant="subtle"
                        color={row.starred ? 'yellow' : 'gray'}
                        aria-pressed={row.starred}
                        aria-label={t('experiments.table.starRun', { name: row.name })}
                        onClick={() => onStar(row)}
                      >
                        {row.starred ? <IconStarFilled size={14} /> : <IconStar size={14} />}
                      </ActionIcon>
                    ) : row.starred ? (
                      <IconStarFilled
                        size={14}
                        className={classes.starStatic}
                        aria-label={t('experiments.table.starred')}
                        role="img"
                      />
                    ) : null}
                  </Table.Td>
                  <Table.Td className={classes.nameCell}>
                    <Group gap={6} wrap="nowrap">
                      {row.search && (
                        <ActionIcon
                          size="xs"
                          variant="subtle"
                          color="gray"
                          aria-expanded={open}
                          aria-controls={detailsId}
                          aria-label={t('experiments.table.showTrials', { name: row.name })}
                          onClick={() =>
                            setExpanded((list) =>
                              list.includes(row.id)
                                ? list.filter((id) => id !== row.id)
                                : [...list, row.id]
                            )
                          }
                        >
                          {open ? <IconChevronDown size={14} /> : <IconChevronRight size={14} />}
                        </ActionIcon>
                      )}
                      <Text size="xs" fw={600} truncate maw={260} title={row.name}>
                        {row.name}
                      </Text>
                      <RunBadges row={row} />
                      {row.note && (
                        <Tooltip label={row.note} multiline maw={320}>
                          <IconNote
                            size={14}
                            className={classes.noteIcon}
                            role="img"
                            aria-label={t('experiments.table.note', { note: row.note })}
                          />
                        </Tooltip>
                      )}
                      {onAnnotate && row.can_annotate && (
                        <ActionIcon
                          size="xs"
                          variant="subtle"
                          color="gray"
                          aria-label={t('experiments.table.annotate', { name: row.name })}
                          onClick={() => onAnnotate(row)}
                        >
                          <IconPencil size={12} />
                        </ActionIcon>
                      )}
                      {onOpen && (
                        <ActionIcon
                          size="xs"
                          variant="subtle"
                          color="gray"
                          aria-label={t('experiments.table.open', { name: row.name })}
                          onClick={() => onOpen(row)}
                        >
                          <IconExternalLink size={12} />
                        </ActionIcon>
                      )}
                    </Group>
                  </Table.Td>
                  {columns.map((column) => (
                    <Table.Td key={column.id} ta={column.numeric ? 'right' : 'left'}>
                      {column.id === 'status' ? (
                        <StatusBadge status={row.status} />
                      ) : column.id === 'tags' ? (
                        <Group gap={4} wrap="nowrap">
                          {(row.tags ?? []).map((tag) => (
                            <Badge key={tag} size="xs" variant="light" color="gray" tt="none">
                              {tag}
                            </Badge>
                          ))}
                        </Group>
                      ) : (
                        <span className={classes.cell}>{column.text(row)}</span>
                      )}
                    </Table.Td>
                  ))}
                </Table.Tr>
                {open && (
                  <Table.Tr id={detailsId}>
                    <Table.Td colSpan={span}>
                      <TrialsOfRun row={row} />
                    </Table.Td>
                  </Table.Tr>
                )}
              </Fragment>
            );
          })}
        </Table.Tbody>
      </Table>
    </ScrollArea>
  );
}
