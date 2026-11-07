import { useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { IconX } from '@tabler/icons-react';
import { Button, Group, Select, Text } from '@mantine/core';
import type { ColumnKind, DataProfile, DataSample, NewStep } from '@/api/types';
import { ColumnDialog, type ColumnDialogState } from './ColumnDialog';
import { ColumnHeader, type SortDirection } from './ColumnHeader';
import { formatCell, sampleCaption } from './format';
import { nextSort, sortOrder } from './sort';
import { columnOffsets, columnRange, rowRange } from './virtual';
import classes from './DataTable.module.css';

export interface DataTableProps {
  sample: DataSample;
  profile: DataProfile | null;
  target: string | null;
  onAddStep: (step: NewStep) => void;
  onSetTarget: (column: string) => void;
  /** Open a column in the Analysis tab. */
  onShowColumn?: (column: string) => void;
}

const ROW_HEIGHT = 30;
const HEADER_HEIGHT = 84;
const INDEX_WIDTH = 68;
const SPLIT_WIDTH = 64;
const COLUMN_WIDTHS: Record<ColumnKind, number> = {
  numeric: 136,
  boolean: 120,
  datetime: 176,
  categorical: 160,
  text: 240,
};
const OVERSCAN_ROWS = 12;
const OVERSCAN_COLUMNS = 2;
/** Scroll positions are rounded to these steps (fewer renders; covered by the overscan). */
const ROW_BLOCK = ROW_HEIGHT * 4;
const COLUMN_BLOCK = 100;
/** Used before the container is measured (first render, tests). */
const FALLBACK = { width: 1280, height: 720 };

const SPLIT_CELL_STYLE = { left: INDEX_WIDTH };
const SPLIT_LABELS = { train: 'train', val: 'val', test: 'test' } as const;
const SPLIT_TITLES = { train: 'Training row', val: 'Validation row', test: 'Test row' } as const;

interface Viewport {
  top: number;
  left: number;
  width: number;
  height: number;
}

/** The sample rows of the prepared data as a scrollable grid with column quick actions. */
export function DataTable({
  sample,
  profile,
  target,
  onAddStep,
  onSetTarget,
  onShowColumn,
}: DataTableProps) {
  const captionId = useId();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [viewport, setViewport] = useState<Viewport>({ top: 0, left: 0, width: 0, height: 0 });
  const [sort, setSort] = useState<{ column: string; direction: SortDirection } | null>(null);
  const [dialog, setDialog] = useState<ColumnDialogState>(null);

  const measure = () => {
    const element = scrollRef.current;
    if (!element) {
      return;
    }
    const next = {
      top: Math.floor(element.scrollTop / ROW_BLOCK) * ROW_BLOCK,
      left: Math.floor(element.scrollLeft / COLUMN_BLOCK) * COLUMN_BLOCK,
      width: element.clientWidth,
      height: element.clientHeight,
    };
    setViewport((current) =>
      current.top === next.top &&
      current.left === next.left &&
      current.width === next.width &&
      current.height === next.height
        ? current
        : next
    );
  };

  useLayoutEffect(() => {
    measure();
    const element = scrollRef.current;
    if (!element || typeof ResizeObserver === 'undefined') {
      return undefined;
    }
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const columns = sample.columns;
  const profiles = useMemo(
    () => new Map((profile?.columns ?? []).map((column) => [column.name, column])),
    [profile]
  );
  // Without a profile, the missing share comes from the sample.
  const sampleMissing = useMemo(
    () =>
      new Map(
        profile
          ? []
          : sample.columns.map((column) => [
              column.name,
              column.values.filter((value) => value === null).length,
            ])
      ),
    [sample, profile]
  );
  const sortColumn = sort ? columns.find((column) => column.name === sort.column) : undefined;
  // A sort by a column that a new preview no longer has is dropped.
  const activeSort = sortColumn ? sort : null;
  const order = useMemo(
    () => (sortColumn && sort ? sortOrder(sortColumn.values, sort.direction) : null),
    [sortColumn, sort]
  );

  const hasSplit = sample.split !== null;
  const fixedWidth = INDEX_WIDTH + (hasSplit ? SPLIT_WIDTH : 0);
  const widths = columns.map((column) => COLUMN_WIDTHS[column.kind] ?? 150);
  const offsets = columnOffsets(widths);
  const totalWidth = offsets[offsets.length - 1];

  const width = viewport.width || FALLBACK.width;
  const height = viewport.height || FALLBACK.height;
  const rows = rowRange(
    viewport.top,
    height - HEADER_HEIGHT + ROW_BLOCK,
    ROW_HEIGHT,
    sample.rows,
    OVERSCAN_ROWS
  );
  const cols = columnRange(
    offsets,
    viewport.left,
    width - fixedWidth + COLUMN_BLOCK,
    OVERSCAN_COLUMNS
  );
  const visibleColumns = columns.slice(cols.start, cols.end);
  const leftSpacer = offsets[cols.start] ?? 0;
  const rightSpacer = totalWidth - (offsets[cols.end] ?? totalWidth);
  const fixedCount = hasSplit ? 2 : 1;
  const spanAll =
    fixedCount + visibleColumns.length + (leftSpacer > 0 ? 1 : 0) + (rightSpacer > 0 ? 1 : 0);
  const topPad = rows.start * ROW_HEIGHT;
  const bottomPad = (sample.rows - rows.end) * ROW_HEIGHT;
  const tableRows = profile?.n_rows ?? sample.n_total;
  const names = columns.map((column) => column.name);

  const missingOf = (name: string) => {
    const columnProfile = profiles.get(name);
    return columnProfile
      ? { missing: columnProfile.missing, rows: tableRows }
      : { missing: sampleMissing.get(name) ?? 0, rows: sample.rows };
  };

  const jumpTo = (name: string | null) => {
    const index = name === null ? -1 : names.indexOf(name);
    const element = scrollRef.current;
    if (index >= 0 && element) {
      element.scrollLeft = offsets[index];
      measure();
    }
  };

  return (
    <div className={classes.root}>
      <Group justify="space-between" gap="xs" wrap="wrap" className={classes.toolbar}>
        <Text size="xs" c="dimmed" id={captionId}>
          {sampleCaption(sample.rows, sample.n_total)} · {columns.length}{' '}
          {columns.length === 1 ? 'column' : 'columns'}
        </Text>
        <Group gap="xs" wrap="wrap" miw={0} maw="100%">
          {activeSort && (
            <Button
              size="compact-xs"
              maw="100%"
              variant="light"
              rightSection={<IconX size={12} />}
              onClick={() => setSort(null)}
              aria-label={`Sorted by ${activeSort.column} (${activeSort.direction}). Clear sorting`}
            >
              Sorted by {activeSort.column}
            </Button>
          )}
          {columns.length > 6 && (
            <Select
              size="xs"
              w={180}
              aria-label="Go to column"
              placeholder="Go to column"
              data={names}
              value={null}
              searchable
              nothingFoundMessage="No matching column"
              comboboxProps={{ withinPortal: true, shadow: 'md' }}
              onChange={jumpTo}
            />
          )}
        </Group>
      </Group>

      <div
        ref={scrollRef}
        className={classes.scroll}
        role="region"
        aria-labelledby={captionId}
        // A scrollable region must be focusable, so keyboard users can scroll it.
        // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
        tabIndex={0}
        onScroll={measure}
      >
        <table
          className={classes.table}
          style={{ width: fixedWidth + totalWidth }}
          aria-rowcount={sample.rows + 1}
          aria-colcount={columns.length + fixedCount}
          aria-label="Data sample"
        >
          <thead>
            <tr aria-rowindex={1} style={{ height: HEADER_HEIGHT }}>
              <th
                scope="col"
                aria-colindex={1}
                className={`${classes.corner} ${classes.indexHead}`}
                style={{ width: INDEX_WIDTH }}
              >
                Row
              </th>
              {hasSplit && (
                <th
                  scope="col"
                  aria-colindex={2}
                  className={`${classes.corner} ${classes.splitHead}`}
                  style={{ width: SPLIT_WIDTH, left: INDEX_WIDTH }}
                >
                  Split
                </th>
              )}
              {leftSpacer > 0 && (
                <th aria-hidden className={classes.spacerHead} style={{ width: leftSpacer }} />
              )}
              {visibleColumns.map((column, i) => {
                const index = cols.start + i;
                const { missing, rows: base } = missingOf(column.name);
                return (
                  <ColumnHeader
                    key={column.name}
                    column={column}
                    profile={profiles.get(column.name)}
                    rows={base}
                    missing={missing}
                    width={widths[index]}
                    colIndex={index + fixedCount + 1}
                    isTarget={column.name === target}
                    sort={sort?.column === column.name ? sort.direction : null}
                    onSort={() => {
                      const direction = nextSort(
                        sort?.column === column.name ? sort.direction : null
                      );
                      setSort(direction ? { column: column.name, direction } : null);
                    }}
                    onAddStep={onAddStep}
                    onSetTarget={onSetTarget}
                    onShowColumn={onShowColumn}
                    onRename={() => setDialog({ mode: 'rename', column })}
                    onCast={() => setDialog({ mode: 'cast', column })}
                  />
                );
              })}
              {rightSpacer > 0 && (
                <th aria-hidden className={classes.spacerHead} style={{ width: rightSpacer }} />
              )}
            </tr>
          </thead>
          <tbody>
            {topPad > 0 && (
              <tr aria-hidden style={{ height: topPad }}>
                <td colSpan={spanAll} className={classes.pad} />
              </tr>
            )}
            {Array.from({ length: rows.end - rows.start }, (_, i) => {
              const position = rows.start + i;
              const row = order ? order[position] : position;
              const split = sample.split?.[row];
              return (
                <tr key={row} aria-rowindex={position + 2} className={classes.row}>
                  <th scope="row" aria-colindex={1} className={classes.indexCell}>
                    {(sample.index[row] ?? row) + 1}
                  </th>
                  {hasSplit && (
                    <td aria-colindex={2} className={classes.splitCell} style={SPLIT_CELL_STYLE}>
                      {split && (
                        <span
                          className={classes.split}
                          data-split={split}
                          title={SPLIT_TITLES[split]}
                        >
                          {SPLIT_LABELS[split]}
                        </span>
                      )}
                    </td>
                  )}
                  {leftSpacer > 0 && <td aria-hidden className={classes.spacer} />}
                  {visibleColumns.map((column, j) => {
                    const cell = formatCell(column.values[row], column.kind);
                    return (
                      <td
                        key={column.name}
                        aria-colindex={cols.start + j + fixedCount + 1}
                        className={classes.cell}
                        data-align={cell.align}
                        data-missing={cell.missing || undefined}
                        data-target={column.name === target || undefined}
                        title={
                          !cell.missing && (column.kind === 'text' || cell.text.length > 18)
                            ? cell.text
                            : undefined
                        }
                      >
                        {cell.missing ? (
                          <span className={classes.missing}>missing</span>
                        ) : (
                          cell.text
                        )}
                      </td>
                    );
                  })}
                  {rightSpacer > 0 && <td aria-hidden className={classes.spacer} />}
                </tr>
              );
            })}
            {bottomPad > 0 && (
              <tr aria-hidden style={{ height: bottomPad }}>
                <td colSpan={spanAll} className={classes.pad} />
              </tr>
            )}
            {sample.rows === 0 && (
              <tr>
                <td colSpan={spanAll} className={classes.empty}>
                  No rows left. A filter or another step removed all rows.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <ColumnDialog
        state={dialog}
        names={names}
        onClose={() => setDialog(null)}
        onAddStep={onAddStep}
      />
    </div>
  );
}
