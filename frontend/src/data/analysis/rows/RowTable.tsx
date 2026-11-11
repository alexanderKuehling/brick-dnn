import { useMemo, useState } from 'react';
import { IconArrowDown, IconArrowUp } from '@tabler/icons-react';
import { Checkbox } from '@mantine/core';
import type { RowAnalysis } from '@/api/types';
import { formatCount, formatNumber } from '@/components/charts/format';
import { rowRange } from '../../table/virtual';
import { sortRows, type RowSort, type RowSortKey } from './rowsData';
import classes from './RowsView.module.css';

const ROW_HEIGHT = 30;
const VIEWPORT = 380;
const OVERSCAN = 10;
const SPLIT_LABELS = { train: 'Train', val: 'Validation', test: 'Test' } as const;

export interface RowTableProps {
  analysis: RowAnalysis;
  /** Indices (into analysis.rows) to show, already filtered. */
  indices: number[];
  sort: RowSort;
  onSortChange: (sort: RowSort) => void;
  selected: ReadonlySet<number>;
  onSelectedChange: (selected: Set<number>) => void;
  /** The row whose similar rows are shown. */
  active: number | null;
  onActivate: (index: number) => void;
}

/**
 * The analysed rows with their statistics: sortable by every column, with selection boxes.
 * Only the rows in view are rendered, so tens of thousands of rows stay fast.
 */
export function RowTable({
  analysis,
  indices,
  sort,
  onSortChange,
  selected,
  onSelectedChange,
  active,
  onActivate,
}: RowTableProps) {
  const [scrollTop, setScrollTop] = useState(0);
  const sorted = useMemo(() => sortRows(analysis, indices, sort), [analysis, indices, sort]);
  const range = rowRange(scrollTop, VIEWPORT, ROW_HEIGHT, sorted.length, OVERSCAN);
  const { rows } = analysis;
  const showId = analysis.id_column !== null && rows.id !== null;
  const showSplit = rows.split !== null;
  const showTarget = rows.target !== null;

  const header = (key: RowSortKey, label: string, numeric = false) => {
    const current = sort.key === key;
    const Icon = sort.descending ? IconArrowDown : IconArrowUp;
    return (
      <th
        className={numeric ? classes.numeric : undefined}
        aria-sort={current ? (sort.descending ? 'descending' : 'ascending') : 'none'}
      >
        <button
          type="button"
          className={classes.sortButton}
          onClick={() =>
            onSortChange({
              key,
              descending: current
                ? !sort.descending
                : key !== 'position' && key !== 'spread' && key !== 'nn_ratio',
            })
          }
        >
          {label}
          {current && <Icon size={12} aria-hidden />}
        </button>
      </th>
    );
  };

  const toggle = (index: number) => {
    const next = new Set(selected);
    if (next.has(index)) {
      next.delete(index);
    } else {
      next.add(index);
    }
    onSelectedChange(next);
  };
  const allShown = sorted.length > 0 && sorted.every((i) => selected.has(i));
  const columns = 7 + Number(showId) + Number(showSplit) + Number(showTarget);

  return (
    <div
      className={classes.tableScroll}
      style={{ maxHeight: VIEWPORT }}
      onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
    >
      <table
        className={classes.table}
        aria-label="Rows with their statistics"
        aria-rowcount={sorted.length + 1}
      >
        <thead>
          <tr>
            <th>
              <Checkbox
                size="xs"
                aria-label="Select all rows shown"
                checked={allShown}
                indeterminate={!allShown && sorted.some((i) => selected.has(i))}
                onChange={() => {
                  const next = new Set(selected);
                  for (const i of sorted) {
                    if (allShown) {
                      next.delete(i);
                    } else {
                      next.add(i);
                    }
                  }
                  onSelectedChange(next);
                }}
              />
            </th>
            {header('position', 'Row', true)}
            {showId && <th>{analysis.id_column}</th>}
            {showSplit && header('split', 'Split')}
            {showTarget &&
              header('target', analysis.target ?? 'Target', analysis.task === 'regression')}
            {header('spread', 'Spread', true)}
            {header('missing', 'Missing', true)}
            {header('nn_ratio', 'NN ratio', true)}
            <th>Nearest row</th>
            {header('group_size', 'Group', true)}
          </tr>
        </thead>
        <tbody>
          {range.start > 0 && (
            <tr data-spacer aria-hidden style={{ height: range.start * ROW_HEIGHT }}>
              <td colSpan={columns} />
            </tr>
          )}
          {sorted.slice(range.start, range.end).map((i) => {
            const neighbor = rows.neighbor[i];
            const target = rows.target?.[i];
            return (
              <tr
                key={i}
                data-active={active === i || undefined}
                onClick={() => onActivate(i)}
                data-testid="row-table-row"
              >
                <td onClick={(event) => event.stopPropagation()}>
                  <Checkbox
                    size="xs"
                    aria-label={`Select row ${rows.position[i]}`}
                    checked={selected.has(i)}
                    onChange={() => toggle(i)}
                  />
                </td>
                <td className={classes.numeric}>{formatCount(rows.position[i])}</td>
                {showId && <td>{rows.id?.[i] ?? '–'}</td>}
                {showSplit && <td>{SPLIT_LABELS[rows.split![i]]}</td>}
                {showTarget && (
                  <td className={analysis.task === 'regression' ? classes.numeric : undefined}>
                    {target === null || target === undefined
                      ? '–'
                      : typeof target === 'number'
                        ? formatNumber(target, 4)
                        : target}
                  </td>
                )}
                <td className={classes.numeric}>{fmt(rows.spread[i])}</td>
                <td className={classes.numeric}>{formatCount(rows.missing[i])}</td>
                <td className={classes.numeric}>{fmt(rows.nn_ratio[i])}</td>
                <td>
                  {neighbor >= 0
                    ? rows.id && showId
                      ? `${rows.id[neighbor] ?? '–'}`
                      : `Row ${formatCount(rows.position[neighbor])}`
                    : '–'}
                </td>
                <td className={classes.numeric}>
                  {rows.group_size[i] > 1 ? `#${rows.group[i] + 1} · ${rows.group_size[i]}` : '–'}
                </td>
              </tr>
            );
          })}
          {range.end < sorted.length && (
            <tr
              data-spacer
              aria-hidden
              style={{ height: (sorted.length - range.end) * ROW_HEIGHT }}
            >
              <td colSpan={columns} />
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function fmt(value: number | null): string {
  return value === null || value === undefined ? '–' : formatNumber(value, 3);
}
