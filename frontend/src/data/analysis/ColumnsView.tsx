import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { IconChevronDown, IconChevronUp, IconSearch, IconSelector } from '@tabler/icons-react';
import { Button, Group, Select, Stack, Text, TextInput } from '@mantine/core';
import type { ColumnKind, ColumnProfile, DataProfile } from '@/api/types';
import {
  formatCoefficient,
  formatCount,
  formatDate,
  formatNumber,
  formatPercent,
} from '@/components/charts/format';
import { MiniHistogram } from '@/components/charts/MiniHistogram';
import { analysisNote } from './sampleData';
import { KIND_LABELS, KIND_ORDER, KindIcon, Note } from './shared';
import classes from './AnalysisPanel.module.css';

type SortKey =
  | 'order'
  | 'name'
  | 'kind'
  | 'missing'
  | 'unique'
  | 'mean'
  | 'median'
  | 'min'
  | 'max'
  | 'association';

/** Short names of the association measures, e.g. next to a value. */
export const SHORT_METHOD = { pearson: '|r|', eta: 'η', cramers_v: 'V' } as const;

export const ASSOCIATION_LABELS = {
  pearson: '|Pearson r|',
  eta: 'Correlation ratio η',
  cramers_v: "Cramér's V",
} as const;

const PAGE = 100;

export interface ColumnsViewProps {
  profile: DataProfile;
  target: string | null;
  onShowColumn: (column: string) => void;
}

function statValue(column: ColumnProfile, key: 'mean' | 'median' | 'min' | 'max'): number | null {
  return column.stats?.[key] ?? null;
}

function sortValue(column: ColumnProfile, key: SortKey, index: number): number | string | null {
  switch (key) {
    case 'order':
      return index;
    case 'name':
      return column.name.toLowerCase();
    case 'kind':
      return KIND_ORDER.indexOf(column.kind);
    case 'missing':
      return column.missing;
    case 'unique':
      return column.unique;
    case 'association':
      return column.target_association?.value ?? null;
    default:
      return column.kind === 'datetime' ? null : statValue(column, key);
  }
}

/** A statistic of a column, dates as dates. */
function formatStat(column: ColumnProfile, value: number | null): string {
  if (value === null) {
    return '–';
  }
  return column.kind === 'datetime' ? formatDate(value) : formatNumber(value);
}

export function AssociationBar({
  value,
  method,
}: {
  value: number;
  method: keyof typeof ASSOCIATION_LABELS;
}) {
  const width = 56;
  return (
    <span
      className={classes.association}
      title={`${ASSOCIATION_LABELS[method]} = ${formatCoefficient(value)}`}
    >
      <svg width={width} height={8} aria-hidden="true" style={{ flexShrink: 0 }}>
        <rect width={width} height={8} rx={4} style={{ fill: 'var(--chart-track)' }} />
        {value > 0 && (
          <rect
            width={Math.max(4, value * width)}
            height={8}
            rx={4}
            style={{ fill: 'var(--chart-series-1)' }}
          />
        )}
      </svg>
      <span className={classes.number}>{formatCoefficient(value)}</span>
      <span className={classes.dimmed}>{SHORT_METHOD[method]}</span>
    </span>
  );
}

function Distribution({ column }: { column: ColumnProfile }) {
  if (column.histogram && column.histogram.counts.length > 0) {
    return <MiniHistogram counts={column.histogram.counts} width={72} height={20} />;
  }
  if (column.top && column.top.length > 0 && column.kind !== 'text') {
    const counts = column.top.slice(0, 12).map((t) => t.count);
    return <MiniHistogram counts={counts} width={72} height={20} />;
  }
  return (
    <span className={classes.dimmed}>
      {column.kind === 'text' && column.mean_length !== null
        ? `⌀ ${formatNumber(column.mean_length, 3)} chars`
        : '–'}
    </span>
  );
}

/** One table row; plain elements and memoised, so hundreds of columns stay fast. */
const ColumnRow = memo(
  ({
    column,
    rows,
    isTarget,
    withAssociation,
    onShow,
  }: {
    column: ColumnProfile;
    rows: number;
    isTarget: boolean;
    withAssociation: boolean;
    onShow: (name: string) => void;
  }) => {
    return (
      <tr>
        <td>
          <span className={classes.nameCell}>
            <KindIcon kind={column.kind} />
            <button
              type="button"
              className={classes.linkButton}
              onClick={() => onShow(column.name)}
              title={`Show the distribution of ${column.name}`}
            >
              {column.name}
            </button>
            {isTarget && <span className={classes.dimmed}>(target)</span>}
          </span>
        </td>
        <td>{KIND_LABELS[column.kind]}</td>
        <td className={column.missing ? classes.right : `${classes.right} ${classes.dimmed}`}>
          <span title={`${formatCount(column.missing)} rows`}>
            {formatPercent(rows ? column.missing / rows : 0)}
          </span>
        </td>
        <td className={classes.right}>{formatCount(column.unique)}</td>
        {(['mean', 'median', 'min', 'max'] as const).map((key) => (
          <td key={key} className={classes.right}>
            {formatStat(column, statValue(column, key))}
          </td>
        ))}
        <td>
          <Distribution column={column} />
        </td>
        {withAssociation && (
          <td>
            {column.target_association ? (
              <AssociationBar
                value={column.target_association.value}
                method={column.target_association.method}
              />
            ) : (
              <span className={classes.dimmed}>–</span>
            )}
          </td>
        )}
      </tr>
    );
  }
);

/** All columns at a glance: a sortable, searchable table; a click opens the distribution. */
export function ColumnsView({ profile, target, onShowColumn }: ColumnsViewProps) {
  const [query, setQuery] = useState('');
  const [kind, setKind] = useState<ColumnKind | null>(null);
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: 'order', desc: false });
  const [limit, setLimit] = useState(PAGE);
  const hasAssociation = profile.columns.some((c) => c.target_association);

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = profile.columns
      .map((column, index) => ({ column, index }))
      .filter(
        ({ column }) =>
          (!q || column.name.toLowerCase().includes(q)) && (!kind || column.kind === kind)
      );
    return filtered.sort((a, b) => {
      const va = sortValue(a.column, sort.key, a.index);
      const vb = sortValue(b.column, sort.key, b.index);
      // Columns without a value go last in both directions.
      if (va === null || vb === null) {
        return va === vb ? a.index - b.index : va === null ? 1 : -1;
      }
      const order = va < vb ? -1 : va > vb ? 1 : a.index - b.index;
      return sort.desc ? -order : order;
    });
  }, [profile.columns, query, kind, sort]);

  // A stable callback keeps the memoised rows from re-rendering.
  const showRef = useRef(onShowColumn);
  useEffect(() => {
    showRef.current = onShowColumn;
  });
  const show = useCallback((name: string) => showRef.current(name), []);

  const header = (key: SortKey, label: string, numeric = true) => {
    const active = sort.key === key;
    const Icon = active ? (sort.desc ? IconChevronDown : IconChevronUp) : IconSelector;
    return (
      <th
        className={numeric ? classes.right : undefined}
        aria-sort={active ? (sort.desc ? 'descending' : 'ascending') : 'none'}
      >
        <button
          type="button"
          className={classes.sortButton}
          onClick={() =>
            setSort((current) =>
              current.key === key
                ? { key, desc: !current.desc }
                : { key, desc: numeric && key !== 'kind' && key !== 'order' }
            )
          }
        >
          <span>{label}</span>
          <Icon size={12} aria-hidden="true" />
        </button>
      </th>
    );
  };

  const kindsPresent = KIND_ORDER.filter((k) => profile.columns.some((c) => c.kind === k));

  return (
    <Stack gap="sm">
      <Group gap="xs" wrap="wrap">
        <TextInput
          size="xs"
          placeholder="Search columns"
          aria-label="Search columns"
          leftSection={<IconSearch size={14} />}
          value={query}
          onChange={(event) => {
            setQuery(event.currentTarget.value);
            setLimit(PAGE);
          }}
          style={{ flex: '1 1 180px' }}
        />
        <Select
          size="xs"
          aria-label="Column type"
          placeholder="All types"
          data={kindsPresent.map((k) => ({ value: k, label: KIND_LABELS[k] }))}
          value={kind}
          onChange={(value) => setKind((value as ColumnKind | null) ?? null)}
          clearable
          w={150}
          comboboxProps={{ withinPortal: true }}
        />
        <Text size="xs" c="dimmed">
          {rows.length === profile.columns.length
            ? `${formatCount(rows.length)} columns`
            : `${formatCount(rows.length)} of ${formatCount(profile.columns.length)} columns`}
        </Text>
      </Group>

      <div className={classes.tableScroll}>
        <table className={classes.table}>
          <thead>
            <tr>
              {header('name', 'Column', false)}
              {header('kind', 'Type', false)}
              {header('missing', 'Missing')}
              {header('unique', 'Unique')}
              {header('mean', 'Mean')}
              {header('median', 'Median')}
              {header('min', 'Min')}
              {header('max', 'Max')}
              <th>Distribution</th>
              {hasAssociation && header('association', 'Target')}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, limit).map(({ column }) => (
              <ColumnRow
                key={column.name}
                column={column}
                rows={profile.n_rows}
                isTarget={column.name === target}
                withAssociation={hasAssociation}
                onShow={show}
              />
            ))}
          </tbody>
        </table>
      </div>
      {rows.length === 0 && (
        <Text size="sm" c="dimmed">
          No column matches the search.
        </Text>
      )}
      {rows.length > limit && (
        <Button
          variant="subtle"
          size="compact-sm"
          onClick={() => setLimit((l) => l + PAGE)}
          style={{ alignSelf: 'flex-start' }}
        >
          Show {formatCount(Math.min(PAGE, rows.length - limit))} more of{' '}
          {formatCount(rows.length - limit)}
        </Button>
      )}
      <Note>
        Statistics are exact (all {formatCount(profile.n_rows)} rows).
        {hasAssociation &&
          ` Target association (0 = none, 1 = perfect): |Pearson r| between numbers, correlation ratio η between numbers and categories, Cramér's V between categories and classes. ${analysisNote(profile)}`}
      </Note>
    </Stack>
  );
}
