import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  IconChevronDown,
  IconChevronUp,
  IconSearch,
  IconSelector,
  IconTrash,
  IconX,
} from '@tabler/icons-react';
import {
  ActionIcon,
  Button,
  Checkbox,
  Group,
  NumberInput,
  Paper,
  Select,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
  UnstyledButton,
} from '@mantine/core';
import type { ColumnRelations, DataProfile, DataSample, NewStep } from '@/api/types';
import { divergingColor, useChartScheme } from '@/components/charts/colors';
import { formatCoefficient, formatCount, formatPercent } from '@/components/charts/format';
import { Heatmap } from '@/components/charts/Heatmap';
import { ScatterPlot } from '@/components/charts/ScatterPlot';
import { usePrepAccess } from '../prepAccess';
import { CorrelationsView, type CorrelationMethod } from './CorrelationsView';
import { BELOW_COLOR, clampThreshold, PartnerHistogram } from './PartnerHistogram';
import {
  countBelow,
  filterRows,
  relationRows,
  sortRows,
  strongestPairColumns,
  type RelationFilter,
  type RelationRow,
  type RelationSortKey,
  type StrengthFilter,
} from './relations';
import { numericPairs, sampleColumn } from './sampleData';
import { Note } from './shared';
import classes from './AnalysisPanel.module.css';

const PAGE = 100;
/** The heatmap of chosen columns shows at most this many. */
export const MAX_HEATMAP_COLUMNS = 60;
const DEFAULT_HEATMAP_COLUMNS = 40;

export interface RelationshipsViewProps {
  profile: DataProfile;
  sample: DataSample;
  target: string | null;
  onAddStep: (step: NewStep) => void;
  onScatter: (x: string, y: string) => void;
  method: CorrelationMethod;
  onMethodChange: (method: CorrelationMethod) => void;
}

type HeatmapChoice = 'pairs' | 'selected' | 'filtered' | 'focus' | 'block' | `group:${number}`;

/** Bar of |value| on the diverging scale, with the value. */
function StrengthBar({ value }: { value: number | null }) {
  const scheme = useChartScheme();
  if (value === null) {
    return <span className={classes.dimmed}>–</span>;
  }
  const width = 48;
  return (
    <span className={classes.association} title={`Spearman ρ = ${formatCoefficient(value)}`}>
      <svg width={width} height={8} aria-hidden="true" style={{ flexShrink: 0 }}>
        <rect width={width} height={8} rx={4} style={{ fill: 'var(--chart-track)' }} />
        <rect
          width={Math.max(3, Math.abs(value) * width)}
          height={8}
          rx={4}
          fill={divergingColor(value, scheme)}
        />
      </svg>
      <span className={classes.number}>{formatCoefficient(value)}</span>
    </span>
  );
}

const RelationRowView = memo(
  ({
    row,
    selected,
    below,
    onToggle,
    onFocus,
  }: {
    row: RelationRow;
    selected: boolean;
    below: boolean;
    onToggle: (name: string) => void;
    onFocus: (name: string) => void;
  }) => (
    <tr>
      <td>
        <Checkbox
          size="xs"
          aria-label={`Select ${row.name}`}
          checked={selected}
          onChange={() => onToggle(row.name)}
        />
      </td>
      <td>
        <span className={classes.nameCell}>
          {below && (
            <svg width={8} height={8} aria-label="No partner above the threshold" role="img">
              <circle cx={4} cy={4} r={4} style={{ fill: BELOW_COLOR }} />
            </svg>
          )}
          <button
            type="button"
            className={classes.linkButton}
            onClick={() => onFocus(row.name)}
            title={`Show the partners of ${row.name}`}
          >
            {row.name}
          </button>
        </span>
      </td>
      <td>{row.partner ?? <span className={classes.dimmed}>–</span>}</td>
      <td>
        <StrengthBar value={row.rho} />
      </td>
      <td className={classes.right}>{row.r === null ? '–' : formatCoefficient(row.r)}</td>
      <td className={classes.right}>{formatCount(row.partners)}</td>
      <td className={classes.right}>{row.group < 0 ? '–' : row.group + 1}</td>
      <td className={classes.right}>{row.target === null ? '–' : formatCoefficient(row.target)}</td>
      <td className={classes.right}>{formatPercent(row.missing)}</td>
    </tr>
  )
);

/** The partners of one column and a scatter plot with its strongest partner. */
function ColumnFocus({
  relations,
  sample,
  name,
  onScatter,
  onClose,
}: {
  relations: ColumnRelations;
  sample: DataSample;
  name: string;
  onScatter: (x: string, y: string) => void;
  onClose: () => void;
}) {
  const index = relations.columns.indexOf(name);
  const top = relations.top[index] ?? [];
  const partner = top.length ? relations.columns[top[0]] : null;
  const pairs = useMemo(() => {
    const x = sampleColumn(sample, partner);
    const y = sampleColumn(sample, name);
    return x && y ? numericPairs(x, y) : null;
  }, [sample, partner, name]);
  return (
    <Paper withBorder p="sm" radius="sm">
      <Group justify="space-between" wrap="nowrap" mb={6}>
        <Text size="sm" fw={600}>
          Partners of {name}
        </Text>
        <ActionIcon variant="subtle" color="gray" aria-label="Close" onClick={onClose}>
          <IconX size={16} />
        </ActionIcon>
      </Group>
      <SimpleGrid type="container" cols={{ base: 1, '640px': 2 }} spacing="md">
        <Stack gap={4}>
          {top.length === 0 && (
            <Text size="sm" c="dimmed">
              No partner could be computed.
            </Text>
          )}
          <ul className={classes.pairList}>
            {top.map((j, k) => {
              const other = relations.columns[j];
              const value = relations.top_rho[index]?.[k] ?? null;
              return (
                <li key={other}>
                  <UnstyledButton
                    className={classes.pair}
                    onClick={() => onScatter(other, name)}
                    aria-label={`${other}: Spearman ρ ${formatCoefficient(value)}. Open the scatter plot.`}
                  >
                    <StrengthBar value={value} />
                    <Text size="xs" truncate style={{ flex: 1 }}>
                      {other}
                    </Text>
                  </UnstyledButton>
                </li>
              );
            })}
          </ul>
          <Note>
            {relations.partners[index] ?? 0} partners at |ρ| ≥ {relations.partner_level}. A click
            opens the scatter plot.
          </Note>
        </Stack>
        {pairs && partner && pairs.x.length > 1 ? (
          <ScatterPlot
            title={`${name} against ${partner}`}
            subtitle="Its strongest partner, sample rows."
            x={pairs.x}
            y={pairs.y}
            xLabel={partner}
            yLabel={name}
            trend
            height={220}
          />
        ) : (
          <span />
        )}
      </SimpleGrid>
    </Paper>
  );
}

/**
 * How the number columns relate: the histogram of every column's strongest partner with a
 * threshold, an overview of all columns clustered into blocks, a sortable table of columns
 * (strongest partner, partner count, group …) with selection and actions, the partners of one
 * column and the heatmap of chosen columns.
 */
export function RelationshipsView({
  profile,
  sample,
  target,
  onAddStep,
  onScatter,
  method,
  onMethodChange,
}: RelationshipsViewProps) {
  const relations = profile.relations ?? null;
  const { readOnly } = usePrepAccess();
  const [threshold, setThreshold] = useState(() => relations?.suggested_level ?? 0.15);
  const [filter, setFilter] = useState<Omit<RelationFilter, 'threshold'>>({
    query: '',
    strength: 'all',
    minPartners: null,
    group: null,
  });
  const [sort, setSort] = useState<{ key: RelationSortKey; desc: boolean }>({
    key: 'order',
    desc: false,
  });
  const [limit, setLimit] = useState(PAGE);
  const [selection, setSelection] = useState<Set<string>>(() => new Set());
  const [focus, setFocus] = useState<string | null>(null);
  const [choice, setChoice] = useState<HeatmapChoice>('pairs');
  const [block, setBlock] = useState<string[]>([]);
  const [clustered, setClustered] = useState(true);
  const [added, setAdded] = useState<{ profile: DataProfile; keys: string[] }>({
    profile,
    keys: [],
  });
  const addedKeys = added.profile === profile ? added.keys : [];

  const rows = useMemo(
    () => (relations ? relationRows(relations, profile) : []),
    [relations, profile]
  );
  const names = useMemo(() => new Set(rows.map((r) => r.name)), [rows]);
  // Columns dropped by a step disappear from the selection.
  const selected = useMemo(
    () => new Set([...selection].filter((name) => names.has(name))),
    [selection, names]
  );
  const filtered = useMemo(
    () => filterRows(rows, { ...filter, threshold }),
    [rows, filter, threshold]
  );
  const sorted = useMemo(
    () => sortRows(filtered, sort.key, sort.desc, relations?.order ?? null),
    [filtered, sort, relations]
  );
  const below = useMemo(
    () => (relations ? countBelow(relations, threshold) : 0),
    [relations, threshold]
  );

  const toggle = useCallback((name: string) => {
    setSelection((current) => {
      const next = new Set(current);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  }, []);
  const focusRef = useRef<HTMLDivElement>(null);
  const showFocus = useCallback((name: string) => setFocus(name), []);
  useEffect(() => {
    if (focus) {
      focusRef.current?.scrollIntoView?.({ block: 'nearest' });
    }
  }, [focus]);

  const heatmapColumns = useMemo(() => {
    if (!relations) {
      return null;
    }
    const first = (list: string[]) => list.slice(0, MAX_HEATMAP_COLUMNS);
    if (choice === 'selected') {
      return first(rows.filter((r) => selected.has(r.name)).map((r) => r.name));
    }
    if (choice === 'filtered') {
      return first(sorted.map((r) => r.name));
    }
    if (choice === 'block') {
      return first(block);
    }
    if (choice === 'focus' && focus) {
      const index = relations.columns.indexOf(focus);
      return [focus, ...(relations.top[index] ?? []).map((j) => relations.columns[j])];
    }
    if (choice.startsWith('group:')) {
      const group = Number(choice.slice(6));
      return first(rows.filter((r) => r.group === group).map((r) => r.name));
    }
    return strongestPairColumns(relations, DEFAULT_HEATMAP_COLUMNS);
  }, [relations, choice, rows, selected, sorted, block, focus]);

  if (!relations || relations.columns.length < 2) {
    return (
      <CorrelationsView
        profile={profile}
        sample={sample}
        method={method}
        onMethodChange={onMethodChange}
        onScatter={onScatter}
        clustered={clustered}
        onClusteredChange={setClustered}
      />
    );
  }

  const n = relations.columns.length;
  const rowsText = relations.training_rows ? 'training rows' : 'rows';
  const sampled = relations.rows < relations.pool_rows;
  const groups = new Set((relations.groups ?? []).filter((g) => g >= 0)).size;
  const markAdded = (key: string) => setAdded({ profile, keys: [...addedKeys, key] });
  const add = (key: string, step: NewStep) => {
    onAddStep(step);
    markAdded(key);
  };
  const selectedNames = rows.filter((r) => selected.has(r.name)).map((r) => r.name);
  const others = [...rows.map((r) => r.name), ...relations.skipped].filter(
    (name) => !selected.has(name)
  );
  const allFilteredSelected = filtered.length > 0 && filtered.every((r) => selected.has(r.name));

  const header = (key: RelationSortKey, label: string, numeric = true, title?: string) => {
    const active = sort.key === key;
    const Icon = active ? (sort.desc ? IconChevronDown : IconChevronUp) : IconSelector;
    return (
      <th
        className={numeric ? classes.right : undefined}
        aria-sort={active ? (sort.desc ? 'descending' : 'ascending') : 'none'}
        title={title}
      >
        <button
          type="button"
          className={classes.sortButton}
          onClick={() =>
            setSort((current) =>
              current.key === key ? { key, desc: !current.desc } : { key, desc: false }
            )
          }
        >
          <span>{label}</span>
          <Icon size={12} aria-hidden="true" />
        </button>
      </th>
    );
  };

  // The overview: blocks of the clustered order.
  const overview = relations.overview;
  const order = relations.order;
  const blockColumns = (b: number) => {
    if (!overview || !order) {
      return [];
    }
    const start = overview.starts[b];
    const end = overview.starts[b + 1] ?? order.length;
    return order.slice(start, end).map((i) => relations.columns[i]);
  };
  const blockLabels = overview
    ? overview.starts.map((_, b) => {
        const columns = blockColumns(b);
        return columns.length === 1 ? columns[0] : `${columns[0]} +${columns.length - 1}`;
      })
    : [];

  const strengthOptions: { value: StrengthFilter; label: string }[] = [
    { value: 'all', label: 'All columns' },
    { value: 'below', label: `No partner at |ρ| ≥ ${threshold.toFixed(2)}` },
    { value: 'above', label: `A partner at |ρ| ≥ ${threshold.toFixed(2)}` },
  ];
  const groupOptions = [
    { value: 'none', label: 'In no group' },
    ...Array.from({ length: groups }, (_, g) => ({
      value: String(g),
      label: `Group ${g + 1} (${rows.filter((r) => r.group === g).length})`,
    })),
  ];
  const heatmapOptions = [
    { value: 'pairs', label: `Strongest pairs (${DEFAULT_HEATMAP_COLUMNS} columns)` },
    { value: 'filtered', label: `Table: matching the filter (${formatCount(filtered.length)})` },
    { value: 'selected', label: `Table: selected (${formatCount(selected.size)})` },
    ...(focus ? [{ value: 'focus', label: `${focus} and its partners` }] : []),
    ...(block.length ? [{ value: 'block', label: `Overview blocks (${block.length})` }] : []),
    ...Array.from({ length: groups }, (_, g) => ({
      value: `group:${g}`,
      label: `Group ${g + 1} (${rows.filter((r) => r.group === g).length})`,
    })),
  ];

  return (
    <Stack gap="md">
      <Text size="sm">
        How the {formatCount(n)} number and yes/no columns relate: each column’s{' '}
        <b>strongest partner</b> is the other column it correlates with most (Spearman ρ, on{' '}
        {sampled
          ? `a random sample of ${formatCount(relations.rows)} of ${formatCount(relations.pool_rows)} ${rowsText}`
          : `${formatCount(relations.rows)} ${rowsText}`}
        ). Measurements of the same thing usually have partners; a column related to no other is
        often noise.
      </Text>

      <SimpleGrid type="container" cols={{ base: 1, '900px': 2 }} spacing="md">
        <Stack gap={6}>
          <PartnerHistogram
            title="Strongest partner of each column"
            subtitle="Columns by their largest |ρ| with any other column. Drag the line to set the threshold."
            edges={relations.histogram.edges}
            counts={relations.histogram.counts}
            threshold={threshold}
            onThresholdChange={setThreshold}
            chance={relations.chance}
            below={below}
            total={n}
          />
          <Group gap="xs" wrap="wrap" align="flex-end">
            <NumberInput
              size="xs"
              label="Threshold |ρ|"
              value={threshold}
              min={0}
              max={1}
              step={0.01}
              decimalScale={2}
              w={110}
              onChange={(value) => {
                if (typeof value === 'number' && Number.isFinite(value)) {
                  setThreshold(clampThreshold(value));
                }
              }}
            />
            <Button
              size="compact-sm"
              variant="light"
              disabled={below === 0}
              onClick={() => {
                setFilter((f) => ({ ...f, strength: 'below' }));
                setLimit(PAGE);
              }}
            >
              Show the {formatCount(below)} in the table
            </Button>
            {!readOnly && (
              <Button
                size="compact-sm"
                variant="filled"
                leftSection={<IconTrash size={14} />}
                disabled={below === 0 || addedKeys.includes(`select:${threshold}`)}
                onClick={() =>
                  add(`select:${threshold}`, {
                    type: 'feature_selection',
                    params: {
                      method: 'no_partner',
                      min_partner: threshold,
                      partner_method: 'spearman',
                    },
                  })
                }
              >
                {addedKeys.includes(`select:${threshold}`)
                  ? 'Step added'
                  : `Drop the ${formatCount(below)} without a partner`}
              </Button>
            )}
          </Group>
          <Note>
            Unrelated columns reach about |ρ| = {relations.chance?.toFixed(2) ?? '–'} by chance with
            this many rows and columns (99.9 % stay below {relations.chance_high?.toFixed(2) ?? '–'}
            ); suggested threshold {relations.suggested_level.toFixed(2)}. A clear gap between two
            humps means two kinds of columns. Dropping adds a step “Select features: no strong
            partner”, learned on the training rows.
            {relations.skipped.length > 0 &&
              ` ${formatCount(relations.skipped.length)} constant columns are not shown and are dropped by that step too.`}
          </Note>
        </Stack>
        {overview && order ? (
          <Heatmap
            title="All columns, related ones together"
            subtitle={
              overview.starts.length < n
                ? `${formatCount(n)} columns in ${overview.starts.length} blocks of about ${Math.round(n / overview.starts.length)}; each cell is the mean |ρ| of two blocks. A click shows them in the heatmap below.`
                : 'Each cell is |ρ| of two columns; a click shows them in the heatmap below.'
            }
            rows={blockLabels}
            columns={blockLabels}
            values={overview.values}
            measure="mean |ρ|"
            scale="sequential"
            onCellClick={(r, c) => {
              setBlock([...new Set([...blockColumns(r), ...blockColumns(c)])]);
              setChoice('block');
            }}
            clickHint="show these columns in the heatmap"
          />
        ) : (
          <Text size="sm" c="dimmed">
            Too many columns for the overview; use the table and the heatmap of chosen columns.
          </Text>
        )}
      </SimpleGrid>

      {relations.sampled_columns && (
        <Note>
          The table has {formatCount(relations.total)} number columns; a random sample of{' '}
          {formatCount(n + relations.skipped.length)} of them is analysed here. The step “Select
          features: no strong partner” looks at all of them.
        </Note>
      )}

      <Stack gap="xs">
        <Text size="sm" fw={600}>
          Columns and their partners
        </Text>
        <Group gap="xs" wrap="wrap" align="flex-end">
          <TextInput
            size="xs"
            placeholder="Search columns"
            aria-label="Search columns"
            leftSection={<IconSearch size={14} />}
            value={filter.query}
            onChange={(event) => {
              const query = event.currentTarget.value;
              setFilter((f) => ({ ...f, query }));
              setLimit(PAGE);
            }}
            style={{ flex: '1 1 160px' }}
          />
          <Select
            size="xs"
            aria-label="Strongest partner"
            data={strengthOptions}
            value={filter.strength}
            allowDeselect={false}
            onChange={(value) => {
              setFilter((f) => ({ ...f, strength: (value as StrengthFilter) ?? 'all' }));
              setLimit(PAGE);
            }}
            comboboxProps={{ withinPortal: true }}
            w={210}
          />
          <NumberInput
            size="xs"
            aria-label={`Partners at |ρ| ≥ ${relations.partner_level} at least`}
            placeholder="Partners ≥ …"
            value={filter.minPartners ?? ''}
            min={0}
            step={1}
            allowDecimal={false}
            w={110}
            onChange={(value) =>
              setFilter((f) => ({ ...f, minPartners: typeof value === 'number' ? value : null }))
            }
          />
          {groups > 0 && (
            <Select
              size="xs"
              aria-label="Group"
              placeholder="All groups"
              data={groupOptions}
              value={
                filter.group === null ? null : filter.group < 0 ? 'none' : String(filter.group)
              }
              clearable
              onChange={(value) =>
                setFilter((f) => ({
                  ...f,
                  group: value === null ? null : value === 'none' ? -1 : Number(value),
                }))
              }
              comboboxProps={{ withinPortal: true }}
              w={160}
            />
          )}
          <Text size="xs" c="dimmed">
            {filtered.length === n
              ? `${formatCount(n)} columns`
              : `${formatCount(filtered.length)} of ${formatCount(n)} columns`}
          </Text>
        </Group>

        <Group gap="xs" wrap="wrap">
          <Checkbox
            size="xs"
            label={`Select all ${formatCount(filtered.length)} matching`}
            checked={allFilteredSelected}
            indeterminate={!allFilteredSelected && filtered.some((r) => selected.has(r.name))}
            onChange={() =>
              setSelection((current) => {
                const next = new Set(current);
                for (const row of filtered) {
                  if (allFilteredSelected) {
                    next.delete(row.name);
                  } else {
                    next.add(row.name);
                  }
                }
                return next;
              })
            }
          />
          <Text size="xs" c="dimmed">
            {formatCount(selected.size)} selected
          </Text>
          {selected.size > 0 && (
            <Button size="compact-xs" variant="subtle" onClick={() => setSelection(new Set())}>
              Clear
            </Button>
          )}
          {selected.size > 0 && (
            <Button size="compact-xs" variant="subtle" onClick={() => setChoice('selected')}>
              Show in the heatmap
            </Button>
          )}
          {!readOnly && selected.size > 0 && (
            <>
              <Button
                size="compact-xs"
                variant="light"
                color="red"
                disabled={addedKeys.includes('drop')}
                onClick={() =>
                  add('drop', { type: 'drop_columns', params: { columns: selectedNames } })
                }
              >
                {addedKeys.includes('drop')
                  ? 'Step added'
                  : `Drop selected (${formatCount(selected.size)})`}
              </Button>
              <Button
                size="compact-xs"
                variant="light"
                disabled={others.length === 0 || addedKeys.includes('keep')}
                title={`Drops the other ${formatCount(others.length)} number columns; other kinds of columns stay.`}
                onClick={() => add('keep', { type: 'drop_columns', params: { columns: others } })}
              >
                {addedKeys.includes('keep')
                  ? 'Step added'
                  : `Keep only selected (drop ${formatCount(others.length)})`}
              </Button>
            </>
          )}
        </Group>

        <div className={classes.tableScroll}>
          <table className={classes.table} aria-label="Columns and their strongest partners">
            <thead>
              <tr>
                <th aria-label="Selected" />
                {header('name', 'Column', false)}
                <th>Strongest partner</th>
                {header(
                  'strength',
                  'ρ',
                  false,
                  'Spearman ρ with the strongest partner (sorted by |ρ|)'
                )}
                {header('r', 'Pearson r', true, 'Pearson r with the same partner')}
                {header('partners', `Partners ≥ ${relations.partner_level}`, true)}
                {header('group', 'Group', true, 'Group of related columns (clustering)')}
                {target &&
                  header('target', 'Target', true, `Association with ${target} (all rows)`)}
                {!target && <th className={classes.right}>Target</th>}
                {header('missing', 'Missing')}
              </tr>
            </thead>
            <tbody>
              {sorted.slice(0, limit).map((row) => (
                <RelationRowView
                  key={row.name}
                  row={row}
                  selected={selected.has(row.name)}
                  below={row.strength < threshold}
                  onToggle={toggle}
                  onFocus={showFocus}
                />
              ))}
            </tbody>
          </table>
        </div>
        {sorted.length === 0 && (
          <Text size="sm" c="dimmed">
            No column matches the filter.
          </Text>
        )}
        {sorted.length > limit && (
          <Button
            variant="subtle"
            size="compact-sm"
            onClick={() => setLimit((l) => l + PAGE)}
            style={{ alignSelf: 'flex-start' }}
          >
            Show {formatCount(Math.min(PAGE, sorted.length - limit))} more of{' '}
            {formatCount(sorted.length - limit)}
          </Button>
        )}
        <Note>
          Sorted by “ρ” ascending, columns without a partner come first. “Order” (the default) puts
          related columns together. {formatCount(groups)} groups of related columns (average |ρ| ≥{' '}
          {relations.partner_level}). A click on a name shows its partners.
        </Note>
      </Stack>

      {focus && relations.columns.includes(focus) && (
        <div ref={focusRef}>
          <ColumnFocus
            relations={relations}
            sample={sample}
            name={focus}
            onScatter={onScatter}
            onClose={() => setFocus(null)}
          />
        </div>
      )}

      <Stack gap="xs">
        <Text size="sm" fw={600}>
          Heatmap of chosen columns
        </Text>
        <CorrelationsView
          profile={profile}
          sample={sample}
          method={method}
          onMethodChange={onMethodChange}
          onScatter={onScatter}
          columns={heatmapColumns}
          total={n}
          clustered={clustered}
          onClusteredChange={setClustered}
          chooser={
            <Select
              size="xs"
              label="Columns"
              data={heatmapOptions}
              value={choice}
              allowDeselect={false}
              onChange={(value) => value && setChoice(value as HeatmapChoice)}
              comboboxProps={{ withinPortal: true }}
              w={260}
            />
          }
        />
      </Stack>
    </Stack>
  );
}
