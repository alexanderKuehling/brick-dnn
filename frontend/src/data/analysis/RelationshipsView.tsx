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
import { i18n, Trans, useTranslation } from '@/i18n';
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
const SEPARATOR = '\u0000';

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
  useTranslation();
  const scheme = useChartScheme();
  if (value === null) {
    return <span className={classes.dimmed}>–</span>;
  }
  const width = 48;
  return (
    <span
      className={classes.association}
      title={i18n.t('ui.data.RelationshipsView.spearman', { value: formatCoefficient(value) })}
    >
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
          aria-label={i18n.t('ui.data.RelationshipsView.select', { name: row.name })}
          checked={selected}
          onChange={() => onToggle(row.name)}
        />
      </td>
      <td>
        <span className={classes.nameCell}>
          {below && (
            <svg
              width={8}
              height={8}
              aria-label={i18n.t('ui.data.RelationshipsView.noPartnerAboveTheThreshold')}
              role="img"
            >
              <circle cx={4} cy={4} r={4} style={{ fill: BELOW_COLOR }} />
            </svg>
          )}
          <button
            type="button"
            className={classes.linkButton}
            onClick={() => onFocus(row.name)}
            title={i18n.t('ui.data.RelationshipsView.showThePartnersOf', { name: row.name })}
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

/** All columns in the clustered order, in blocks; a click picks the columns of two blocks. */
const OverviewHeatmap = memo(
  ({ relations, onPick }: { relations: ColumnRelations; onPick: (columns: string[]) => void }) => {
    useTranslation();
    const { overview, order } = relations;
    if (!overview || !order) {
      return (
        <Text size="sm" c="dimmed">
          {i18n.t('ui.data.RelationshipsView.tooManyColumnsForThe')}
        </Text>
      );
    }
    const n = relations.columns.length;
    const blockColumns = (b: number) => {
      const end = overview.starts[b + 1] ?? order.length;
      return order.slice(overview.starts[b], end).map((i) => relations.columns[i]);
    };
    const labels = overview.starts.map((_, b) => {
      const columns = blockColumns(b);
      return columns.length === 1 ? columns[0] : `${columns[0]} +${columns.length - 1}`;
    });
    return (
      <Heatmap
        title={i18n.t('ui.data.RelationshipsView.allColumnsRelatedOnesTogether')}
        subtitle={
          overview.starts.length < n
            ? i18n.t('ui.data.RelationshipsView.columnsInBlocksOfAbout', {
                n: formatCount(n),
                length: overview.starts.length,
                round: Math.round(n / overview.starts.length),
              })
            : i18n.t('ui.data.RelationshipsView.eachCellIsOfTwo')
        }
        rows={labels}
        columns={labels}
        values={overview.values}
        measure="mean |ρ|"
        scale="sequential"
        onCellClick={(r, c) => onPick([...new Set([...blockColumns(r), ...blockColumns(c)])])}
        clickHint={i18n.t('ui.data.RelationshipsView.showTheseColumnsInTheHeatmap')}
      />
    );
  }
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
  useTranslation();
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
          {i18n.t('ui.data.RelationshipsView.partnersOf', { name })}
        </Text>
        <ActionIcon
          variant="subtle"
          color="gray"
          aria-label={i18n.t('ui.data.RelationshipsView.close')}
          onClick={onClose}
        >
          <IconX size={16} />
        </ActionIcon>
      </Group>
      <SimpleGrid type="container" cols={{ base: 1, '640px': 2 }} spacing="md">
        <Stack gap={4}>
          {top.length === 0 && (
            <Text size="sm" c="dimmed">
              {i18n.t('ui.data.RelationshipsView.noPartnerCouldBeComputed')}
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
                    fz="xs"
                    onClick={() => onScatter(other, name)}
                    aria-label={i18n.t('ui.data.RelationshipsView.spearmanOpenTheScatterPlot', {
                      other,
                      value: formatCoefficient(value),
                    })}
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
            {i18n.t('ui.data.RelationshipsView.partnersAtAClickOpens', {
              value: relations.partners[index] ?? 0,
              partner_level: relations.partner_level,
            })}
          </Note>
        </Stack>
        {pairs && partner && pairs.x.length > 1 ? (
          <ScatterPlot
            title={i18n.t('ui.data.RelationshipsView.against', { name, partner })}
            subtitle={i18n.t('ui.data.RelationshipsView.itsStrongestPartnerSampleRows')}
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
  useTranslation();
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

  // A key of the chosen names: the heatmap only recomputes when they change (not on every
  // move of the threshold).
  const heatmapKey = useMemo(() => {
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
  }, [relations, choice, rows, selected, sorted, block, focus])?.join(SEPARATOR);
  const heatmapColumns = useMemo(
    () =>
      heatmapKey === undefined || heatmapKey === null
        ? null
        : heatmapKey.split(SEPARATOR).filter(Boolean),
    [heatmapKey]
  );
  const pickBlocks = useCallback((columns: string[]) => {
    setBlock(columns);
    setChoice('block');
  }, []);

  const groups = useMemo(
    () => new Set((relations?.groups ?? []).filter((g) => g >= 0)).size,
    [relations]
  );
  const groupSizes = useMemo(() => {
    const sizes = new Array<number>(groups).fill(0);
    rows.forEach((r) => {
      if (r.group >= 0) {
        sizes[r.group] += 1;
      }
    });
    return sizes;
  }, [rows, groups]);
  const filteredCount = filtered.length;
  const chooser = useMemo(() => {
    const options = [
      {
        value: 'pairs',
        label: i18n.t('ui.data.RelationshipsView.strongestPairsColumns', {
          DEFAULT_HEATMAP_COLUMNS,
        }),
      },
      {
        value: 'filtered',
        label: i18n.t('ui.data.RelationshipsView.tableMatchingTheFilter', {
          filteredCount: formatCount(filteredCount),
        }),
      },
      {
        value: 'selected',
        label: i18n.t('ui.data.RelationshipsView.tableSelected', {
          size: formatCount(selected.size),
        }),
      },
      ...(focus
        ? [{ value: 'focus', label: i18n.t('ui.data.RelationshipsView.andItsPartners', { focus }) }]
        : []),
      ...(block.length
        ? [
            {
              value: 'block',
              label: i18n.t('ui.data.RelationshipsView.overviewBlocks', { length: block.length }),
            },
          ]
        : []),
      ...groupSizes.map((size, g) => ({
        value: `group:${g}`,
        label: i18n.t('ui.data.RelationshipsView.group', { value: g + 1, size }),
      })),
    ];
    return (
      <Select
        size="xs"
        label={i18n.t('ui.data.RelationshipsView.columns')}
        data={options}
        value={choice}
        allowDeselect={false}
        onChange={(value) => value && setChoice(value as HeatmapChoice)}
        comboboxProps={{ withinPortal: true }}
        w={260}
      />
    );
  }, [filteredCount, selected.size, focus, block.length, groupSizes, choice]);

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
  const sampled = relations.rows < relations.pool_rows;
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

  const strengthOptions: { value: StrengthFilter; label: string }[] = [
    { value: 'all', label: i18n.t('ui.data.RelationshipsView.allColumns') },
    {
      value: 'below',
      label: i18n.t('ui.data.RelationshipsView.noPartnerAt', { threshold: threshold.toFixed(2) }),
    },
    {
      value: 'above',
      label: i18n.t('ui.data.RelationshipsView.aPartnerAt', { threshold: threshold.toFixed(2) }),
    },
  ];
  const groupOptions = [
    { value: 'none', label: i18n.t('ui.data.RelationshipsView.inNoGroup') },
    ...Array.from({ length: groups }, (_, g) => ({
      value: String(g),
      label: i18n.t('ui.data.RelationshipsView.group2', { value: g + 1, value2: groupSizes[g] }),
    })),
  ];

  return (
    <Stack gap="md">
      <Text size="sm">
        <Trans
          i18nKey="ui.data.RelationshipsView.howTheNumberAndYes"
          values={{
            n: formatCount(n),
            value: i18n.t(
              `ui.data.RelationshipsView.${sampled ? 'sampleOf' : 'all'}${relations.training_rows ? 'TrainingRows' : 'Rows'}`,
              { rows: formatCount(relations.rows), pool: formatCount(relations.pool_rows) }
            ),
          }}
          components={[<b />]}
        />
      </Text>

      <SimpleGrid type="container" cols={{ base: 1, '900px': 2 }} spacing="md">
        <Stack gap={6}>
          <PartnerHistogram
            title={i18n.t('ui.data.RelationshipsView.strongestPartnerOfEachColumn')}
            subtitle={i18n.t('ui.data.RelationshipsView.columnsByTheirLargestWith')}
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
              label={i18n.t('ui.data.RelationshipsView.threshold')}
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
              {i18n.t('ui.data.RelationshipsView.showTheInTheTable', { below: formatCount(below) })}
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
                  ? i18n.t('ui.data.RelationshipsView.stepAdded')
                  : i18n.t('ui.data.RelationshipsView.dropTheWithoutAPartner', {
                      below: formatCount(below),
                    })}
              </Button>
            )}
          </Group>
          <Note>
            {i18n.t('ui.data.RelationshipsView.unrelatedColumnsReachAbout')}{' '}
            {relations.chance?.toFixed(2) ?? '–'}{' '}
            {i18n.t('ui.data.RelationshipsView.byChanceWithThisMany')}{' '}
            {relations.chance_high?.toFixed(2) ?? '–'}
            {i18n.t('ui.data.RelationshipsView.suggestedThreshold')}{' '}
            {relations.suggested_level.toFixed(2)}
            {i18n.t('ui.data.RelationshipsView.aClearGapBetweenTwo')}
            {relations.skipped.length > 0 &&
              i18n.t('ui.data.RelationshipsView.constantColumnsAreNotShown', {
                length: formatCount(relations.skipped.length),
              })}
          </Note>
        </Stack>
        <OverviewHeatmap relations={relations} onPick={pickBlocks} />
      </SimpleGrid>

      {relations.sampled_columns && (
        <Note>
          {i18n.t('ui.data.RelationshipsView.theTableHasNumberColumns', {
            total: formatCount(relations.total),
            value: formatCount(n + relations.skipped.length),
          })}
        </Note>
      )}

      <Stack gap="xs">
        <Text size="sm" fw={600}>
          {i18n.t('ui.data.RelationshipsView.columnsAndTheirPartners')}
        </Text>
        <Group gap="xs" wrap="wrap" align="flex-end">
          <TextInput
            size="xs"
            placeholder={i18n.t('ui.data.RelationshipsView.searchColumns')}
            aria-label={i18n.t('ui.data.RelationshipsView.searchColumns')}
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
            aria-label={i18n.t('ui.data.RelationshipsView.strongestPartner')}
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
            aria-label={i18n.t('ui.data.RelationshipsView.partnersAtAtLeast', {
              partner_level: relations.partner_level,
            })}
            placeholder={i18n.t('ui.data.RelationshipsView.partners')}
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
              aria-label={i18n.t('ui.data.RelationshipsView.group3')}
              placeholder={i18n.t('ui.data.RelationshipsView.allGroups')}
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
              ? i18n.t('ui.data.RelationshipsView.columns2', { n: formatCount(n) })
              : i18n.t('ui.data.RelationshipsView.ofColumns', {
                  length: formatCount(filtered.length),
                  n: formatCount(n),
                })}
          </Text>
        </Group>

        <Group gap="xs" wrap="wrap">
          <Checkbox
            size="xs"
            label={i18n.t('ui.data.RelationshipsView.selectAllMatching', {
              length: formatCount(filtered.length),
            })}
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
            {i18n.t('ui.data.RelationshipsView.selected', { size: formatCount(selected.size) })}
          </Text>
          {selected.size > 0 && (
            <Button size="compact-xs" variant="subtle" onClick={() => setSelection(new Set())}>
              {i18n.t('ui.data.RelationshipsView.clear')}
            </Button>
          )}
          {selected.size > 0 && (
            <Button size="compact-xs" variant="subtle" onClick={() => setChoice('selected')}>
              {i18n.t('ui.data.RelationshipsView.showInTheHeatmap')}
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
                  ? i18n.t('ui.data.RelationshipsView.stepAdded')
                  : i18n.t('ui.data.RelationshipsView.dropSelected', {
                      size: formatCount(selected.size),
                    })}
              </Button>
              <Button
                size="compact-xs"
                variant="light"
                disabled={others.length === 0 || addedKeys.includes('keep')}
                title={i18n.t('ui.data.RelationshipsView.dropsTheOtherNumberColumns', {
                  length: formatCount(others.length),
                })}
                onClick={() => add('keep', { type: 'drop_columns', params: { columns: others } })}
              >
                {addedKeys.includes('keep')
                  ? i18n.t('ui.data.RelationshipsView.stepAdded')
                  : i18n.t('ui.data.RelationshipsView.keepOnlySelectedDrop', {
                      length: formatCount(others.length),
                    })}
              </Button>
            </>
          )}
        </Group>

        <div className={classes.tableScroll}>
          <table
            className={classes.table}
            aria-label={i18n.t('ui.data.RelationshipsView.columnsAndTheirStrongestPartners')}
          >
            <thead>
              <tr>
                <th aria-label={i18n.t('ui.data.RelationshipsView.selected2')} />
                {header('name', i18n.t('ui.data.RelationshipsView.column'), false)}
                <th>{i18n.t('ui.data.RelationshipsView.strongestPartner')}</th>
                {header(
                  'strength',
                  'ρ',
                  false,
                  i18n.t('ui.data.RelationshipsView.spearmanWithTheStrongestPartner')
                )}
                {header(
                  'r',
                  'Pearson r',
                  true,
                  i18n.t('ui.data.RelationshipsView.pearsonWithTheSamePartner')
                )}
                {header(
                  'partners',
                  i18n.t('ui.data.RelationshipsView.partnersAtLeast', {
                    level: relations.partner_level,
                  }),
                  true
                )}
                {header(
                  'group',
                  i18n.t('ui.data.RelationshipsView.group'),
                  true,
                  i18n.t('ui.data.RelationshipsView.groupOfRelatedColumns')
                )}
                {target &&
                  header(
                    'target',
                    i18n.t('ui.data.RelationshipsView.target'),
                    true,
                    i18n.t('ui.data.RelationshipsView.associationWith', { target })
                  )}
                {!target && (
                  <th className={classes.right}>{i18n.t('ui.data.RelationshipsView.target')}</th>
                )}
                {header('missing', i18n.t('ui.data.RelationshipsView.missing'))}
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
            {i18n.t('ui.data.RelationshipsView.noColumnMatchesTheFilter')}
          </Text>
        )}
        {sorted.length > limit && (
          <Button
            variant="subtle"
            size="compact-sm"
            onClick={() => setLimit((l) => l + PAGE)}
            style={{ alignSelf: 'flex-start' }}
          >
            {i18n.t('ui.data.RelationshipsView.showMoreOf', {
              min: formatCount(Math.min(PAGE, sorted.length - limit)),
              value: formatCount(sorted.length - limit),
            })}
          </Button>
        )}
        <Note>
          {i18n.t('ui.data.RelationshipsView.sortedByAscendingColumnsWithout', {
            groups: formatCount(groups),
            partner_level: relations.partner_level,
          })}
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
          {i18n.t('ui.data.RelationshipsView.heatmapOfChosenColumns')}
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
          chooser={chooser}
        />
      </Stack>
    </Stack>
  );
}
