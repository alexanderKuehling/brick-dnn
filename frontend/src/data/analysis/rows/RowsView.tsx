import { useEffect, useMemo, useRef, useState } from 'react';
import { IconAlertTriangle, IconInfoCircle } from '@tabler/icons-react';
import {
  Alert,
  Button,
  Group,
  Loader,
  MultiSelect,
  SegmentedControl,
  Select,
  Stack,
  Text,
  TextInput,
} from '@mantine/core';
import type {
  NewStep,
  RowAnalysis,
  RowAnalysisRequest,
  RowFeatureMode,
  SimilarRows,
  SimilarRowsRequest,
  SplitSettings,
} from '@/api/types';
import { formatCount, formatNumber, formatPercent } from '@/components/charts/format';
import { StatTile } from '@/components/charts/StatTile';
import { usePrepAccess } from '../../prepAccess';
import { PairsView } from './PairsView';
import { ProjectionView, type ProjectionColor } from './ProjectionView';
import {
  countBelow,
  dropRowsStep,
  filterRows,
  freeName,
  type RowFilter,
  type RowSort,
} from './rowsData';
import { RowTable } from './RowTable';
import { SimilarPanel } from './SimilarPanel';
import { ThresholdHistogram } from './ThresholdHistogram';
import classes from './RowsView.module.css';

/** How the Rows view loads its data (the page knows the preparation and its steps). */
export interface RowsSource {
  /** Changes whenever a new preview arrived: the analysis is loaded again. */
  revision: unknown;
  load: (body: RowAnalysisRequest, signal: AbortSignal) => Promise<RowAnalysis>;
  similar: (body: SimilarRowsRequest, signal: AbortSignal) => Promise<SimilarRows>;
  /** Change split settings (split by the new group column); missing when read-only. */
  onChangeSplit?: (split: Partial<SplitSettings>) => void;
}

export interface RowsViewProps {
  source: RowsSource;
  /** Number columns of the previewed data (for "Columns I choose"). */
  numberColumns: string[];
  /** All column names (a free name for the group column). */
  columns: string[];
  onAddStep: (step: NewStep) => void;
  readOnly?: boolean;
}

const FEATURE_MODES: { value: RowFeatureMode; label: string }[] = [
  { value: 'auto', label: 'Automatic' },
  { value: 'partnered', label: 'Partnered' },
  { value: 'all', label: 'All number columns' },
  { value: 'columns', label: 'I choose' },
];

const FILTERS: { value: RowFilter; label: string }[] = [
  { value: 'all', label: 'All rows' },
  { value: 'atypical', label: 'Unusually typical' },
  { value: 'twins', label: 'With a near twin' },
  { value: 'group', label: 'Group of the chosen row' },
  { value: 'selected', label: 'Selected' },
];

const REASONS: Record<string, string> = {
  target_or_group: 'target / group column',
  id: 'ID column',
  constant: 'constant',
  no_partner: 'no strong partner',
  limit: 'over the limit',
};

const RELOAD_DELAY = 400;

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : 'The rows could not be analysed.';
}

/**
 * Analysis → Rows: how typical every row is, how close its nearest row is, groups of near
 * twins, close vs random pairs, a projection of the rows and a table to explore and pick rows
 * — with steps to act on what the user sees.
 */
export function RowsView({
  source,
  numberColumns,
  columns,
  onAddStep,
  readOnly: readOnlyProp = false,
}: RowsViewProps) {
  const access = usePrepAccess();
  const readOnly = readOnlyProp || access.readOnly;
  const [mode, setMode] = useState<RowFeatureMode>('auto');
  const [chosen, setChosen] = useState<string[]>([]);
  const [twinOverride, setTwinOverride] = useState<number | null>(null);
  const [atypicalOverride, setAtypicalOverride] = useState<number | null>(null);
  const [requestedTwin, setRequestedTwin] = useState<number | null>(null);
  const [analysis, setAnalysis] = useState<RowAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<RowFilter>('all');
  const [search, setSearch] = useState('');
  const [sort, setSort] = useState<RowSort>({ key: 'position', descending: false });
  const [selected, setSelected] = useState<Set<number>>(() => new Set());
  const [active, setActive] = useState<number | null>(null);
  const [compareWith, setCompareWith] = useState<number | null>(null);
  const [similar, setSimilar] = useState<SimilarRows | null>(null);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [similarError, setSimilarError] = useState<string | null>(null);
  const [color, setColor] = useState<ProjectionColor>('structure');
  const [attempt, setAttempt] = useState(0);
  const loadRef = useRef(source.load);
  loadRef.current = source.load;
  const similarRef = useRef(source.similar);
  similarRef.current = source.similar;

  const features = useMemo(
    () => (mode === 'columns' ? { mode, columns: chosen } : { mode }),
    [mode, chosen]
  );
  const waitingForColumns = mode === 'columns' && chosen.length === 0;

  // The twin threshold decides the groups on the server: ask again a moment after it changed.
  useEffect(() => {
    const timer = setTimeout(() => setRequestedTwin(twinOverride), RELOAD_DELAY);
    return () => clearTimeout(timer);
  }, [twinOverride]);

  // New data or other columns: the rows may be others, so the selection starts again.
  const dataKey = useRef<unknown>(null);
  useEffect(() => {
    if (waitingForColumns) {
      setLoading(false);
      return undefined;
    }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    loadRef
      .current({ features, twin_threshold: requestedTwin }, controller.signal)
      .then((result) => {
        const key = [source.revision, features];
        const changed =
          dataKey.current === null ||
          (dataKey.current as unknown[])[0] !== key[0] ||
          (dataKey.current as unknown[])[1] !== key[1];
        dataKey.current = key;
        if (changed) {
          setSelected(new Set());
          setActive(null);
          setCompareWith(null);
          setSimilar(null);
        }
        setAnalysis(result);
        setLoading(false);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(errorText(reason));
          setLoading(false);
        }
      });
    return () => controller.abort();
  }, [source.revision, features, requestedTwin, waitingForColumns, attempt]);

  // The similar rows of the chosen row.
  useEffect(() => {
    if (active === null || !analysis) {
      return undefined;
    }
    const controller = new AbortController();
    setSimilarLoading(true);
    setSimilarError(null);
    similarRef
      .current(
        { features, twin_threshold: requestedTwin, row: active, other: compareWith },
        controller.signal
      )
      .then((result) => {
        setSimilar(result);
        setSimilarLoading(false);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setSimilarError(errorText(reason));
          setSimilarLoading(false);
        }
      });
    return () => controller.abort();
    // `analysis` changes with every threshold; the rows stay the same.
  }, [active, compareWith, features, source.revision]);

  const thresholds = useMemo(
    () => ({
      twins: twinOverride ?? analysis?.thresholds.twins ?? 0.5,
      atypical: atypicalOverride ?? analysis?.thresholds.atypical ?? 0.25,
    }),
    [twinOverride, atypicalOverride, analysis]
  );
  const shown = useMemo(
    () =>
      analysis
        ? filterRows(analysis, filter, thresholds, {
            search,
            selected,
            group: active !== null ? analysis.rows.group[active] : null,
          })
        : [],
    [analysis, filter, thresholds, search, selected, active]
  );
  const randomRatios = useMemo(() => analysis?.random_pairs.ratio ?? [], [analysis]);

  const featureControl = (
    <Stack gap={6}>
      <Group gap="sm" align="flex-end" wrap="wrap">
        <div>
          <Text size="xs" fw={500} mb={4}>
            Compare rows on
          </Text>
          <SegmentedControl
            size="xs"
            aria-label="Compare rows on"
            data={FEATURE_MODES}
            value={mode}
            onChange={(value) => setMode(value as RowFeatureMode)}
          />
        </div>
        {mode === 'columns' && (
          <MultiSelect
            size="xs"
            label="Columns"
            placeholder="Choose number columns"
            data={numberColumns}
            value={chosen}
            onChange={setChosen}
            searchable
            clearable
            maxValues={1000}
            comboboxProps={{ withinPortal: true }}
            style={{ flex: 1, minWidth: 220 }}
          />
        )}
      </Group>
      {analysis && !waitingForColumns && <FeatureNote analysis={analysis} />}
    </Stack>
  );

  if (waitingForColumns) {
    return (
      <Stack gap="md">
        {featureControl}
        <Text size="sm" c="dimmed">
          Choose the number columns to compare the rows on.
        </Text>
      </Stack>
    );
  }
  if (!analysis) {
    return (
      <Stack gap="md">
        {featureControl}
        {error ? (
          <Alert color="red" icon={<IconAlertTriangle size={18} />} title="Cannot analyse the rows">
            <Text size="sm">{error}</Text>
            <Button
              size="compact-sm"
              variant="light"
              color="red"
              mt="xs"
              onClick={() => setAttempt((n) => n + 1)}
            >
              Try again
            </Button>
          </Alert>
        ) : (
          <Group gap="xs">
            <Loader size="sm" />
            <Text size="sm" c="dimmed">
              Comparing every row with every other row …
            </Text>
          </Group>
        )}
      </Stack>
    );
  }

  const atypicalCount = countBelow(analysis.rows.spread, thresholds.atypical);
  const twinCount = countBelow(analysis.rows.nn_ratio, thresholds.twins);
  const { groups } = analysis;
  const groupName = freeName(columns);
  const featureParams = {
    features: analysis.features.mode,
    columns: analysis.features.mode === 'columns' ? chosen : [],
  };
  const removeAtypical = () =>
    onAddStep({
      type: 'remove_atypical',
      params: { ...featureParams, side: 'low', threshold: thresholds.atypical },
    });
  const findGroups = () => {
    onAddStep({
      type: 'find_groups',
      params: { ...featureParams, threshold: thresholds.twins, name: groupName },
    });
    source.onChangeSplit?.({ group_by: groupName });
  };
  const matching = shown;

  return (
    <Stack gap="md" data-loading={loading || undefined} style={{ opacity: loading ? 0.7 : 1 }}>
      <Text size="sm" c="dimmed">
        Each row compared with all others: how typical its values are, and how close its nearest row
        is. Unusually typical rows (every value near its median) are often filler or synthetic rows;
        rows with a near twin are often the same person or object measured twice. Drag the
        thresholds, look at the rows, then act with a step.
      </Text>
      {featureControl}
      {error && (
        <Alert color="red" p="xs">
          <Text size="sm">{error}</Text>
        </Alert>
      )}
      {analysis.sampled && (
        <Alert color="blue" p="xs" icon={<IconInfoCircle size={16} />}>
          <Text size="sm">
            A random sample of {formatCount(analysis.analysed)} of {formatCount(analysis.n_total)}{' '}
            rows is analysed (the limit for {formatCount(analysis.features.count)} columns). A row's
            twin may be outside the sample, so near twins are undercounted.
          </Text>
        </Alert>
      )}

      <div className={classes.tiles}>
        <StatTile
          label="Rows analysed"
          value={formatCount(analysis.analysed)}
          hint={`${formatNumber((analysis.stats_ms ?? 0) / 1000, 2)} s`}
        />
        <StatTile
          label="Unusually typical"
          value={formatCount(atypicalCount)}
          hint={`spread below ${formatNumber(thresholds.atypical, 3)}`}
        />
        <StatTile
          label="With a near twin"
          value={formatCount(twinCount)}
          hint={`NN ratio below ${formatNumber(thresholds.twins, 3)}`}
        />
        <StatTile
          label="Groups of near twins"
          value={formatCount(groups.count)}
          hint={
            groups.count ? `${formatCount(groups.rows)} rows, up to ${groups.largest} each` : 'none'
          }
        />
        {groups.held_out > 0 && (
          <StatTile
            label="Validation/test rows with a twin in training"
            value={formatPercent(groups.held_out_with_twin / groups.held_out)}
            hint={
              analysis.split_group_by
                ? `split keeps '${analysis.split_group_by}' together`
                : `${formatCount(groups.held_out_with_twin)} of ${formatCount(groups.held_out)}`
            }
          />
        )}
      </div>

      <div className={classes.grid}>
        <ThresholdHistogram
          title="How typical is each row? (spread)"
          description="Median distance of the row's values from their column medians, in robust standard deviations. Ordinary rows: about 0.67. A separate cluster near 0: rows whose every value sits at the median."
          values={analysis.rows.spread}
          threshold={thresholds.atypical}
          onThresholdChange={setAtypicalOverride}
          suggested={analysis.suggested.atypical}
          cap={4}
          xLabel="spread"
          belowText={(n) => `${formatCount(n)} ${n === 1 ? 'row is' : 'rows are'} below`}
          actions={
            <Group gap="xs">
              <Button size="compact-xs" variant="default" onClick={() => setFilter('atypical')}>
                Show them
              </Button>
              {!readOnly && (
                <Button
                  size="compact-xs"
                  variant="light"
                  disabled={atypicalCount === 0}
                  onClick={removeAtypical}
                >
                  Remove {formatCount(atypicalCount)} rows (step)
                </Button>
              )}
            </Group>
          }
        />
        <ThresholdHistogram
          title="How close is each row's nearest row? (NN ratio)"
          description="Distance to the nearest other row divided by the distance of a typical pair of rows. Below about 0.5: a near twin. The dashed outline shows random pairs."
          values={analysis.rows.nn_ratio}
          baseline={randomRatios}
          baselineLabel="random pairs (same number of rows)"
          threshold={thresholds.twins}
          onThresholdChange={setTwinOverride}
          suggested={analysis.suggested.twins}
          cap={2}
          xLabel="NN ratio"
          belowText={(n) => `${formatCount(n)} ${n === 1 ? 'row has' : 'rows have'} a near twin`}
          actions={
            <Group gap="xs">
              <Button size="compact-xs" variant="default" onClick={() => setFilter('twins')}>
                Show them
              </Button>
              {!readOnly && (
                <Button
                  size="compact-xs"
                  variant="light"
                  disabled={twinCount === 0}
                  onClick={findGroups}
                >
                  Find groups &amp; split by them (step)
                </Button>
              )}
            </Group>
          }
        />
      </div>
      {analysis.suggested.atypical === null && analysis.suggested.twins === null && (
        <Text size="xs" c="dimmed">
          Neither histogram has a clear gap: no cluster of unusually typical rows and no separate
          group of near twins was found. The thresholds are starting points to explore.
        </Text>
      )}

      <div className={classes.grid}>
        <PairsView analysis={analysis} threshold={thresholds.twins} />
        <ProjectionView
          analysis={analysis}
          thresholds={thresholds}
          color={color}
          onColorChange={setColor}
          active={active}
          onActivate={(index) => {
            setActive(index);
            setCompareWith(null);
          }}
        />
      </div>

      <Stack gap="xs">
        <Group gap="sm" align="flex-end" wrap="wrap">
          <Select
            size="xs"
            label="Show"
            w={210}
            data={FILTERS}
            value={filter}
            allowDeselect={false}
            comboboxProps={{ withinPortal: true }}
            onChange={(value) => value && setFilter(value as RowFilter)}
          />
          <TextInput
            size="xs"
            label={analysis.id_column ? `Row or ${analysis.id_column}` : 'Row'}
            placeholder="Find …"
            value={search}
            onChange={(event) => setSearch(event.currentTarget.value)}
            w={160}
          />
          <Text size="xs" c="dimmed" pb={6} data-testid="row-count">
            {formatCount(matching.length)} of {formatCount(analysis.analysed)} rows ·{' '}
            {formatCount(selected.size)} selected
          </Text>
          <Group gap="xs" pb={2}>
            <Button
              size="compact-xs"
              variant="default"
              disabled={matching.length === 0}
              onClick={() => setSelected(new Set([...selected, ...matching]))}
            >
              Select all {formatCount(matching.length)} shown
            </Button>
            <Button
              size="compact-xs"
              variant="default"
              disabled={selected.size === 0}
              onClick={() => setSelected(new Set())}
            >
              Clear selection
            </Button>
            {!readOnly && (
              <Button
                size="compact-xs"
                color="red"
                variant="light"
                disabled={selected.size === 0}
                onClick={() => {
                  onAddStep(
                    dropRowsStep(
                      analysis,
                      [...selected].sort((a, b) => a - b)
                    )
                  );
                  setSelected(new Set());
                }}
              >
                Drop {formatCount(selected.size)} selected rows (step)
              </Button>
            )}
          </Group>
        </Group>
        <RowTable
          analysis={analysis}
          indices={matching}
          sort={sort}
          onSortChange={setSort}
          selected={selected}
          onSelectedChange={setSelected}
          active={active}
          onActivate={(index) => {
            setActive(index);
            setCompareWith(null);
          }}
        />
        <Text size="xs" c="dimmed">
          Click a row to see its most similar rows. Spread and NN ratio as in the histograms; Group:
          the group of near twins (number · rows) at the NN ratio threshold.
        </Text>
      </Stack>

      {active !== null && (
        <SimilarPanel
          analysis={analysis}
          row={active}
          data={similar}
          loading={similarLoading}
          error={similarError}
          onCompare={setCompareWith}
          onOpen={(index) => {
            setActive(index);
            setCompareWith(null);
          }}
        />
      )}
    </Stack>
  );
}

function FeatureNote({ analysis }: { analysis: RowAnalysis }) {
  const { features } = analysis;
  const counts = features.left_out_counts;
  const leftOut = Object.entries(counts)
    .filter(([reason]) => reason !== 'no_partner')
    .map(([reason, n]) => `${formatCount(n)} ${REASONS[reason] ?? reason}`);
  let what: string;
  if (features.used === 'partnered') {
    what = `${formatCount(features.count)} of ${formatCount(features.candidates)} number columns: the ones with a strong partner (|Spearman r| ≥ ${formatNumber(features.partner_threshold, 2)} with another column)`;
  } else if (features.used === 'all') {
    what = `all ${formatCount(features.count)} number columns`;
  } else {
    what = `${formatCount(features.count)} chosen columns`;
  }
  return (
    <Text size="xs" c="dimmed" data-testid="feature-note">
      Compared on {what}
      {leftOut.length ? `; left out: ${leftOut.join(', ')}` : ''}. Columns without any partner are
      mostly noise in wide tables, and noise drowns the differences between rows.
      {features.note ? ` ${features.note}` : ''}
    </Text>
  );
}
