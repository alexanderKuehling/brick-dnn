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
import { i18n, useTranslation } from '@/i18n';
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
  {
    value: 'auto',
    get label() {
      return i18n.t('ui.data.RowsView.automatic');
    },
  },
  {
    value: 'partnered',
    get label() {
      return i18n.t('ui.data.RowsView.partnered');
    },
  },
  {
    value: 'all',
    get label() {
      return i18n.t('ui.data.RowsView.allNumberColumns');
    },
  },
  {
    value: 'columns',
    get label() {
      return i18n.t('ui.data.RowsView.iChoose');
    },
  },
];

const FILTERS: { value: RowFilter; label: string }[] = [
  {
    value: 'all',
    get label() {
      return i18n.t('ui.data.RowsView.allRows');
    },
  },
  {
    value: 'atypical',
    get label() {
      return i18n.t('ui.data.RowsView.unusuallyTypical');
    },
  },
  {
    value: 'twins',
    get label() {
      return i18n.t('ui.data.RowsView.withANearTwin');
    },
  },
  {
    value: 'group',
    get label() {
      return i18n.t('ui.data.RowsView.groupOfTheChosenRow');
    },
  },
  {
    value: 'selected',
    get label() {
      return i18n.t('ui.data.RowsView.selected');
    },
  },
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
  return error instanceof Error ? error.message : i18n.t('ui.data.RowsView.theRowsCouldNotBe');
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
  useTranslation();
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
            {i18n.t('ui.data.RowsView.compareRowsOn')}
          </Text>
          <SegmentedControl
            size="xs"
            aria-label={i18n.t('ui.data.RowsView.compareRowsOn')}
            data={FEATURE_MODES}
            value={mode}
            onChange={(value) => setMode(value as RowFeatureMode)}
          />
        </div>
        {mode === 'columns' && (
          <MultiSelect
            size="xs"
            label={i18n.t('ui.data.RowsView.columns')}
            placeholder={i18n.t('ui.data.RowsView.chooseNumberColumns')}
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
          {i18n.t('ui.data.RowsView.chooseTheNumberColumnsTo')}
        </Text>
      </Stack>
    );
  }
  if (!analysis) {
    return (
      <Stack gap="md">
        {featureControl}
        {error ? (
          <Alert
            color="red"
            icon={<IconAlertTriangle size={18} />}
            title={i18n.t('ui.data.RowsView.cannotAnalyseTheRows')}
          >
            <Text size="sm">{error}</Text>
            <Button
              size="compact-sm"
              variant="light"
              color="red"
              mt="xs"
              onClick={() => setAttempt((n) => n + 1)}
            >
              {i18n.t('ui.data.RowsView.tryAgain')}
            </Button>
          </Alert>
        ) : (
          <Group gap="xs">
            <Loader size="sm" />
            <Text size="sm" c="dimmed">
              {i18n.t('ui.data.RowsView.comparingEveryRowWithEvery')}
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
        {i18n.t('ui.data.RowsView.eachRowComparedWithAll')}
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
            {i18n.t('ui.data.RowsView.aRandomSampleOfOf', {
              analysed: formatCount(analysis.analysed),
              n_total: formatCount(analysis.n_total),
              count: formatCount(analysis.features.count),
            })}
          </Text>
        </Alert>
      )}

      <div className={classes.tiles}>
        <StatTile
          label={i18n.t('ui.data.RowsView.rowsAnalysed')}
          value={formatCount(analysis.analysed)}
          hint={`${formatNumber((analysis.stats_ms ?? 0) / 1000, 2)} s`}
        />
        <StatTile
          label={i18n.t('ui.data.RowsView.unusuallyTypical')}
          value={formatCount(atypicalCount)}
          hint={i18n.t('ui.data.RowsView.spreadBelow', {
            atypical: formatNumber(thresholds.atypical, 3),
          })}
        />
        <StatTile
          label={i18n.t('ui.data.RowsView.withANearTwin')}
          value={formatCount(twinCount)}
          hint={i18n.t('ui.data.RowsView.nnRatioBelow', {
            twins: formatNumber(thresholds.twins, 3),
          })}
        />
        <StatTile
          label={i18n.t('ui.data.RowsView.groupsOfNearTwins')}
          value={formatCount(groups.count)}
          hint={
            groups.count
              ? i18n.t('ui.data.RowsView.rowsUpToEach', {
                  rows: formatCount(groups.rows),
                  largest: groups.largest,
                })
              : 'none'
          }
        />
        {groups.held_out > 0 && (
          <StatTile
            label={i18n.t('ui.data.RowsView.validationTestRowsWithA')}
            value={formatPercent(groups.held_out_with_twin / groups.held_out)}
            hint={
              analysis.split_group_by
                ? i18n.t('ui.data.RowsView.splitKeepsTogether', {
                    split_group_by: analysis.split_group_by,
                  })
                : i18n.t('ui.data.RowsView.of', {
                    held_out_with_twin: formatCount(groups.held_out_with_twin),
                    held_out: formatCount(groups.held_out),
                  })
            }
          />
        )}
      </div>

      <div className={classes.grid}>
        <ThresholdHistogram
          title={i18n.t('ui.data.RowsView.howTypicalIsEachRow')}
          description={i18n.t('ui.data.RowsView.medianDistanceOfTheRow')}
          values={analysis.rows.spread}
          threshold={thresholds.atypical}
          onThresholdChange={setAtypicalOverride}
          suggested={analysis.suggested.atypical}
          cap={4}
          xLabel="spread"
          belowText={(n) =>
            i18n.t('ui.data.RowsView.below', {
              n: formatCount(n),
              value:
                n === 1 ? i18n.t('ui.data.RowsView.rowIs') : i18n.t('ui.data.RowsView.rowsAre'),
            })
          }
          actions={
            <Group gap="xs">
              <Button size="compact-xs" variant="default" onClick={() => setFilter('atypical')}>
                {i18n.t('ui.data.RowsView.showThem')}
              </Button>
              {!readOnly && (
                <Button
                  size="compact-xs"
                  variant="light"
                  disabled={atypicalCount === 0}
                  onClick={removeAtypical}
                >
                  {i18n.t('ui.data.RowsView.removeRowsStep', {
                    atypicalCount: formatCount(atypicalCount),
                  })}
                </Button>
              )}
            </Group>
          }
        />
        <ThresholdHistogram
          title={i18n.t('ui.data.RowsView.howCloseIsEachRow')}
          description={i18n.t('ui.data.RowsView.distanceToTheNearestOther')}
          values={analysis.rows.nn_ratio}
          baseline={randomRatios}
          baselineLabel="random pairs (same number of rows)"
          threshold={thresholds.twins}
          onThresholdChange={setTwinOverride}
          suggested={analysis.suggested.twins}
          cap={2}
          xLabel={i18n.t('ui.data.RowsView.nnRatio')}
          belowText={(n) =>
            i18n.t('ui.data.RowsView.aNearTwin', {
              n: formatCount(n),
              value:
                n === 1 ? i18n.t('ui.data.RowsView.rowHas') : i18n.t('ui.data.RowsView.rowsHave'),
            })
          }
          actions={
            <Group gap="xs">
              <Button size="compact-xs" variant="default" onClick={() => setFilter('twins')}>
                {i18n.t('ui.data.RowsView.showThem')}
              </Button>
              {!readOnly && (
                <Button
                  size="compact-xs"
                  variant="light"
                  disabled={twinCount === 0}
                  onClick={findGroups}
                >
                  {i18n.t('ui.data.RowsView.findGroupsSplitByThem')}
                </Button>
              )}
            </Group>
          }
        />
      </div>
      {analysis.suggested.atypical === null && analysis.suggested.twins === null && (
        <Text size="xs" c="dimmed">
          {i18n.t('ui.data.RowsView.neitherHistogramHasAClear')}
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
            label={i18n.t('ui.data.RowsView.show')}
            w={210}
            data={FILTERS}
            value={filter}
            allowDeselect={false}
            comboboxProps={{ withinPortal: true }}
            onChange={(value) => value && setFilter(value as RowFilter)}
          />
          <TextInput
            size="xs"
            label={
              analysis.id_column
                ? i18n.t('ui.data.RowsView.rowOr', { id_column: analysis.id_column })
                : i18n.t('ui.data.RowsView.row')
            }
            placeholder={i18n.t('ui.data.RowsView.find')}
            value={search}
            onChange={(event) => setSearch(event.currentTarget.value)}
            w={160}
          />
          <Text size="xs" c="dimmed" pb={6} data-testid="row-count">
            {i18n.t('ui.data.RowsView.ofRowsSelected', {
              length: formatCount(matching.length),
              analysed: formatCount(analysis.analysed),
              size: formatCount(selected.size),
            })}
          </Text>
          <Group gap="xs" pb={2}>
            <Button
              size="compact-xs"
              variant="default"
              disabled={matching.length === 0}
              onClick={() => setSelected(new Set([...selected, ...matching]))}
            >
              {i18n.t('ui.data.RowsView.selectAllShown', { length: formatCount(matching.length) })}
            </Button>
            <Button
              size="compact-xs"
              variant="default"
              disabled={selected.size === 0}
              onClick={() => setSelected(new Set())}
            >
              {i18n.t('ui.data.RowsView.clearSelection')}
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
                {i18n.t('ui.data.RowsView.dropSelectedRowsStep', {
                  size: formatCount(selected.size),
                })}
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
          {i18n.t('ui.data.RowsView.clickARowToSee')}
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
  useTranslation();
  const { features } = analysis;
  const counts = features.left_out_counts;
  const leftOut = Object.entries(counts)
    .filter(([reason]) => reason !== 'no_partner')
    .map(([reason, n]) => `${formatCount(n)} ${REASONS[reason] ?? reason}`);
  let what: string;
  if (features.used === 'partnered') {
    what = i18n.t('ui.data.RowsView.ofNumberColumnsTheOnes', {
      count: formatCount(features.count),
      candidates: formatCount(features.candidates),
      partner_threshold: formatNumber(features.partner_threshold, 2),
    });
  } else if (features.used === 'all') {
    what = i18n.t('ui.data.RowsView.allNumberColumns2', { count: formatCount(features.count) });
  } else {
    what = i18n.t('ui.data.RowsView.chosenColumns', { count: formatCount(features.count) });
  }
  return (
    <Text size="xs" c="dimmed" data-testid="feature-note">
      {i18n.t('ui.data.RowsView.comparedOnColumnsWithoutAny', {
        what,
        value: leftOut.length
          ? i18n.t('ui.data.RowsView.leftOut', { leftOut: leftOut.join(', ') })
          : '',
        value2: features.note ? ` ${features.note}` : '',
      })}
    </Text>
  );
}
