import { useEffect, useRef, useState } from 'react';
import { Box, SegmentedControl, Select, Stack, Text } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import type { DataProfile, DataSample, NewStep } from '@/api/types';
import { ColumnsView } from './ColumnsView';
import { CompareView } from './CompareView';
import { CorrelationsView, type CorrelationMethod } from './CorrelationsView';
import { DistributionView, type DistributionOptions } from './DistributionView';
import { MissingView } from './MissingView';
import { OverviewView } from './OverviewView';
import { RowsView, type RowsSource } from './rows/RowsView';
import { sampleColumn } from './sampleData';
import { defaultScatterAxes, ScatterView, type ScatterOptions } from './ScatterView';
import { TargetView } from './TargetView';
import classes from './AnalysisPanel.module.css';

export type AnalysisView =
  | 'overview'
  | 'columns'
  | 'distribution'
  | 'missing'
  | 'correlations'
  | 'scatter'
  | 'target'
  | 'compare'
  // Row structure (atypical rows, near twins, groups)
  | 'rows';

export const ANALYSIS_VIEWS: { value: AnalysisView; label: string }[] = [
  { value: 'overview', label: 'Overview' },
  { value: 'columns', label: 'Columns' },
  { value: 'distribution', label: 'Distribution' },
  { value: 'missing', label: 'Missing values' },
  { value: 'correlations', label: 'Correlations' },
  { value: 'scatter', label: 'Scatter' },
  { value: 'target', label: 'Target' },
  { value: 'compare', label: 'Compare' },
  // Row structure (atypical rows, near twins, groups)
  { value: 'rows', label: 'Rows' },
];

/** Below this width the view switcher is a select instead of a segmented control. */
const SEGMENTED_MIN_WIDTH = 720;

export interface AnalysisPanelProps {
  profile: DataProfile;
  /** Profile before the cleaning steps; null until compare was requested. */
  before: DataProfile | null;
  sample: DataSample;
  target: string | null;
  /** Ask the page to load `before` (compare: true on the next preview). */
  onRequestCompare: () => void;
  onAddStep: (step: NewStep) => void;
  /** Focused column (controlled, optional). Setting it from outside opens its distribution. */
  column?: string | null;
  onColumnChange?: (column: string | null) => void;
  /** Shown view (controlled, optional). */
  view?: AnalysisView;
  onViewChange?: (view: AnalysisView) => void;
  /** Loads the row analysis (Rows view); without it the view explains it is unavailable. */
  rows?: RowsSource;
  /** Hide actions that change the preparation. */
  readOnly?: boolean;
}

/**
 * Visual analysis of the previewed data: overview, columns, distributions, missing values,
 * correlations, scatter plots, the target and a before/after comparison. Has its natural
 * height (the page scrolls it); works from phone width up.
 */
export function AnalysisPanel({
  profile,
  before,
  sample,
  target,
  onRequestCompare,
  onAddStep,
  column: columnProp,
  onColumnChange,
  view: viewProp,
  onViewChange,
  rows,
  readOnly = false,
}: AnalysisPanelProps) {
  const { ref, width } = useElementSize();
  const [ownView, setOwnView] = useState<AnalysisView>('overview');
  const view = viewProp ?? ownView;
  const setView = (next: AnalysisView) => {
    setOwnView(next);
    onViewChange?.(next);
  };

  // The focused column: controlled by the page if it passes `column`.
  const [ownColumn, setOwnColumn] = useState<string | null>(null);
  const controlled = columnProp !== undefined;
  const column = controlled ? columnProp : ownColumn;
  const emitted = useRef<string | null | undefined>(undefined);
  const changeColumn = (next: string | null) => {
    emitted.current = next;
    if (!controlled) {
      setOwnColumn(next);
    }
    onColumnChange?.(next);
  };
  // A column focused from outside (e.g. "Show in Analysis" in the table) opens its distribution.
  const viewChange = useRef(setView);
  useEffect(() => {
    viewChange.current = setView;
  });
  useEffect(() => {
    if (columnProp && columnProp !== emitted.current) {
      viewChange.current('distribution');
    }
  }, [columnProp]);

  const showColumn = (name: string) => {
    changeColumn(name);
    setView('distribution');
  };

  const focused =
    column && profile.columns.some((c) => c.name === column)
      ? column
      : (profile.columns.find((c) => c.name !== target)?.name ?? profile.columns[0]?.name ?? null);

  const [distribution, setDistribution] = useState<DistributionOptions>({
    log: false,
    split: true,
    zoom: null,
  });
  const [method, setMethod] = useState<CorrelationMethod>('pearson');
  const [compareLog, setCompareLog] = useState(false);
  const [scatter, setScatter] = useState<ScatterOptions>(() => {
    const [x, y] = defaultScatterAxes(profile, sample, target);
    const color =
      sampleColumn(sample, target)?.kind !== 'text' && sampleColumn(sample, target) ? target : null;
    return { x, y, color, trend: true, jitter: false };
  });
  // Columns can disappear when the steps change; fall back to sensible defaults.
  const scatterOptions = (() => {
    const exists = (name: string | null) => !!name && !!sampleColumn(sample, name);
    if (exists(scatter.x) && exists(scatter.y)) {
      return { ...scatter, color: exists(scatter.color) ? scatter.color : null };
    }
    const [x, y] = defaultScatterAxes(profile, sample, target);
    return { ...scatter, x, y, color: exists(scatter.color) ? scatter.color : null };
  })();
  const openScatter = (x: string, y: string) => {
    setScatter({ ...scatterOptions, x, y });
    setView('scatter');
  };

  const narrow = width > 0 && width < SEGMENTED_MIN_WIDTH;

  return (
    <Box ref={ref} className={classes.root}>
      <Stack gap="md">
        {narrow ? (
          <Select
            size="xs"
            label="View"
            data={ANALYSIS_VIEWS}
            value={view}
            onChange={(value) => value && setView(value as AnalysisView)}
            allowDeselect={false}
            comboboxProps={{ withinPortal: true }}
            w={220}
          />
        ) : (
          <SegmentedControl
            size="xs"
            aria-label="Analysis view"
            data={ANALYSIS_VIEWS}
            value={view}
            onChange={(value) => setView(value as AnalysisView)}
            style={{ alignSelf: 'flex-start' }}
          />
        )}

        {view === 'overview' && (
          <OverviewView
            profile={profile}
            sample={sample}
            target={target}
            onAddStep={onAddStep}
            onShowColumn={showColumn}
          />
        )}
        {view === 'columns' && (
          <ColumnsView profile={profile} target={target} onShowColumn={showColumn} />
        )}
        {view === 'distribution' && (
          <DistributionView
            profile={profile}
            sample={sample}
            target={target}
            column={focused}
            onColumnChange={changeColumn}
            options={distribution}
            onOptionsChange={setDistribution}
            onAddStep={onAddStep}
            onScatter={openScatter}
          />
        )}
        {view === 'missing' && (
          <MissingView
            profile={profile}
            sample={sample}
            target={target}
            onAddStep={onAddStep}
            onShowColumn={showColumn}
          />
        )}
        {view === 'correlations' && (
          <CorrelationsView
            profile={profile}
            method={method}
            onMethodChange={setMethod}
            onScatter={openScatter}
          />
        )}
        {view === 'scatter' && (
          <ScatterView
            profile={profile}
            sample={sample}
            target={target}
            options={scatterOptions}
            onOptionsChange={setScatter}
          />
        )}
        {view === 'target' && (
          <TargetView profile={profile} sample={sample} target={target} onShowColumn={showColumn} />
        )}
        {view === 'rows' &&
          (rows ? (
            <RowsView
              source={rows}
              numberColumns={profile.columns
                .filter((c) => (c.kind === 'numeric' || c.kind === 'boolean') && c.name !== target)
                .map((c) => c.name)}
              columns={profile.columns.map((c) => c.name)}
              onAddStep={onAddStep}
              readOnly={readOnly}
            />
          ) : (
            <Text size="sm" c="dimmed">
              The row analysis is not available here.
            </Text>
          ))}
        {view === 'compare' && (
          <CompareView
            profile={profile}
            before={before}
            onRequestCompare={onRequestCompare}
            column={focused}
            onColumnChange={changeColumn}
            log={compareLog}
            onLogChange={setCompareLog}
          />
        )}
      </Stack>
    </Box>
  );
}
