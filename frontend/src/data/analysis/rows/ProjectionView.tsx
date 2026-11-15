import { useMemo } from 'react';
import { Select } from '@mantine/core';
import type { RowAnalysis } from '@/api/types';
import { MAX_SCATTER_SERIES } from '@/components/charts/colors';
import { formatNumber, formatPercent } from '@/components/charts/format';
import { ScatterPlot, type ScatterColor } from '@/components/charts/ScatterPlot';
import { i18n, useTranslation } from '@/i18n';
import { rowName, type Thresholds } from './rowsData';

export type ProjectionColor = 'structure' | 'split' | 'target' | 'spread' | 'nn_ratio';

export interface ProjectionViewProps {
  analysis: RowAnalysis;
  thresholds: Thresholds;
  color: ProjectionColor;
  onColorChange: (color: ProjectionColor) => void;
  /** The chosen row (ring). */
  active: number | null;
  onActivate: (index: number) => void;
}

const NAN = Number.NaN;
const asNumbers = (values: (number | null)[]) => values.map((v) => (v === null ? NAN : v));

/**
 * The rows on the first two principal components of the compared columns: groups of near
 * twins and unusually typical rows show up as tight clusters. Coloured by the structure, the
 * split, the target, the spread or the NN ratio.
 */
export function ProjectionView({
  analysis,
  thresholds,
  color,
  onColorChange,
  active,
  onActivate,
}: ProjectionViewProps) {
  useTranslation();
  const { rows } = analysis;
  // Rows without a projection (no values) are left out; `shown` maps points back to rows.
  const shown = useMemo(
    () => rows.pc1.flatMap((x, i) => (x !== null && rows.pc2[i] !== null ? [i] : [])),
    [rows]
  );
  const x = useMemo(() => shown.map((i) => rows.pc1[i]!), [rows, shown]);
  const y = useMemo(() => shown.map((i) => rows.pc2[i]!), [rows, shown]);

  const colorOptions = [
    { value: 'structure', label: i18n.t('ui.data.ProjectionView.typicalNearTwin') },
    ...(rows.split ? [{ value: 'split', label: i18n.t('ui.data.ProjectionView.splitPart') }] : []),
    ...(rows.target
      ? [{ value: 'target', label: analysis.target ?? i18n.t('ui.data.ProjectionView.target') }]
      : []),
    { value: 'spread', label: i18n.t('ui.data.ProjectionView.spread') },
    { value: 'nn_ratio', label: i18n.t('ui.data.ProjectionView.nnRatio') },
  ];
  const used = colorOptions.some((o) => o.value === color) ? color : 'structure';

  const scatterColor = useMemo((): ScatterColor => {
    if (used === 'split' && rows.split) {
      const codes = { train: 0, val: 1, test: 2 } as const;
      return {
        type: 'category',
        label: i18n.t('ui.data.ProjectionView.splitPart'),
        values: shown.map((i) => codes[rows.split![i]]),
        categories: [
          { key: 'train', label: i18n.t('ui.data.ProjectionView.train'), slot: 0 },
          { key: 'val', label: i18n.t('ui.data.ProjectionView.validation'), slot: 1 },
          { key: 'test', label: i18n.t('ui.data.ProjectionView.test'), slot: 2 },
        ],
      };
    }
    if (used === 'target' && rows.target) {
      if (analysis.task === 'regression') {
        return {
          type: 'numeric',
          label: analysis.target ?? i18n.t('ui.data.ProjectionView.target'),
          values: shown.map((i) =>
            typeof rows.target![i] === 'number' ? (rows.target![i] as number) : NAN
          ),
        };
      }
      const counts = new Map<string, number>();
      for (const value of rows.target) {
        if (value !== null) {
          counts.set(String(value), (counts.get(String(value)) ?? 0) + 1);
        }
      }
      const top = [...counts.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, MAX_SCATTER_SERIES)
        .map(([name]) => name);
      const categories = [
        ...top.map((name, slot) => ({ key: name, label: name, slot })),
        { key: '__other', label: i18n.t('ui.data.ProjectionView.other'), slot: -1 },
      ];
      return {
        type: 'category',
        label: analysis.target ?? i18n.t('ui.data.ProjectionView.target'),
        values: shown.map((i) => {
          const at = top.indexOf(String(rows.target![i]));
          return at >= 0 ? at : top.length;
        }),
        categories,
      };
    }
    if (used === 'spread' || used === 'nn_ratio') {
      return {
        type: 'numeric',
        label:
          used === 'spread'
            ? i18n.t('ui.data.ProjectionView.spread')
            : i18n.t('ui.data.ProjectionView.nnRatio'),
        values: shown.map((i) => asNumbers(rows[used])[i]),
        format: (v) => formatNumber(v, 3),
      };
    }
    // Structure: too typical, has a near twin, other.
    return {
      type: 'category',
      label: i18n.t('ui.data.ProjectionView.row'),
      values: shown.map((i) => {
        const spread = rows.spread[i];
        const ratio = rows.nn_ratio[i];
        if (spread !== null && spread < thresholds.atypical) {
          return 0;
        }
        return ratio !== null && ratio < thresholds.twins ? 1 : 2;
      }),
      categories: [
        { key: 'atypical', label: i18n.t('ui.data.ProjectionView.unusuallyTypical'), slot: 1 },
        { key: 'twin', label: i18n.t('ui.data.ProjectionView.hasANearTwin'), slot: 0 },
        { key: 'other', label: i18n.t('ui.data.ProjectionView.otherRows'), slot: -1 },
      ],
    };
  }, [used, rows, shown, analysis.task, analysis.target, thresholds]);

  const explained = analysis.explained;
  const activePoint = active === null ? null : shown.indexOf(active);
  return (
    <ScatterPlot
      title={i18n.t('ui.data.ProjectionView.rowsOnTheTwoMain')}
      subtitle={i18n.t('ui.data.ProjectionView.principalComponentsOfTheCompared', {
        value: formatPercent(explained[0] ?? 0),
        value2: formatPercent(explained[1] ?? 0),
      })}
      controls={
        <Select
          size="xs"
          w={190}
          label={i18n.t('ui.data.ProjectionView.colourBy')}
          data={colorOptions}
          value={used}
          allowDeselect={false}
          comboboxProps={{ withinPortal: true }}
          onChange={(value) => value && onColorChange(value as ProjectionColor)}
        />
      }
      x={x}
      y={y}
      xLabel={i18n.t('ui.data.ProjectionView.component1')}
      yLabel={i18n.t('ui.data.ProjectionView.component2')}
      color={scatterColor}
      pointLabel={(point) => rowName(analysis, shown[point])}
      onPointClick={(point) => onActivate(shown[point])}
      highlight={activePoint !== null && activePoint >= 0 ? activePoint : null}
      height={340}
    />
  );
}
