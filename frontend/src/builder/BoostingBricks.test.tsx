/** XGBoost, LightGBM and CatBoost bricks: the palette, the inspector (also in German), the
 * summary on the canvas and the search settings. */
import { render, screen, userEvent } from '@test-utils';
import { vi } from 'vitest';
import type { Analysis, LayerCatalog, LayerSpec } from '@/api/types';
import { i18n } from '@/i18n';
import { MODEL_KINDS } from '@/training/classical';
import { CLASSICAL_TUNABLES, DEFAULT_CLASSICAL_PARAMS } from '@/training/search';
import { BOOSTING_LAYERS } from './boostingFixtures';
import { BuilderContext, type BuilderContextValue } from './BuilderContext';
import { CLASSICAL_CATALOG, GP_ANALYSIS, SPECS } from './classicalFixtures';
import { networkSize, summarize } from './format';
import type { LayerFlowNode } from './graph';
import { Inspector } from './Inspector';
import { BOOSTING_LIBRARY_TYPES, isBoosting } from './kinds';
import { Palette } from './Palette';

const CATALOG: LayerCatalog = {
  ...CLASSICAL_CATALOG,
  layers: [...CLASSICAL_CATALOG.layers, ...BOOSTING_LAYERS],
};
const BOOSTING_SPECS: Record<string, LayerSpec> = {
  ...SPECS,
  ...Object.fromEntries(BOOSTING_LAYERS.map((spec) => [spec.type, spec])),
};

function analysisOf(type: string, params: Record<string, unknown>): Analysis {
  return {
    ...GP_ANALYSIS,
    nodes: {
      m: {
        type,
        output_shape: null,
        dtype: null,
        num_params: null,
        params: params as never,
        output_kind: 'estimator',
        kernel: null,
        task: 'classification',
      },
    },
    total_params: null,
    classical: { node: 'm', type, estimator: type, task: 'classification', kernel: null },
  };
}

function renderBrick(type: string, params: Record<string, unknown>, updateNode = vi.fn()) {
  const analysis = analysisOf(type, params);
  const context: BuilderContextValue = {
    specs: BOOSTING_SPECS,
    analysis,
    errorsByNode: {},
    warningsByNode: {},
    readOnly: false,
    updateNode,
    deleteNode: vi.fn(),
    duplicateNode: vi.fn(),
    focusNode: vi.fn(),
  };
  const node: LayerFlowNode = {
    id: 'm',
    type: 'layer',
    position: { x: 0, y: 0 },
    data: { type, params: params as never },
  };
  render(
    <BuilderContext.Provider value={context}>
      <Inspector node={node} nodes={[node]} />
    </BuilderContext.Provider>
  );
}

describe('XGBoost, LightGBM and CatBoost bricks', () => {
  it('are classical models in the palette, found by their library', async () => {
    render(<Palette catalog={CATALOG} onAdd={() => {}} />);
    for (const label of [
      'XGBoost classification',
      'XGBoost regression',
      'LightGBM classification',
      'LightGBM regression',
      'CatBoost classification',
      'CatBoost regression',
    ]) {
      const button = screen.getByRole('button', { name: label });
      expect(button.querySelector('[data-kernel]')).toBeNull();
    }
    await userEvent.type(screen.getByLabelText('Search layers'), 'lightgbm');
    expect(screen.getByRole('button', { name: 'LightGBM classification' })).toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: 'XGBoost classification' })
    ).not.toBeInTheDocument();
  });

  it('show their settings with defaults, and trees after the fit', async () => {
    const updateNode = vi.fn();
    renderBrick('LGBMClassifier', { class_weight: 'none' }, updateNode);
    expect(screen.getByText('classes from data')).toBeInTheDocument();
    expect(screen.getByText('Decision trees').nextSibling).toHaveTextContent('after the fit');
    expect(screen.getByLabelText('Max. trees')).toHaveAttribute('placeholder', '500');
    expect(screen.getByLabelText('Learning rate')).toHaveAttribute('placeholder', '0.05');
    expect(screen.getByLabelText('Early stopping rounds')).toHaveAttribute('placeholder', '50');
    // Empty max. depth: no limit.
    expect(screen.getByLabelText('Max. depth')).toHaveAttribute('placeholder', 'auto');
    await userEvent.click(screen.getByRole('combobox', { name: 'Class weights' }));
    await userEvent.click(await screen.findByRole('option', { name: 'Balanced' }));
    expect(updateNode).toHaveBeenCalledWith('m', { params: { class_weight: 'balanced' } });
  });

  it('are in German, with the option labels translated', async () => {
    await i18n.changeLanguage('de');
    renderBrick('CatBoostClassifier', { iterations: 1000, auto_class_weights: 'none' });
    expect(screen.getByText('Klassen aus den Daten')).toBeInTheDocument();
    expect(screen.getByText('Entscheidungsbäume').nextSibling).toHaveTextContent(
      'nach dem Anpassen'
    );
    expect(screen.getByRole('combobox', { name: 'Class weights' })).toHaveValue('Keine');
    await userEvent.click(screen.getByRole('combobox', { name: 'Class weights' }));
    expect(
      await screen.findByRole('option', { name: 'Wurzel aus ausgeglichen' })
    ).toBeInTheDocument();
    expect(MODEL_KINDS.XGBClassifier).toBe('XGBoost-Klassifikation');
    expect(
      summarize('CatBoostClassifier', { iterations: 1000, learning_rate: 0.05, depth: 6 })
    ).toBe('≤ 1000 Bäume · Rate 0.05 · Tiefe 6');
  });

  it('are summarised on the canvas and count as gradient boosting', () => {
    expect(
      summarize('LGBMRegressor', {
        n_estimators: 500,
        learning_rate: 0.05,
        num_leaves: 31,
        max_depth: null,
      })
    ).toBe('≤ 500 trees · rate 0.05 · 31 leaves');
    expect(
      summarize('XGBClassifier', { n_estimators: 300, learning_rate: 0.1, max_depth: 6 })
    ).toBe('≤ 300 trees · rate 0.1 · depth 6');
    expect(
      summarize('CatBoostClassifier', {
        iterations: 1000,
        learning_rate: 0.05,
        depth: 6,
        auto_class_weights: 'Balanced',
      })
    ).toBe('≤ 1000 trees · rate 0.05 · depth 6 · balanced');
    for (const type of BOOSTING_LIBRARY_TYPES) {
      expect(isBoosting(type)).toBe(true);
      expect(MODEL_KINDS[type]).toMatch(
        /^(XGBoost|LightGBM|CatBoost) (classification|regression)$/
      );
    }
    expect(isBoosting('SVC')).toBe(false);
    expect(networkSize(analysisOf('XGBClassifier', {}))).toBe('Gradient boosting classifier');
  });

  it('can be searched over the parameters of their bricks', () => {
    for (const spec of BOOSTING_LAYERS) {
      const names = spec.params.map((param) => param.name);
      const tunables = CLASSICAL_TUNABLES[spec.type];
      expect(tunables.length).toBeGreaterThanOrEqual(4);
      for (const item of tunables) {
        expect(names).toContain(item.name);
      }
      expect(DEFAULT_CLASSICAL_PARAMS[spec.type]).toHaveLength(3);
    }
  });
});
