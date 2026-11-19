import { useState } from 'react';
import { render, screen, userEvent } from '@test-utils';
import { vi } from 'vitest';
import type { RunDetail } from '@/api/types';
import { DOC, regressionDetail, detail as runDetail } from './fixtures';
import { lossText } from './format';
import { RunView } from './RunView';
import { availableMetrics } from './search';
import {
  DEFAULT_SETTINGS,
  LOSS_SEARCH_METRIC,
  lossScopeFor,
  parseClassWeights,
  plannedTrials,
  toConfig,
  toTuning,
  validateSettings,
  type LossScope,
  type SettingsValues,
} from './settings';
import { SettingsForm } from './SettingsForm';

let latest: SettingsValues = DEFAULT_SETTINGS;

const FRAUD: LossScope = { task: 'classification', classes: ['fraud', 'ok'] };
const PRICES: LossScope = { task: 'regression', classes: null };

function Harness({
  loss,
  initial = DEFAULT_SETTINGS,
  segmentation = false,
}: {
  loss: LossScope | null;
  initial?: SettingsValues;
  segmentation?: boolean;
}) {
  const [values, setValues] = useState<SettingsValues>(initial);
  latest = values;
  const task = loss?.task ?? null;
  const metrics = task === 'regression' ? availableMetrics('regression', null) : null;
  return (
    <>
      <SettingsForm
        values={values}
        onChange={setValues}
        errors={validateSettings(values, {
          loss,
          segmentation: segmentation ? { maxBatch: null } : null,
        })}
        name=""
        namePlaceholder="Net"
        onNameChange={() => {}}
        devices={[]}
        defaultDevice={null}
        builtin={false}
        segmentation={segmentation ? { maxBatch: null } : null}
        loss={loss}
        metrics={metrics}
      />
      <output>{JSON.stringify(toConfig(values, { loss, segmentation }))}</output>
    </>
  );
}

const config = () => JSON.parse(screen.getByRole('status').textContent ?? '{}');

async function choose(name: string, option: string) {
  await userEvent.click(screen.getByRole('combobox', { name }));
  await userEvent.click(await screen.findByRole('option', { name: option }));
}

describe('the loss of the settings', () => {
  it('offers the losses of the task of the data', async () => {
    const { unmount } = render(<Harness loss={FRAUD} />);
    const select = screen.getByRole('combobox', { name: 'Loss' });
    expect(select).toHaveValue('Cross-entropy');
    await userEvent.click(select);
    const options = screen.getAllByRole('option').map((option) => option.textContent);
    expect(options).toEqual(['Cross-entropy', 'Weighted cross-entropy', 'Focal loss']);
    await userEvent.keyboard('{Escape}');
    unmount();

    render(<Harness loss={PRICES} />);
    expect(screen.getByRole('combobox', { name: 'Loss' })).toHaveValue('Mean squared error');
    await userEvent.click(screen.getByRole('combobox', { name: 'Loss' }));
    expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual([
      'Mean squared error',
      'Mean absolute error',
      'Huber loss',
    ]);
  });

  it('is not offered while the task is not known, and keeps the default', () => {
    render(<Harness loss={null} />);
    expect(screen.queryByRole('combobox', { name: 'Loss' })).not.toBeInTheDocument();
    expect(config()).not.toHaveProperty('loss');
  });

  it('sends nothing new for the default loss', () => {
    render(<Harness loss={FRAUD} />);
    expect(config()).not.toHaveProperty('loss');
    expect(config()).not.toHaveProperty('label_smoothing');
    expect(config()).not.toHaveProperty('class_weights');
    // A loss of another task (the data changed) is the default of this one.
    expect(toConfig({ ...DEFAULT_SETTINGS, loss: 'mae' }, { loss: FRAUD })).not.toHaveProperty(
      'loss'
    );
  });

  it('shows the settings of the weighted cross-entropy and sends its class weights', async () => {
    render(<Harness loss={FRAUD} />);
    expect(screen.getByLabelText('Label smoothing')).toBeInTheDocument();
    expect(screen.queryByRole('combobox', { name: 'Class weights' })).not.toBeInTheDocument();
    await choose('Loss', 'Weighted cross-entropy');
    expect(config()).toMatchObject({ loss: 'weighted_cross_entropy', class_weights: 'balanced' });
    expect(screen.getByRole('combobox', { name: 'Class weights' })).toHaveValue(
      'Balanced (from the class counts)'
    );
    expect(screen.getByText(/an alternative to oversampling/)).toBeInTheDocument();
    await choose('Class weights', 'Custom');
    const weights = screen.getByLabelText('Weights');
    await userEvent.type(weights, '1, 2, 3');
    expect(await screen.findByText('One weight per class: 2 (fraud, ok)')).toBeInTheDocument();
    expect(config()).not.toHaveProperty('class_weights');
    await userEvent.clear(weights);
    await userEvent.type(weights, 'fraud: 25');
    expect(config()).toMatchObject({ class_weights: { fraud: 25 } });
    expect(validateSettings(latest, { loss: FRAUD })).toEqual({});
    const smoothing = screen.getByLabelText('Label smoothing');
    await userEvent.clear(smoothing);
    await userEvent.type(smoothing, '0.1');
    expect(config()).toMatchObject({ label_smoothing: 0.1 });
  });

  it('shows gamma and optional weights for the focal loss', async () => {
    render(<Harness loss={FRAUD} />);
    await choose('Loss', 'Focal loss');
    expect(screen.queryByLabelText('Label smoothing')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Focal gamma')).toHaveValue('2');
    expect(config()).toMatchObject({ loss: 'focal', focal_gamma: 2, class_weights: 'balanced' });
    await choose('Class weights', 'None');
    expect(config()).toMatchObject({ class_weights: null });
    const gamma = screen.getByLabelText('Focal gamma');
    await userEvent.clear(gamma);
    await userEvent.type(gamma, '12');
    expect(validateSettings(latest, { loss: FRAUD }).focal_gamma).toBe('Between 0 and 10');
  });

  it("shows Huber's delta in standard deviations", async () => {
    render(<Harness loss={PRICES} />);
    expect(screen.queryByLabelText('Huber delta')).not.toBeInTheDocument();
    await choose('Loss', 'Huber loss');
    const delta = screen.getByLabelText('Huber delta');
    // What delta means is shown on demand, next to its label.
    await userEvent.click(screen.getByRole('button', { name: 'More about “Huber delta”' }));
    expect(
      await screen.findByText(
        'Where the loss turns from squared to linear, in standard deviations of the target'
      )
    ).toBeVisible();
    await userEvent.keyboard('{Escape}');
    expect(config()).toMatchObject({ loss: 'huber', huber_delta: 1 });
    await userEvent.clear(delta);
    await userEvent.type(delta, '0');
    expect(validateSettings(latest, { loss: PRICES }).huber_delta).toBe(
      'Greater than 0, at most 10'
    );
  });

  it('offers the weighted cross-entropy of pixels for a segmentation', async () => {
    render(<Harness loss={null} segmentation />);
    await choose('Loss', 'Weighted cross-entropy');
    expect(config()).toMatchObject({ loss: 'weighted_cross_entropy', class_weights: 'balanced' });
    expect(screen.getByText(/pixels of the training images/)).toBeInTheDocument();
    expect(screen.queryByLabelText('Label smoothing')).not.toBeInTheDocument();
  });

  it('parses given class weights', () => {
    expect(parseClassWeights('1, 10')).toEqual([1, 10]);
    expect(parseClassWeights('fraud: 10; ok=1', ['fraud', 'ok'])).toEqual({ fraud: 10, ok: 1 });
    expect(parseClassWeights('1: 4', ['fraud', 'ok'])).toEqual({ 1: 4 });
    expect(parseClassWeights('spam: 2', ['fraud', 'ok'])).toBe("'spam' is not a class: fraud, ok");
    expect(parseClassWeights('1, -2')).toBe('Weights greater than 0 and at most 1000');
    expect(parseClassWeights('fraud:')).toBe('Weights greater than 0 and at most 1000');
    expect(parseClassWeights(' ')).toBe('Enter a weight per class, e.g. "1, 10", or "fraud: 10"');
  });

  it('follows the plan of the training check', () => {
    expect(
      lossScopeFor({
        task: 'classification',
        targets: [{ kind: 'class', num_classes: 2 }],
        class_names: ['fraud', 'ok'],
      } as never)
    ).toEqual(FRAUD);
    expect(
      lossScopeFor({ task: 'regression', targets: [{ kind: 'value', shape: [1] }] } as never)
    ).toEqual(PRICES);
    expect(
      lossScopeFor({
        task: 'demo',
        targets: [
          { kind: 'class', num_classes: 2 },
          { kind: 'value', shape: [1] },
        ],
      } as never)
    ).toBeNull();
    const classical = { task: 'classification', targets: [{ kind: 'class', num_classes: 2 }] };
    expect(lossScopeFor({ ...classical, classical: { type: 'SVC' } } as never)).toBeNull();
    expect(
      lossScopeFor({ ...classical, classical: { type: 'SVC' }, hybrid: { features: 4 } } as never)
    ).toEqual({ task: 'classification', classes: ['0', '1'] });
  });
});

describe('a search over the loss', () => {
  const SEARCH: SettingsValues = { ...DEFAULT_SETTINGS, search: true };

  it('has rows for the loss and the settings of the losses it tries', async () => {
    render(<Harness loss={FRAUD} initial={SEARCH} />);
    const loss = screen.getByRole('checkbox', { name: 'Search loss' });
    expect(screen.queryByRole('checkbox', { name: 'Search focal gamma' })).not.toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Search label smoothing' })).toBeInTheDocument();
    await userEvent.click(loss);
    expect(screen.getByRole('combobox', { name: 'Loss' })).toBeDisabled();
    expect(screen.getByRole('checkbox', { name: 'Search focal gamma' })).toBeInTheDocument();
    expect(validateSettings(latest, { loss: FRAUD }).metric).toBe(LOSS_SEARCH_METRIC());
    const tuning = toTuning({ ...latest, metric: 'pr_auc' }, null, { loss: FRAUD });
    expect(tuning?.space).toEqual({
      learning_rate: { min: 0.0001, max: 0.01 },
      batch_size: [32, 64, 128],
      loss: ['cross_entropy', 'focal'],
    });
    expect(config()).not.toHaveProperty('loss');
    await userEvent.click(screen.getByRole('checkbox', { name: 'Search focal gamma' }));
    const values = { ...latest, metric: 'pr_auc' as const, search_method: 'grid' as const };
    expect(toTuning(values, null, { loss: FRAUD })?.space).toMatchObject({
      loss: ['cross_entropy', 'focal'],
      focal_gamma: [1, 2, 3],
    });
    // 3 learning rates × 3 batch sizes × 2 losses × 3 gammas.
    expect(plannedTrials(values, 'classification')).toBe(54);
    expect(validateSettings({ ...values, tune_learning_rate: false }, { loss: FRAUD })).toEqual({});
  });

  it('of a regression varies the Huber delta', async () => {
    const initial = { ...SEARCH, loss: 'huber' as const, metric: 'rmse' as const };
    render(<Harness loss={PRICES} initial={initial} />);
    const row = screen.getByRole('checkbox', { name: 'Search Huber delta' });
    await userEvent.click(row);
    expect(screen.getByRole('textbox', { name: 'Huber delta' })).toBeDisabled();
    expect(config()).not.toHaveProperty('huber_delta');
    expect(screen.getByRole('combobox', { name: 'Huber deltas to try' })).toBeInTheDocument();
    expect(toTuning(latest, null, { loss: PRICES })?.space).toMatchObject({
      huber_delta: [0.5, 1, 2],
    });
    expect(validateSettings({ ...latest, huber_deltas: [] }, { loss: PRICES })).toEqual({
      huber_deltas: 'Choose at least one',
    });
  });

  it('of a hybrid network may rank by the loss of its classical model', () => {
    const values = { ...SEARCH, search_neural: true, tune_loss: true };
    const errors = validateSettings(values, { hybrid: { model: 'SVC' }, loss: FRAUD });
    expect(errors.metric).toBeUndefined();
    const tuning = toTuning(values, 'SVC', { hybrid: true, loss: FRAUD });
    expect(tuning?.neural_space?.loss).toEqual(['cross_entropy', 'focal']);
  });
});

describe('the loss of a run', () => {
  it('is named with its settings', () => {
    const run = runDetail();
    expect(lossText(run)).toBe('Cross-entropy');
    const focal: RunDetail = {
      ...run,
      state: {
        ...run.state!,
        loss: {
          name: 'focal',
          label: 'focal loss',
          task: 'classification',
          focal_gamma: 2,
          class_weights_from: 'balanced',
          class_weights: [12, 0.52],
        },
      },
    };
    expect(lossText(focal)).toBe('Focal loss (gamma 2, balanced class weights 12, 0.52)');
    // Runs of older versions said cross_entropy for every task.
    const old = regressionDetail();
    expect(lossText({ ...old, config: { ...old.config, loss: 'cross_entropy' } })).toBe(
      'Mean squared error'
    );
    const huber = { ...old, config: { ...old.config, loss: 'huber' as const, huber_delta: 0.5 } };
    expect(lossText(huber)).toBe('Huber loss (delta 0.5 standard deviations)');
  });
});

describe('the run view', () => {
  it('says which loss the curves show', async () => {
    const run = regressionDetail();
    const huber: RunDetail = {
      ...run,
      config: { ...run.config, loss: 'huber', huber_delta: 0.5 },
      state: {
        ...run.state!,
        loss: { name: 'huber', label: 'Huber loss', task: 'regression', huber_delta: 0.5 },
      },
    };
    render(
      <RunView
        run={huber}
        doc={DOC}
        onBack={vi.fn()}
        onNewTraining={vi.fn()}
        onUpdate={vi.fn()}
        onDeleted={vi.fn()}
        onLoadNetwork={vi.fn()}
      />
    );
    // Next to the heading of the curves, on demand.
    await userEvent.click(screen.getByRole('button', { name: 'More about “Training curves”' }));
    expect(
      await screen.findByText(
        'Loss: Huber loss (delta 0.5 standard deviations). The curves, early stopping and the kept epoch use this loss; losses of different kinds are not comparable, so compare runs by their metrics.'
      )
    ).toBeInTheDocument();
  });
});
