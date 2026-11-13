import { render, screen, within } from '@test-utils';
import type { TuningConfig } from '@/api';
import { trial, TUNING } from './fixtures';
import { TrialsTable } from './RunView';

const CLASSICAL: TuningConfig = {
  method: 'random',
  trials: 3,
  space: { C: { min: 0.1, max: 100 }, gamma: ['scale', 0.01], max_depth: [null, 3] },
  metric: 'balanced_accuracy',
  cv_folds: 5,
};

describe('TrialsTable', () => {
  it('shows the brick parameters and the metric of a classical search', () => {
    const trials = [
      trial({
        params: { C: 3.16228, gamma: 'scale', max_depth: null },
        epochs_done: 1,
        val_loss: 0.4,
        val_accuracy: 0.8,
        score: 0.75,
        folds: 5,
      }),
      trial({
        trial: 2,
        params: { C: 12.5, gamma: 0.01, max_depth: 3 },
        epochs_done: 1,
        val_loss: 0.3,
        val_accuracy: 0.85,
        score: 0.8312,
        folds: 5,
      }),
      trial({ trial: 3, params: { C: 0.2, gamma: 'scale', max_depth: 3 }, status: 'running' }),
    ];
    render(
      <TrialsTable
        tuning={CLASSICAL}
        trials={trials}
        total={3}
        epochs={1}
        best={2}
        active
        classification
        classical
      />
    );
    expect(
      screen.getByText('Ranked by balanced accuracy on 5-fold cross-validation (higher is better).')
    ).toBeInTheDocument();
    expect(
      screen.getByText('Best so far: trial 2 (C 12.5, gamma 0.01, max. depth 3)')
    ).toBeVisible();
    const rows = within(screen.getByRole('table', { name: 'Trials' })).getAllByRole('row');
    expect(rows[0]).toHaveTextContent(
      'TrialCGammaMax. depthCV lossCV acc.CV Balanced accuracyStatus'
    );
    expect(rows[1]).toHaveTextContent('13.16scalenone0.480.0%75.0%Done');
    expect(rows[2]).toHaveTextContent('212.50.0130.385.0%83.1%Best');
    expect(rows[3]).toHaveTextContent('Fitting');
  });

  it('adds the metric of a network search and keeps the loss by default', () => {
    const trials = [trial({ score: 0.912 }), trial({ trial: 2, score: 0.95 })];
    const { unmount } = render(
      <TrialsTable
        tuning={{ ...TUNING, metric: 'roc_auc' }}
        trials={trials}
        total={4}
        epochs={5}
        best={2}
        active={false}
        classification
      />
    );
    expect(
      screen.getByText('Ranked by ROC AUC on the validation data (higher is better).')
    ).toBeInTheDocument();
    let rows = within(screen.getByRole('table', { name: 'Trials' })).getAllByRole('row');
    expect(rows[0]).toHaveTextContent('Val. lossVal. acc.Val. ROC AUCStatus');
    expect(rows[2]).toHaveTextContent('0.95Best');
    expect(screen.getByText(/Its weights are kept/)).toBeInTheDocument();
    unmount();

    // Runs of older versions have no metric: the loss ranked their trials.
    render(
      <TrialsTable
        tuning={TUNING}
        trials={trials}
        total={4}
        epochs={5}
        best={1}
        active={false}
        classification
      />
    );
    expect(
      screen.getByText('Ranked by the loss on the validation data (lower is better).')
    ).toBeInTheDocument();
    rows = within(screen.getByRole('table', { name: 'Trials' })).getAllByRole('row');
    expect(rows[0]).toHaveTextContent('TrialLearning rateBatch sizeVal. lossVal. acc.Status');
  });
});
