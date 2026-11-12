/**
 * The final evaluation of a run in the run view: metric tiles (validation and test side by
 * side), the confusion matrix (counts or % of the true class), the sortable table per class,
 * the regression tiles with predicted vs. actual, and nothing for runs without it.
 */
import { render, screen, userEvent, within } from '@test-utils';
import { vi } from 'vitest';
import type { RunDetail, RunEvaluation } from '@/api';
import { sortClasses } from './EvaluationResults';
import {
  classificationEvaluation,
  detail,
  DOC,
  regressionDetail,
  regressionEvaluation,
  svcDetail,
} from './fixtures';
import { RunView } from './RunView';

function show(run: RunDetail) {
  return render(
    <RunView
      run={run}
      doc={DOC}
      onBack={vi.fn()}
      onNewTraining={vi.fn()}
      onUpdate={vi.fn()}
      onDeleted={vi.fn()}
      onLoadNetwork={vi.fn()}
    />
  );
}

function finished(evaluation: RunEvaluation | null, base: RunDetail = detail()): RunDetail {
  return {
    ...base,
    status: 'completed',
    state: { ...base.state!, status: 'completed', progress: null, evaluation },
  };
}

const metrics = () => screen.getByRole('region', { name: 'Metrics' });
const tile = (label: string) => within(metrics()).getByRole('group', { name: label });
const matrix = () => screen.getByRole('table', { name: /^Confusion matrix/ });
const rowsOf = (table: HTMLElement) =>
  within(table)
    .getAllByRole('row')
    .map((row) =>
      within(row)
        .queryAllByRole('cell')
        .map((cell) => cell.textContent)
    );

describe('The metrics of a classification run', () => {
  it('shows validation and test side by side in tiles', () => {
    show(finished(classificationEvaluation()));
    expect(within(metrics()).getByText('Weights of epoch 3')).toBeInTheDocument();
    expect(tile('Accuracy')).toHaveTextContent('Accuracy80.0%Validation75.0%Test');
    expect(tile('Balanced accuracy')).toHaveTextContent('80.0%Validation51.7%Test');
    expect(tile('Macro F1')).toHaveTextContent('0.797Validation0.489Test');
    expect(tile('Log loss (cross-entropy)')).toHaveTextContent('0.2Train0.31Validation0.35Test');
    // Three classes: top-3 accuracy is always 100 % and not shown.
    expect(within(metrics()).queryByRole('group', { name: 'Top-3 accuracy' })).toBeNull();
  });

  it('draws the confusion matrix with class labels, counts and % of the true class', async () => {
    show(finished(classificationEvaluation()));
    const table = matrix();
    const headers = within(table)
      .getAllByRole('columnheader')
      .map((h) => h.textContent);
    expect(headers).toEqual(['true ↓ predicted →', 'cat', 'dog', 'bird']);
    expect(
      within(table)
        .getAllByRole('rowheader')
        .map((h) => h.textContent)
    ).toEqual(['cat', 'dog', 'bird']);
    expect(rowsOf(table).slice(1)).toEqual([
      ['9', '1', '0'],
      ['2', '7', '1'],
      ['1', '1', '8'],
    ]);
    // Hover: both numbers.
    expect(within(table).getByTitle('True dog, predicted cat: 2 (20.0% of dog)')).toHaveTextContent(
      '2'
    );

    await userEvent.click(screen.getByRole('radio', { name: '% of true class' }));
    expect(rowsOf(matrix()).slice(1)).toEqual([
      ['90%', '10%', '0%'],
      ['20%', '70%', '10%'],
      ['10%', '10%', '80%'],
    ]);
    expect(screen.getByText(/Rows: true class · columns: predicted class/)).toBeInTheDocument();
  });

  it('lists precision, recall, F1 and support per class and sorts them', async () => {
    show(finished(classificationEvaluation()));
    const table = screen.getByRole('table', { name: 'Metrics per class' });
    expect(rowsOf(table).slice(1)).toEqual([
      ['cat', '75.0%', '90.0%', '0.818', '10'],
      ['dog', '77.8%', '70.0%', '0.737', '10'],
      ['bird', '90.0%', '80.0%', '0.847', '10'],
    ]);
    await userEvent.click(within(table).getByRole('button', { name: 'F1' }));
    expect(
      rowsOf(table)
        .slice(1)
        .map((row) => row[0])
    ).toEqual(['bird', 'cat', 'dog']);
    expect(within(table).getByRole('columnheader', { name: 'F1' })).toHaveAttribute(
      'aria-sort',
      'descending'
    );
    await userEvent.click(within(table).getByRole('button', { name: 'F1' }));
    expect(
      rowsOf(table)
        .slice(1)
        .map((row) => row[0])
    ).toEqual(['dog', 'cat', 'bird']);
  });

  it('switches the matrix and the table to the test rows', async () => {
    show(finished(classificationEvaluation()));
    await userEvent.click(screen.getByRole('radio', { name: 'Test' }));
    expect(rowsOf(matrix()).slice(1)).toEqual([
      ['8', '2', '0'],
      ['2', '6', '0'],
      ['0', '2', '0'],
    ]);
    const table = screen.getByRole('table', { name: 'Metrics per class' });
    // bird was never predicted: no precision.
    expect(rowsOf(table)[3]).toEqual(['bird', '–', '0.0%', '0.000', '2']);
    expect(screen.getByText(/20 test rows · 3 classes/)).toBeInTheDocument();
  });

  it('shows the most confused pairs instead of a matrix for many classes', () => {
    const base = classificationEvaluation();
    const names = Array.from({ length: 60 }, (_, i) => `class ${i}`);
    const perClass = {
      precision: names.map(() => 0.5),
      recall: names.map(() => 0.5),
      f1: names.map(() => 0.5),
      support: names.map(() => 10),
    };
    const evaluation = classificationEvaluation({
      class_names: names,
      test: null,
      val: {
        ...base.val!,
        size: 600,
        top_k_accuracy: { '3': 0.81, '5': 0.9 },
        per_class: perClass,
        confusion_matrix: null,
        most_confused: [{ true: 12, predicted: 40, count: 4 }],
      },
    });
    show(finished(evaluation));
    expect(screen.queryByRole('table', { name: /^Confusion matrix/ })).toBeNull();
    expect(screen.getByText(/No confusion matrix for 60 classes/)).toBeInTheDocument();
    const pairs = screen.getByRole('table', { name: 'Most confused classes' });
    expect(rowsOf(pairs)[1]).toEqual(['class 12', 'class 40', '4', '40.0%']);
    expect(tile('Top-5 accuracy')).toHaveTextContent('90.0%Validation');
    expect(screen.queryByRole('radio', { name: 'Test' })).toBeNull(); // no test rows
    expect(
      within(screen.getByRole('table', { name: 'Metrics per class' })).getAllByRole('row')
    ).toHaveLength(61);
  });

  it('names the fitted model of a classical run', () => {
    show(finished(classificationEvaluation({ test: null }), svcDetail()));
    expect(within(metrics()).getByText('The fitted model')).toBeInTheDocument();
    expect(tile('Accuracy')).toHaveTextContent(/^Accuracy80.0%Validation$/);
    expect(tile('Log loss (cross-entropy)')).toHaveTextContent(
      'train: the fitted model on its training rows'
    );
  });
});

describe('The metrics of a regression run', () => {
  it('shows the error tiles and predicted vs. actual', async () => {
    show(finished(regressionEvaluation(), regressionDetail()));
    expect(tile('RMSE')).toHaveTextContent('41.27Validation43.9Test');
    expect(tile('MAE')).toHaveTextContent('30.5Validation33.1Test');
    expect(tile('R²')).toHaveTextContent('0.823Validation0.796Test');
    expect(tile('MAPE')).toHaveTextContent('4.1%Validation–Test'); // a test target is 0
    expect(tile('Max error')).toHaveTextContent('98.4Validation101.2Test');
    expect(tile('Loss (MSE, standardised)')).toHaveTextContent('0.18Train0.2Validation0.22Test');
    expect(screen.getByText(/half between -20.1 and 22.4/)).toBeInTheDocument();
    expect(screen.getByRole('slider', { name: /^Predicted vs. actual/ })).toBeInTheDocument();
    expect(screen.getByText('All 4 rows.')).toBeInTheDocument();
    expect(screen.getByText('Predicted = actual')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('radio', { name: 'Test' }));
    expect(screen.getByText('All 3 rows.')).toBeInTheDocument();
    expect(screen.queryByRole('table', { name: /^Confusion matrix/ })).toBeNull();
  });
});

/** A finished run trained without validation data (a validation split of 0). */
function withoutValidation(evaluation: RunEvaluation, testSize: number | null): RunDetail {
  const base = finished(evaluation);
  const history = base.state!.history.map((entry) => ({
    ...entry,
    val_loss: null,
    val_accuracy: null,
  }));
  return {
    ...base,
    metrics: {
      train_loss: 0.2,
      train_accuracy: 0.94,
      val_loss: null,
      val_accuracy: null,
      val_rmse: null,
    },
    best: null,
    state: {
      ...base.state!,
      val_size: 0,
      test_size: testSize,
      history,
      best: null,
      weights_epoch: 3,
    },
  };
}

describe('Runs without validation data', () => {
  it('show the test rows only', () => {
    const base = classificationEvaluation();
    show(withoutValidation({ ...base, val: null }, 20));
    expect(tile('Accuracy')).toHaveTextContent('Accuracy75.0%Test');
    expect(tile('Log loss (cross-entropy)')).toHaveTextContent('0.2Train0.35Test');
    expect(screen.getByText(/20 test rows · 3 classes/)).toBeInTheDocument();
    // One split: nothing to pick.
    expect(screen.queryByRole('radio', { name: 'Validation' })).toBeNull();
    expect(screen.queryByTestId('optimistic-metrics')).toBeNull();
    // The tiles of the epochs: training values and the kept (last) epoch.
    expect(screen.getByText('Training accuracy')).toBeInTheDocument();
    expect(screen.getByText('no validation data')).toBeInTheDocument();
    expect(screen.getByText('Kept epoch')).toBeInTheDocument();
    expect(screen.queryByText('Best epoch')).toBeNull();
    expect(screen.queryByText('Validation loss')).toBeNull();
  });

  it('mark numbers from the training rows as optimistic', async () => {
    const base = classificationEvaluation();
    show(withoutValidation({ ...base, val: null, test: null, training: base.val }, null));
    expect(tile('Accuracy')).toHaveTextContent('Accuracy80.0%Training (optimistic)');
    expect(screen.getByTestId('optimistic-metrics')).toHaveTextContent(
      /No validation or test rows: these numbers come from the training rows/
    );
    expect(screen.getByText(/training rows · 3 classes/)).toBeInTheDocument();
    // The values per epoch have only training columns.
    await userEvent.click(screen.getByRole('button', { name: /Values per epoch/ }));
    expect(await screen.findByRole('columnheader', { name: 'Train acc.' })).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Val. loss' })).toBeNull();
  });

  it('show a regression measured on its training rows', () => {
    const base = regressionEvaluation();
    show(
      finished(
        { ...base, val: null, test: null, training: base.val },
        regressionDetail({ status: 'completed' })
      )
    );
    expect(tile('RMSE')).toHaveTextContent('41.27Training (optimistic)');
    expect(screen.getByText(/Residuals \(predicted − actual\), training rows/)).toBeInTheDocument();
  });
});

describe('Runs without an evaluation', () => {
  it('show no metrics section (older runs, running runs)', () => {
    show(finished(null));
    expect(screen.queryByRole('region', { name: 'Metrics' })).toBeNull();
    const old = finished(null);
    delete old.state!.evaluation;
    show(old);
    expect(screen.queryByRole('region', { name: 'Metrics' })).toBeNull();
    // While it runs, the evaluation is not shown yet.
    const running = detail({ status: 'running' });
    show({ ...running, state: { ...running.state!, evaluation: classificationEvaluation() } });
    expect(screen.queryByRole('region', { name: 'Metrics' })).toBeNull();
  });
});

describe('sortClasses', () => {
  it('sorts by a column with undefined values last', () => {
    const perClass = {
      precision: [0.5, null, 0.9, 0.5],
      recall: [1, 1, 1, 1],
      f1: [0, 0, 0, 0],
      support: [3, 1, 2, 4],
    };
    expect(sortClasses(perClass, 4, 'precision', true)).toEqual([2, 0, 3, 1]);
    expect(sortClasses(perClass, 4, 'precision', false)).toEqual([0, 3, 2, 1]);
    expect(sortClasses(perClass, 4, 'support', false)).toEqual([1, 2, 0, 3]);
    expect(sortClasses(perClass, 4, 'class', true)).toEqual([3, 2, 1, 0]);
  });
});
