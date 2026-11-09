/** Regressions of neural networks: R² next to the RMSE, per epoch, per trial and in the list. */
import { cleanup, render, screen, userEvent, within } from '@test-utils';
import { vi } from 'vitest';
import type { HistoryEntry, RunDetail } from '@/api';
import { NO_SCOPE } from './access';
import { CONFIG, detail, DOC, regressionDetail, summary, trial, TUNING } from './fixtures';
import { reportsR2 } from './format';
import { HistoryView } from './HistoryView';
import { RunView } from './RunView';

function show(run: RunDetail) {
  render(
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

/** The tile or card whose first label is `label` (it comes before the chart of that name). */
const card = (label: string) => screen.getAllByText(label)[0].closest('.mantine-Paper-root');

/** The history of `run` without the R² (as runs of older versions report it). */
function withoutR2(run: RunDetail): RunDetail {
  const history = run.state!.history.map(({ val_r2: _, ...entry }) => entry as HistoryEntry);
  return {
    ...run,
    metrics: { ...run.metrics!, val_r2: null, test_r2: null },
    state: { ...run.state!, history, test: { ...run.state!.test!, r2: undefined } },
  };
}

describe('Regression runs of neural networks', () => {
  it('show the R² next to the RMSE, per epoch and as a curve', async () => {
    show(regressionDetail());
    expect(card('Validation RMSE')).toHaveTextContent(/^Validation RMSE41.27R²0.823$/);
    expect(card('Test RMSE')).toHaveTextContent(/^Test RMSE43.9R²0.796loss 0.2 · 200 test/);
    expect(screen.getByRole('slider', { name: /^Validation RMSE/ })).toBeInTheDocument();
    const curve = screen.getByRole('slider', { name: /^Validation R²/ });
    expect(curve).toHaveAttribute('aria-valuenow', '3');

    await userEvent.click(screen.getByRole('button', { name: 'Values per epoch' }));
    const table = await screen.findByRole('table');
    const rows = within(table).getAllByRole('row');
    expect(rows.map((row) => row.textContent)).toEqual([
      'EpochTrain lossVal. lossVal. RMSEVal. R²Time',
      '10.650.6276.4-0.4510 s',
      '20.210.1841.270.8230 s',
      '30.230.243.50.8000 s',
    ]);
  });

  it('show a dash when the validation targets are all equal', async () => {
    const base = regressionDetail();
    const run: RunDetail = {
      ...base,
      metrics: { ...base.metrics!, val_r2: null, test_r2: null },
      state: {
        ...base.state!,
        history: base.state!.history.map((entry) => ({ ...entry, val_r2: null })),
        test: { ...base.state!.test!, r2: null },
      },
    };
    show(run);
    expect(card('Validation RMSE')).toHaveTextContent(/^Validation RMSE41.27R²–$/);
    expect(card('Test RMSE')).toHaveTextContent(/^Test RMSE43.9R²–/);
    expect(screen.queryByRole('slider', { name: /^Validation R²/ })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Values per epoch' }));
    const table = await screen.findByRole('table');
    expect(within(table).getAllByRole('row')[1]).toHaveTextContent('76.4–');
  });

  it('show no R² for classifications and for runs of older versions', () => {
    show(detail());
    expect(screen.queryByText('R²')).not.toBeInTheDocument();
    expect(screen.queryByRole('slider', { name: /R²/ })).not.toBeInTheDocument();
    cleanup();
    show(withoutR2(regressionDetail()));
    expect(card('Validation RMSE')).toHaveTextContent(/^Validation RMSE41.27$/);
    expect(card('Test RMSE')).toHaveTextContent(/^Test RMSE43.9loss/);
    expect(screen.queryByText('R²')).not.toBeInTheDocument();
    expect(screen.queryByRole('slider', { name: /R²/ })).not.toBeInTheDocument();
  });

  it('show the R² of every trial of a search', () => {
    const base = regressionDetail();
    const trials = [
      trial({ val_loss: 0.25, val_accuracy: null, val_rmse: 48.1, val_r2: 0.7612 }),
      trial({
        trial: 2,
        params: { learning_rate: 0.0011, batch_size: 32 },
        val_loss: 0.18,
        val_accuracy: null,
        val_rmse: 41.27,
        val_r2: 0.8234,
      }),
    ];
    show({
      ...base,
      tuning: TUNING,
      trials_done: 2,
      best_trial: { ...trials[1], weights_epoch: 2 },
      config: { ...CONFIG, epochs: 3 },
      state: { ...base.state!, tuning: TUNING, trials_total: 2, trials, best_trial: 2 },
    });
    const rows = within(screen.getByRole('table', { name: 'Trials' })).getAllByRole('row');
    expect(rows[0]).toHaveTextContent('Val. lossVal. RMSEVal. R²Status');
    expect(rows[1]).toHaveTextContent('0.2548.10.761Done');
    expect(rows[2]).toHaveTextContent('0.1841.270.823Best');
  });

  it('are listed with their test R²', () => {
    render(
      <HistoryView
        runs={[summary({ ...regressionDetail(), id: 'mlp' })]}
        error={null}
        onRetry={vi.fn()}
        onOpen={vi.fn()}
        onDeleted={vi.fn()}
        onNewTraining={vi.fn()}
        scope={NO_SCOPE}
        filter="mine"
        onFilterChange={vi.fn()}
      />
    );
    expect(screen.getByText('0.796')).toBeInTheDocument();
    expect(screen.getByText('test R²')).toBeInTheDocument();
  });

  it('know whether a run reports the R²', () => {
    const run = regressionDetail();
    expect(reportsR2(run, run.state!.history)).toBe(true);
    const old = withoutR2(run);
    expect(reportsR2(old, old.state!.history)).toBe(false);
    // A new regression before its first epoch: the R² follows.
    expect(reportsR2({ ...old, status: 'running' }, [])).toBe(true);
    const classification = detail();
    expect(reportsR2(classification, classification.state!.history)).toBe(false);
  });
});
