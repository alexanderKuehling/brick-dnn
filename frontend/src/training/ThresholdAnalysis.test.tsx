/**
 * Binary classification in the run view: ROC AUC and PR AUC tiles, the decision-threshold
 * slider with precision, recall, F1, false-positive rate and counts per split, the two
 * suggestions, saving the threshold (only who started the run), the curves, and the threshold
 * shown where predictions are made.
 */
import { fireEvent, render, screen, userEvent, waitFor, within } from '@test-utils';
import { afterEach, vi } from 'vitest';
import { api, type RunDetail, type RunEvaluation } from '@/api';
import { CurveChart } from '@/components/charts/CurveChart';
import { binaryEvaluation, classificationEvaluation, detail, DOC, svcDetail } from './fixtures';
import { RunView } from './RunView';
import { formatThreshold, nearestIndex, pointAt, prLine, rocLine } from './threshold';

function finished(
  evaluation: RunEvaluation | null,
  base: RunDetail = detail(),
  threshold: number | null = 0.5
): RunDetail {
  return {
    ...base,
    status: 'completed',
    decision_threshold: threshold,
    state: { ...base.state!, status: 'completed', progress: null, evaluation },
  };
}

function show(run: RunDetail, options: { username?: string; onUpdate?: () => void } = {}) {
  return render(
    <RunView
      run={run}
      doc={DOC}
      scope={{ username: options.username ?? null, project: null }}
      onBack={vi.fn()}
      onNewTraining={vi.fn()}
      onUpdate={options.onUpdate ?? vi.fn()}
      onDeleted={vi.fn()}
      onLoadNetwork={vi.fn()}
    />
  );
}

const metrics = () => screen.getByRole('region', { name: 'Metrics' });
const tile = (label: string) => within(metrics()).getByRole('group', { name: label });
const section = () => screen.getByRole('region', { name: 'Decision threshold' });
const slider = () => within(section()).getByRole('slider', { name: 'Decision threshold' });
const results = () =>
  within(screen.getByRole('table', { name: 'Results at the threshold' }))
    .getAllByRole('row')
    .slice(1)
    .map((row) =>
      within(row)
        .getAllByRole('cell')
        .map((cell) => cell.textContent)
    );

afterEach(() => {
  vi.restoreAllMocks();
});

describe('The decision threshold of a binary classification', () => {
  it('shows ROC AUC and PR AUC per split', () => {
    show(finished(binaryEvaluation()));
    expect(tile('ROC AUC')).toHaveTextContent('ROC AUC0.912Validation0.871Test');
    expect(tile('PR AUC')).toHaveTextContent('PR AUC0.634Validation0.552Test');
    expect(tile('PR AUC')).toHaveTextContent('average precision of “fraud”');
  });

  it('starts at the saved threshold with the results of every split', () => {
    show(finished(binaryEvaluation()));
    expect(within(section()).getByTestId('threshold-value')).toHaveTextContent('0.5');
    expect(slider()).toHaveAttribute('aria-valuenow', '4');
    expect(results()).toEqual([
      ['Precision', '66.7%', '66.7%'],
      ['Recall', '40.0%', '40.0%'],
      ['F1', '0.500', '0.500'],
      ['False-positive rate', '2.22%', '2.22%'],
      ['True positives (fraud found)', '4', '2'],
      ['False positives (false alarms)', '2', '1'],
      ['True negatives', '88', '44'],
      ['False negatives (fraud missed)', '6', '3'],
    ]);
    expect(
      within(section()).getByText(/The thresholds and suggestions come from the validation rows/)
    ).toBeInTheDocument();
    expect(within(section()).getByTestId('saved-threshold')).toHaveTextContent(
      'Predictions use 0.5.'
    );
  });

  it('jumps to the suggestions and moves with the keyboard', async () => {
    show(finished(binaryEvaluation()));
    await userEvent.click(
      within(section()).getByRole('button', { name: 'Best F1: threshold 0.3' })
    );
    expect(within(section()).getByTestId('threshold-value')).toHaveTextContent('0.3');
    expect(results().slice(0, 4)).toEqual([
      ['Precision', '53.8%', '40.0%'],
      ['Recall', '70.0%', '80.0%'],
      ['F1', '0.609', '0.533'],
      ['False-positive rate', '6.67%', '13.33%'],
    ]);
    await userEvent.click(
      within(section()).getByRole('button', { name: 'Recall ≥ 80%: threshold 0.2' })
    );
    expect(results()[1]).toEqual(['Recall', '90.0%', '100.0%']);
    fireEvent.keyDown(slider(), { key: 'ArrowLeft' });
    expect(within(section()).getByTestId('threshold-value')).toHaveTextContent('0.1');
    expect(results()[4]).toEqual(['True positives (fraud found)', '10', '5']);
    await userEvent.click(
      within(section()).getByRole('button', { name: 'Default: threshold 0.5' })
    );
    expect(within(section()).getByTestId('threshold-value')).toHaveTextContent('0.5');
  });

  it('saves the threshold for predictions', async () => {
    const save = vi
      .spyOn(api.training.runs, 'setDecisionThreshold')
      .mockResolvedValue({ id: 'run-1', decision_threshold: 0.3 });
    const onUpdate = vi.fn();
    const run = finished(binaryEvaluation());
    show(run, { onUpdate });
    const button = within(section()).getByRole('button', {
      name: 'Use this threshold for predictions',
    });
    expect(button).toBeDisabled(); // already the saved one
    await userEvent.click(
      within(section()).getByRole('button', { name: 'Best F1: threshold 0.3' })
    );
    expect(button).toBeEnabled();
    await userEvent.click(button);
    expect(save).toHaveBeenCalledWith(run.id, 0.3);
    await waitFor(() =>
      expect(onUpdate).toHaveBeenCalledWith(expect.objectContaining({ decision_threshold: 0.3 }))
    );
  });

  it('shows another saved threshold and its mark', () => {
    show(finished(binaryEvaluation(), detail(), 0.2));
    expect(within(section()).getByTestId('threshold-value')).toHaveTextContent('0.2');
    expect(within(section()).getByTestId('saved-threshold')).toHaveTextContent(
      'Predictions use 0.2.'
    );
    expect(within(section()).getByText('0.5')).toBeInTheDocument(); // the default's mark
  });

  it('cannot be saved by others than who started the run', () => {
    show(finished(binaryEvaluation(), detail({ started_by: 'ada' })), { username: 'bob' });
    expect(within(section()).queryByRole('button', { name: /Use this threshold/ })).toBeNull();
    expect(within(section()).getByTestId('saved-threshold')).toHaveTextContent(
      'Only who started the run can change it.'
    );
    // The slider still works for looking.
    expect(slider()).toBeInTheDocument();
  });

  it('says when the thresholds come from the test rows', () => {
    const base = binaryEvaluation();
    show(finished({ ...base, val: null, binary: { ...base.binary!, source: 'test' } }));
    expect(results()[0]).toEqual(['Precision', '66.7%']);
    expect(
      within(section()).getByText(/No validation rows: the thresholds and suggestions come from/)
    ).toBeInTheDocument();
  });

  it('works for classical classifiers too', () => {
    show(finished(binaryEvaluation(), svcDetail()));
    expect(section()).toBeInTheDocument();
    expect(tile('ROC AUC')).toBeInTheDocument();
  });

  it('is not shown for more classes or runs from before the analysis', () => {
    show(finished(classificationEvaluation()));
    expect(screen.queryByRole('region', { name: 'Decision threshold' })).toBeNull();
    expect(within(metrics()).queryByRole('group', { name: 'ROC AUC' })).toBeNull();
    const { binary: _binary, ...old } = binaryEvaluation();
    show(finished({ ...old, val: { ...old.val!, threshold_curve: undefined }, test: null }));
    expect(screen.queryByRole('region', { name: 'Decision threshold' })).toBeNull();
  });
});

describe('The curves', () => {
  it('reads precision-recall and ROC lines from the curve', () => {
    const curve = binaryEvaluation().val!.threshold_curve!;
    const pr = prLine(curve);
    // From the highest threshold to the lowest; nothing predicted at 1: no precision.
    expect(pr.threshold).toEqual([0.7, 0.5, 0.3, 0.2, 0.1, 0]);
    expect(pr.x[0]).toBeCloseTo(0.3);
    expect(pr.y[0]).toBe(1);
    const roc = rocLine(curve);
    expect([roc.x[0], roc.y[0]]).toEqual([0, 0]);
    expect([roc.x.at(-1), roc.y.at(-1)]).toEqual([1, 1]);
    expect(roc.x).toHaveLength(8);
    expect(pointAt(curve, 3)).toMatchObject({ threshold: 0.3, tp: 7, fp: 6, tn: 84, fn: 3 });
    expect(nearestIndex(curve.points.threshold, 0.42)).toBe(4);
    expect(nearestIndex(curve.points.threshold, 0.04)).toBe(0);
    expect(formatThreshold(0.123456)).toBe('0.1235');
    expect(formatThreshold(0.0001234)).toBe('0.000123');
    expect(formatThreshold(0)).toBe('0');
  });

  it('draws a line and a marker per split, and the diagonal', () => {
    render(
      <CurveChart
        title="ROC curve"
        xLabel="False-positive rate"
        yLabel="Recall"
        diagonal="Guessing"
        width={320}
        series={[
          { key: 'val', label: 'Validation', x: [0, 0.5, 1], y: [0, 0.8, 1], marker: null },
          {
            key: 'test',
            label: 'Test',
            x: [0, 0.4, 1],
            y: [0, 0.7, 1],
            marker: { x: 0.4, y: 0.7 },
          },
        ]}
      />
    );
    expect(
      screen.getByRole('img', { name: /ROC curve: Recall against False-positive rate/ })
    ).toBeInTheDocument();
    expect(screen.getByTestId('curve-val').getAttribute('points')?.split(' ')).toHaveLength(3);
    expect(screen.getByTestId('curve-test')).toBeInTheDocument();
    expect(screen.queryByTestId('curve-marker-val')).toBeNull();
    expect(screen.getByTestId('curve-marker-test')).toBeInTheDocument();
    expect(screen.getByTestId('curve-diagonal')).toBeInTheDocument();
    expect(screen.getByText('Guessing')).toBeInTheDocument();
  });
});
