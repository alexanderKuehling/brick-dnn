/**
 * The run view of an ensemble: its members with links and weights, the comparison of the
 * members with the ensemble per split, its metrics with the threshold analysis, changing the
 * weights, a member that is gone, and predictions with what each member predicts.
 */
import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { afterEach, beforeEach, vi } from 'vitest';
import { api, type PredictionForm, type RunDetail } from '@/api';
import { DOC, ensembleDetail } from './fixtures';
import { RunView } from './RunView';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      training: {
        runs: {
          setEnsembleWeights: vi.fn(),
          setDecisionThreshold: vi.fn(),
          predictForm: vi.fn(),
          predict: vi.fn(),
        },
      },
    },
  };
});

const setWeights = vi.mocked(api.training.runs.setEnsembleWeights);
const predictForm = vi.mocked(api.training.runs.predictForm);
const predict = vi.mocked(api.training.runs.predict);

function show(run: RunDetail, username = 'ada') {
  const handlers = { onUpdate: vi.fn(), onOpenRun: vi.fn(), onDeleted: vi.fn() };
  render(
    <RunView
      run={run}
      doc={DOC}
      scope={{ username, project: null }}
      onBack={vi.fn()}
      onNewTraining={vi.fn()}
      onUpdate={handlers.onUpdate}
      onDeleted={handlers.onDeleted}
      onLoadNetwork={vi.fn()}
      onOpenRun={handlers.onOpenRun}
    />
  );
  return handlers;
}

const members = () => screen.getByRole('table', { name: 'Members of the ensemble' });
const comparison = () =>
  within(screen.getByRole('table', { name: 'Members and ensemble' }))
    .getAllByRole('row')
    .map((row) =>
      within(row)
        .queryAllByRole('cell')
        .map((cell) => cell.textContent)
    )
    .filter((cells) => cells.length > 0);

const FORM: PredictionForm = {
  available: true,
  reason: null,
  kind: 'tabular',
  task: 'classification',
  class_names: ['ok', 'fraud'],
  target: 'label',
  max_rows: 64,
  max_form_rows: 500,
  max_top_k: 10,
  budget_note: null,
  prepared: {
    columns: [
      { name: 'amount', mean: 50, std: 20 },
      { name: 'hour', mean: 12, std: 6 },
    ],
    samples: [],
  },
};

beforeEach(() => {
  predictForm.mockResolvedValue(FORM);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('The run view of an ensemble', () => {
  it('shows its members, method and the comparison per split', async () => {
    const { onOpenRun } = show(ensembleDetail());
    expect(screen.getByRole('heading', { name: 'Fraud ensemble' })).toBeInTheDocument();
    expect(screen.getByText(/Ensemble · weighted average · Fraud/)).toBeInTheDocument();
    expect(
      screen.getByText('The weighted average of the class probabilities the members predict.')
    ).toBeInTheDocument();
    // A member the user can open is a link; the other one is only named.
    await userEvent.click(within(members()).getByRole('button', { name: 'MLP' }));
    expect(onOpenRun).toHaveBeenCalledWith('m1');
    expect(within(members()).queryByRole('button', { name: 'Trees' })).toBeNull();
    expect(within(members()).getByText('Gradient boosting classification')).toBeInTheDocument();

    // The test rows first, then the validation rows.
    expect(comparison()).toEqual([
      ['MLP', '92.0%', '72.0%', '0.740', '0.840', '0.080'],
      ['Trees', '90.0%', '70.0%', '0.720', '0.860', '0.100'],
      ['Ensemble', '94.0%', '74.0%', '0.760', '0.871', '0.060'],
    ]);
    const region = screen.getByRole('region', { name: 'Comparison' });
    await userEvent.click(within(region).getByRole('radio', { name: 'Validation' }));
    expect(comparison()[2]).toEqual(['Ensemble', '92.0%', '72.0%', '0.740', '0.912', '0.080']);
    expect(screen.queryByTestId('ensemble-optimistic')).toBeNull();
    expect(screen.getByText(/100 rows, the same for all/)).toBeInTheDocument();
  });

  it('shows the ensemble’s metrics with the threshold analysis of two classes', () => {
    show(ensembleDetail());
    const metrics = screen.getByRole('region', { name: 'Metrics' });
    expect(within(metrics).getByText('The ensemble')).toBeInTheDocument();
    expect(within(metrics).getByRole('group', { name: 'ROC AUC' })).toHaveTextContent('0.912');
    expect(screen.getByRole('region', { name: 'Decision threshold' })).toBeInTheDocument();
  });

  it('labels stacking metrics on the rows the model was fitted on as optimistic', async () => {
    const base = ensembleDetail();
    const run = ensembleDetail({
      ensemble: { ...base.ensemble!, method: 'stacking', method_label: 'Stacking' },
      state: {
        ...base.state!,
        ensemble: {
          ...base.state!.ensemble!,
          method: 'stacking',
          weights: null,
          fitted_on: 'val',
          optimistic: true,
          splits: ['val'],
          rows: { val: 100 },
        },
      },
    });
    show(run);
    expect(screen.getByTestId('ensemble-optimistic')).toHaveTextContent(
      'The stacking model was fitted on these rows, and the dataset has no test rows'
    );
    // Stacking has no weights to show or change.
    expect(within(members()).queryByText('Weight')).toBeNull();
  });

  it('lets its owner change the weights', async () => {
    const updated = ensembleDetail();
    setWeights.mockResolvedValue(updated);
    const { onUpdate } = show(ensembleDetail());
    expect(screen.queryByRole('button', { name: 'Save weights' })).toBeNull();
    const weight = within(members()).getByRole('textbox', { name: 'Weight of Trees' });
    await userEvent.clear(weight);
    await userEvent.type(weight, '0.75');
    await userEvent.click(screen.getByRole('button', { name: 'Save weights' }));
    await waitFor(() => expect(onUpdate).toHaveBeenCalledWith(updated));
    expect(setWeights).toHaveBeenCalledWith('e1', [0.75, 0.75]);
  });

  it('shows the weights read-only to others and when a member is gone', () => {
    show(ensembleDetail(), 'bob');
    expect(within(members()).queryByRole('textbox')).toBeNull();
    expect(within(members()).getByText('75%')).toBeInTheDocument();

    const base = ensembleDetail();
    const broken = ensembleDetail({
      ensemble: {
        ...base.ensemble!,
        broken: "The member 'Trees' of this ensemble was deleted, so it cannot predict any more.",
        members: [
          base.ensemble!.members[0],
          { ...base.ensemble!.members[1], id: null, deleted: true, visible: false },
        ],
      },
    });
    document.body.innerHTML = '';
    show(broken);
    expect(screen.getByText(/was deleted, so it cannot predict any more/)).toBeInTheDocument();
    expect(screen.getAllByText('deleted').length).toBeGreaterThan(0);
    expect(screen.queryByRole('button', { name: /Try the ensemble/ })).toBeNull();
    expect(within(members()).queryByRole('textbox')).toBeNull();
  });

  it('predicts with what each member predicts', async () => {
    predict.mockResolvedValue({
      run: { id: 'e1', name: 'Fraud ensemble' },
      task: 'classification',
      predictions: [
        {
          class: 'fraud',
          index: 1,
          probability: 0.6,
          positive_probability: 0.6,
          top: [
            { class: 'fraud', index: 1, probability: 0.6 },
            { class: 'ok', index: 0, probability: 0.4 },
          ],
        },
      ],
      members: [
        {
          id: 'm1',
          name: 'MLP',
          predictions: [{ class: 'fraud', index: 1, probability: 0.7, positive_probability: 0.7 }],
        },
        {
          id: 'm2',
          name: 'Trees',
          predictions: [{ class: 'ok', index: 0, probability: 0.7, positive_probability: 0.3 }],
        },
      ],
    });
    show(ensembleDetail());
    await userEvent.click(screen.getByRole('button', { name: 'Try the ensemble: predict' }));
    const toggle = await screen.findByRole('switch', { name: 'Show what each member predicts' });
    expect(toggle).toBeChecked();
    await userEvent.type(screen.getByRole('textbox', { name: 'amount' }), '120');
    await userEvent.type(screen.getByRole('textbox', { name: 'hour' }), '3');
    await userEvent.click(screen.getByRole('button', { name: 'Predict' }));
    const table = await screen.findByRole('table', { name: 'Predictions of the members' });
    expect(predict).toHaveBeenCalledWith(
      'e1',
      expect.objectContaining({ members: true, inputs: [{ amount: 120, hour: 3 }] }),
      undefined
    );
    const rows = within(table).getAllByRole('row');
    expect(rows[1]).toHaveTextContent('MLPfraud (70.0 %)70.0 %');
    expect(rows[2]).toHaveTextContent('Treesok (70.0 %)30.0 %');
    expect(screen.getByText('Predicted label')).toBeInTheDocument();
  });
});
