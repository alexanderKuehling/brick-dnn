/**
 * Combining finished runs into an ensemble from the list of runs: selecting the runs, the
 * dialog with the method and the weights, and what is sent to the server.
 */
import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { afterEach, beforeEach, vi } from 'vitest';
import { api, ApiError, type RunSummary } from '@/api';
import { ensembleDetail, fraudRun } from './fixtures';
import { HistoryView } from './HistoryView';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: { training: { ensembles: { create: vi.fn() }, runs: { remove: vi.fn() } } },
  };
});

const create = vi.mocked(api.training.ensembles.create);

const RUNS: RunSummary[] = [
  fraudRun({ id: 'a', name: 'MLP', family: 'neural' }),
  fraudRun({ id: 'b', name: 'Trees', family: 'classical' }),
  fraudRun({ id: 'c', name: 'Other data', dataset: { kind: 'upload', id: 9, name: 'Loans' } }),
  fraudRun({ id: 'd', name: 'Still training', status: 'running', has_weights: false }),
  { ...ensembleDetail({ id: 'e', name: 'Old ensemble' }) },
];

function show(onCombined = vi.fn()) {
  render(
    <HistoryView
      runs={RUNS}
      error={null}
      onRetry={vi.fn()}
      onOpen={vi.fn()}
      onDeleted={vi.fn()}
      onNewTraining={vi.fn()}
      scope={{ username: 'ada', project: null }}
      filter="mine"
      onFilterChange={vi.fn()}
      onCombined={onCombined}
      combineProject={null}
    />
  );
  return onCombined;
}

beforeEach(() => {
  create.mockResolvedValue(ensembleDetail({ id: 'new', name: 'Ensemble of MLP and Trees' }));
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('Combining runs into an ensemble', () => {
  it('offers only finished runs of a table, and marks ensembles', async () => {
    show();
    expect(screen.getByText('Ensemble')).toBeInTheDocument(); // the badge of the old one
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Combine runs' }));
    expect(screen.getByRole('checkbox', { name: 'Select MLP' })).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Select Trees' })).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Select Other data' })).toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: 'Select Still training' })).toBeNull();
    expect(screen.queryByRole('checkbox', { name: 'Select Old ensemble' })).toBeNull();
    const combine = screen.getByRole('button', { name: 'Combine into an ensemble' });
    expect(combine).toBeDisabled();
    expect(screen.getByText('Select at least 2 finished runs.')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('checkbox', { name: 'Select MLP' }));
    await userEvent.click(screen.getByRole('checkbox', { name: 'Select Other data' }));
    expect(screen.getByText(/trained on different datasets/)).toBeInTheDocument();
    expect(combine).toBeDisabled();
    await userEvent.click(screen.getByRole('checkbox', { name: 'Select Other data' }));
    await userEvent.click(screen.getByRole('checkbox', { name: 'Select Trees' }));
    expect(screen.getByText('2 runs selected.')).toBeInTheDocument();
    expect(combine).toBeEnabled();

    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
  });

  it('creates a weighted average with the chosen weights', async () => {
    const onCombined = show();
    await userEvent.click(screen.getByRole('button', { name: 'Combine runs' }));
    await userEvent.click(screen.getByRole('checkbox', { name: 'Select MLP' }));
    await userEvent.click(screen.getByRole('checkbox', { name: 'Select Trees' }));
    await userEvent.click(screen.getByRole('button', { name: 'Combine into an ensemble' }));

    const dialog = await screen.findByRole('dialog', { name: 'Combine into an ensemble' });
    expect(within(dialog).getByLabelText('Name')).toHaveValue('Ensemble of MLP and Trees');
    expect(within(dialog).getByText(/weighted average of the class probabilities/)).toBeVisible();
    const weight = within(dialog).getByRole('textbox', { name: 'Weight of MLP' });
    await userEvent.clear(weight);
    await userEvent.type(weight, '3');
    const rows = within(within(dialog).getByRole('table', { name: 'Members' })).getAllByRole('row');
    expect(rows[1]).toHaveTextContent('75%');
    expect(rows[2]).toHaveTextContent('25%');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Create ensemble' }));

    await waitFor(() => expect(onCombined).toHaveBeenCalled());
    expect(create).toHaveBeenCalledWith({
      members: ['a', 'b'],
      method: 'average',
      weights: [3, 1],
      name: 'Ensemble of MLP and Trees',
      project: null,
    });
    expect(onCombined.mock.calls[0][0].id).toBe('new');
  });

  it('creates a stacking ensemble without weights and shows why the server refuses', async () => {
    create.mockRejectedValueOnce(
      new ApiError(400, {}, "The runs predict different classes: 'MLP' has 3 classes.")
    );
    const onCombined = show();
    await userEvent.click(screen.getByRole('button', { name: 'Combine runs' }));
    await userEvent.click(screen.getByRole('checkbox', { name: 'Select MLP' }));
    await userEvent.click(screen.getByRole('checkbox', { name: 'Select Trees' }));
    await userEvent.click(screen.getByRole('button', { name: 'Combine into an ensemble' }));
    const dialog = await screen.findByRole('dialog', { name: 'Combine into an ensemble' });
    await userEvent.click(within(dialog).getByText('Stacking'));
    expect(within(dialog).queryByRole('textbox', { name: 'Weight of MLP' })).toBeNull();
    expect(
      within(dialog).getByText(/logistic regression learns from the validation rows/)
    ).toBeVisible();

    await userEvent.click(within(dialog).getByRole('button', { name: 'Create ensemble' }));
    expect(await within(dialog).findByText(/predict different classes/)).toBeInTheDocument();
    expect(onCombined).not.toHaveBeenCalled();

    await userEvent.click(within(dialog).getByRole('button', { name: 'Create ensemble' }));
    await waitFor(() => expect(onCombined).toHaveBeenCalled());
    expect(create).toHaveBeenLastCalledWith({
      members: ['a', 'b'],
      method: 'stacking',
      name: 'Ensemble of MLP and Trees',
      project: null,
    });
  });
});
