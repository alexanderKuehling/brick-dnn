/** The ways into the explanations in the run view: the headline button and the section. */
import { render, screen, userEvent } from '@test-utils';
import { vi } from 'vitest';
import type { RunDetail } from '@/api';
import type { ExplainMethods } from '@/explain/types';
import { detail, DOC } from './fixtures';
import { RunView } from './RunView';

const { methods, run } = vi.hoisted(() => ({ methods: vi.fn(), run: vi.fn() }));

vi.mock('@/explain/api', async (importOriginal) => {
  const original = await importOriginal<typeof import('@/explain/api')>();
  return { ...original, explainApi: { methods, run } };
});

function show(runDetail: RunDetail) {
  return render(
    <RunView
      run={runDetail}
      doc={DOC}
      onBack={vi.fn()}
      onNewTraining={vi.fn()}
      onUpdate={vi.fn()}
      onDeleted={vi.fn()}
      onLoadNetwork={vi.fn()}
    />
  );
}

function methodsFor(kind: ExplainMethods['kind'], ids: string[]): ExplainMethods {
  return {
    family: 'neural',
    kind,
    unavailable: null,
    demo: kind === 'demo',
    methods: ['samples', 'image', 'tabular', 'permutation', 'activations', 'weights'].map((id) => ({
      id: id as ExplainMethods['methods'][number]['id'],
      label: id,
      available: ids.includes(id),
      reason: ids.includes(id) ? null : 'Not for this run.',
    })),
  };
}

const section = () => screen.queryByRole('region', { name: 'Why the model decides' });

beforeEach(() => {
  localStorage.clear();
  methods.mockReset();
  run.mockReset();
});

describe('Explanations in the run view', () => {
  it('offers them in the headline and as a section of a finished run', async () => {
    methods.mockResolvedValue(methodsFor('image', ['samples', 'image', 'activations', 'weights']));
    show(detail({ status: 'completed', has_weights: true }));
    expect(screen.getByRole('button', { name: 'Explain the model' })).toBeInTheDocument();
    expect(screen.getByText(/Why it decides as it does/)).toBeInTheDocument();
    expect(section()).toBeInTheDocument();
    expect(await screen.findByRole('button', { name: 'Show where it looks' })).toBeInTheDocument();
    // Nothing is computed until asked for.
    expect(run).not.toHaveBeenCalled();
    // The section comes after the results, before "Try the model".
    const tryIt = screen.getByRole('button', { name: 'Try the model: predict' });
    expect(section()!.compareDocumentPosition(tryIt)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  });

  it('opens the dialog from the headline', async () => {
    methods.mockResolvedValue(methodsFor('image', ['activations', 'weights']));
    run.mockResolvedValue({ method: 'weights', filters: null, layers: [] });
    show(detail({ status: 'completed', has_weights: true }));
    await userEvent.click(screen.getByRole('button', { name: 'Explain the model' }));
    expect(await screen.findByRole('dialog', { name: 'Explain “Small CNN”' })).toBeInTheDocument();
  });

  it('shows neither for runs without weights or still running', () => {
    show(detail({ status: 'failed', has_weights: false }));
    expect(screen.queryByRole('button', { name: 'Explain the model' })).not.toBeInTheDocument();
    expect(section()).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Small CNN' })).toBeInTheDocument();
    expect(methods).not.toHaveBeenCalled();
  });

  it('shows neither while the run trains', () => {
    show(detail({ status: 'running', has_weights: true }));
    expect(screen.getByText('Epoch 4 of 10')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Explain the model' })).not.toBeInTheDocument();
    expect(section()).not.toBeInTheDocument();
  });

  it('explains what a run on demo data can show', async () => {
    methods.mockResolvedValue(methodsFor('demo', ['activations', 'weights']));
    show(
      detail({
        status: 'completed',
        has_weights: true,
        dataset: { kind: 'random', id: null, name: 'Demo data' },
      })
    );
    expect(await screen.findByText(/Trained on random demo data/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Look inside the network' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Explain the model' })).toBeInTheDocument();
  });
});
