/** "Why the model decides" in the run view, with the server's answers faked. */
import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { vi } from 'vitest';
import type { RunDetail } from '@/api';
import { detail } from '@/training/fixtures';
import { EXPLAINED_RUNS_KEY, pickSamples, RunExplanations } from './RunExplanations';
import type { ExplainMethods, ExplainRequest, ListedSample } from './types';

const { methods, run } = vi.hoisted(() => ({ methods: vi.fn(), run: vi.fn() }));

vi.mock('./api', async (importOriginal) => {
  const original = await importOriginal<typeof import('./api')>();
  return { ...original, explainApi: { methods, run } };
});
vi.mock('@/api', async (importOriginal) => {
  const original = await importOriginal<typeof import('@/api')>();
  return {
    ...original,
    api: { ...original.api, layers: () => Promise.resolve({ categories: [], layers: [] }) },
  };
});

const PNG = 'data:image/png;base64,iVBORw0KGgo=';
const GRADCAM = 'data:image/png;base64,Z3JhZGNhbQ==';
const SALIENCY = 'data:image/png;base64,c2FsaWVuY3k=';

function available(kind: ExplainMethods['kind'], ids: string[]): ExplainMethods {
  const all = ['samples', 'image', 'tabular', 'permutation', 'boundary', 'activations', 'weights'];
  return {
    family: 'neural',
    kind,
    unavailable: null,
    demo: kind === 'demo',
    methods: all.map((id) => ({
      id: id as ExplainMethods['methods'][number]['id'],
      label: id,
      available: ids.includes(id),
      reason: ids.includes(id) ? null : `Not for this run (${id}).`,
    })),
  };
}

const RUN: RunDetail = {
  ...detail({ id: 'run-7', status: 'completed', has_weights: true }),
  plan: {
    task: 'classification',
    summary: '',
    inputs: [{ shape: [12], dtype: 'float', vocab: null }],
    targets: [{ kind: 'class', num_classes: 2 }],
    image: null,
    num_classes: 2,
    class_names: ['no', 'yes'],
    notes: [],
  },
};

const FEATURES = Array.from({ length: 12 }, (_, i) => ({
  name: `feature ${i + 1}`,
  mean: (12 - i) / 100,
  std: 0.01,
}));

const ANSWERS: Record<string, unknown> = {
  permutation: {
    method: 'permutation',
    kind: 'tabular',
    metric: 'accuracy',
    baseline: 0.9,
    rows: 50,
    repeats: 5,
    features: FEATURES,
  },
  boundary: {
    method: 'boundary',
    kind: 'tabular',
    features: ['petal', 'sepal'],
    x: [0, 7],
    y: [2, 4.5],
    resolution: 64,
    png: PNG,
    classes: ['no', 'yes'],
    colors: ['#4c78a8', '#f58518'],
    grid: [[0]],
    points: [[1.4, 3.5, 0]],
  },
  weights: {
    method: 'weights',
    filters: null,
    layers: [],
  },
};

const SAMPLES: ListedSample[] = [
  { index: 4, png: PNG, label: '2', predicted: '2', probability: 0.97, correct: true },
  { index: 5, png: PNG, label: '1', predicted: '1', probability: 0.9, correct: true },
  { index: 6, png: PNG, label: '0', predicted: '0', probability: 0.8, correct: true },
  { index: 9, png: PNG, label: '7', predicted: '1', probability: 0.6, correct: false },
];

function imageAnswer(sample: number) {
  return {
    method: 'image',
    kind: 'image',
    sample: { source: 'validation', index: sample },
    image: PNG,
    size: [28, 28],
    prediction: { class: '2', index: 2, probability: 0.97 },
    target: { index: 2, class: '2' },
    saliency: { png: SALIENCY, samples: 16, noise: 0.15 },
    gradcam: { png: GRADCAM, layer: { id: 'c1', name: 'conv1', type: 'Conv2d', shape: [8, 28] } },
    occlusion: { unavailable: 'Skipped.' },
  };
}

beforeEach(() => {
  localStorage.clear();
  methods.mockReset();
  run.mockReset();
  run.mockImplementation((_id: string, body: ExplainRequest) => {
    if (body.method === 'samples') {
      return Promise.resolve({
        method: 'samples',
        kind: 'image',
        source: 'validation',
        total: 64,
        samples: SAMPLES,
      });
    }
    if (body.method === 'image') {
      return Promise.resolve(imageAnswer(body.sample ?? 0));
    }
    return Promise.resolve(ANSWERS[body.method]);
  });
});

const section = () => screen.getByRole('region', { name: 'Why the model decides' });

describe('Why the model decides: a model trained on a CSV file', () => {
  it('computes the feature importance only on request', async () => {
    methods.mockResolvedValue(available('tabular', ['samples', 'tabular', 'permutation']));
    render(<RunExplanations run={RUN} />);

    const show = await screen.findByRole('button', { name: 'Show what matters' });
    expect(within(section()).getByText(/relies on most\.$/)).toBeInTheDocument();
    expect(run).not.toHaveBeenCalled();

    await userEvent.click(show);
    expect(run).toHaveBeenCalledWith(RUN.id, { method: 'permutation' }, expect.anything());
    const table = await screen.findByRole('table', { name: 'Permutation importance' });
    // The top 10 of 12, the most important first.
    expect(within(table).getAllByRole('row')).toHaveLength(10);
    expect(within(table).getByRole('img', { name: 'feature 1: 12.0 % ± 1.0 %' })).toBeVisible();
    expect(within(table).queryByText('feature 12')).not.toBeInTheDocument();
    expect(run).toHaveBeenCalledTimes(1);
    expect(JSON.parse(localStorage.getItem(EXPLAINED_RUNS_KEY)!)).toEqual([RUN.id]);

    await userEvent.click(screen.getByRole('button', { name: 'Show all 12 features' }));
    expect(within(table).getAllByRole('row')).toHaveLength(12);
    expect(within(table).getByText('feature 12')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Show the top 10' }));
    expect(within(table).getAllByRole('row')).toHaveLength(10);
  });

  it('opens the full explanation on the feature importance tab', async () => {
    methods.mockResolvedValue(available('tabular', ['samples', 'tabular', 'permutation']));
    render(<RunExplanations run={RUN} />);
    await userEvent.click(await screen.findByRole('button', { name: 'Show what matters' }));
    await userEvent.click(
      await screen.findByRole('button', { name: 'Open the full explanation: Feature importance' })
    );
    const dialog = await screen.findByRole('dialog', { name: /Explain “Small CNN”/ });
    expect(
      await within(dialog).findByRole('tab', { name: 'Feature importance', selected: true })
    ).toBeInTheDocument();
    expect(
      await within(dialog).findByRole('img', { name: 'feature 1: 12.0 % ± 1.0 %' })
    ).toBeInTheDocument();
  });

  it('loads at once for a run explained before in this browser', async () => {
    localStorage.setItem(EXPLAINED_RUNS_KEY, JSON.stringify(['other', RUN.id]));
    methods.mockResolvedValue(available('tabular', ['samples', 'tabular', 'permutation']));
    render(<RunExplanations run={RUN} />);
    expect(await screen.findByRole('table', { name: 'Permutation importance' })).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Show what matters' })).not.toBeInTheDocument();
    // The run moves to the front of the list.
    expect(JSON.parse(localStorage.getItem(EXPLAINED_RUNS_KEY)!)).toEqual([RUN.id, 'other']);
  });

  it('shows the decision boundary of a model with two input features', async () => {
    methods.mockResolvedValue(
      available('tabular', ['samples', 'tabular', 'permutation', 'boundary'])
    );
    render(<RunExplanations run={RUN} />);
    expect(
      await within(section()).findByText(/and what it predicts across its two features\.$/)
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Show what matters' }));
    expect(
      await screen.findByRole('img', { name: 'Decision boundary over petal and sepal' })
    ).toBeInTheDocument();
    expect(run).toHaveBeenCalledWith(RUN.id, { method: 'boundary' }, expect.anything());

    await userEvent.click(
      screen.getByRole('button', { name: 'Open the full explanation: Decision boundary' })
    );
    expect(
      await screen.findByRole('tab', { name: 'Decision boundary', selected: true })
    ).toBeInTheDocument();
  });

  it('says why there is no preview, with the reason of the server', async () => {
    methods.mockResolvedValue(available('tabular', ['weights']));
    render(<RunExplanations run={RUN} />);
    expect(await screen.findByText('Not for this run (permutation).')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Show/ })).not.toBeInTheDocument();
    // The rest still works: the dialog opens on the first tab that does.
    await userEvent.click(screen.getByRole('button', { name: 'Open the full explanation' }));
    expect(
      await screen.findByRole('tab', { name: 'Inside the network', selected: true })
    ).toBeInTheDocument();
  });

  it('says when nothing can be explained on this server', async () => {
    methods.mockResolvedValue({
      ...available('tabular', []),
      unavailable: 'PyTorch is not installed on the server.',
    });
    render(<RunExplanations run={RUN} />);
    expect(await screen.findByText('PyTorch is not installed on the server.')).toBeInTheDocument();
    expect(within(section()).queryByRole('button')).not.toBeInTheDocument();
    expect(run).not.toHaveBeenCalled();
  });

  it('offers to try again when the methods cannot be loaded', async () => {
    methods.mockRejectedValueOnce(new Error('Network error'));
    methods.mockResolvedValue(available('tabular', ['samples', 'tabular', 'permutation']));
    render(<RunExplanations run={RUN} />);
    expect(await screen.findByText(/Network error/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(await screen.findByRole('button', { name: 'Show what matters' })).toBeInTheDocument();
  });
});

describe('Why the model decides: an image classifier', () => {
  it('shows Grad-CAM and saliency of a few validation images on request', async () => {
    methods.mockResolvedValue(available('image', ['samples', 'image', 'activations', 'weights']));
    render(<RunExplanations run={RUN} />);
    await userEvent.click(await screen.findByRole('button', { name: 'Show where it looks' }));

    // Two right ones and the first mistake.
    const tile = await screen.findByRole('button', {
      name: 'Explain sample 9 in full: true 7 · model 1',
    });
    expect(screen.getByRole('button', { name: /Explain sample 4 in full/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Explain sample 5 in full/ })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Explain sample 6/ })).not.toBeInTheDocument();
    const heat = () => tile.querySelectorAll('img')[1];
    await waitFor(() => expect(heat()).toHaveAttribute('src', GRADCAM));
    expect(run).toHaveBeenCalledWith(RUN.id, { method: 'image', sample: 9 }, expect.anything());

    await userEvent.click(screen.getByRole('radio', { name: 'Saliency' }));
    expect(heat()).toHaveAttribute('src', SALIENCY);
    expect(run).toHaveBeenCalledTimes(4);

    // The full explanation of that image.
    await userEvent.click(tile);
    expect(
      await screen.findByRole('tab', { name: 'Explain a prediction', selected: true })
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(run).toHaveBeenLastCalledWith(
        RUN.id,
        { method: 'image', sample: 9, target: undefined, layer: undefined },
        expect.anything()
      )
    );
  });

  it('picks right ones and the first mistake', () => {
    expect(pickSamples(SAMPLES).map((s) => s.index)).toEqual([4, 5, 9]);
    expect(pickSamples(SAMPLES.slice(0, 3)).map((s) => s.index)).toEqual([4, 5, 6]);
    expect(pickSamples([])).toEqual([]);
  });
});

describe('Why the model decides: demo data', () => {
  it('explains that there is nothing to explain and opens the inside of the network', async () => {
    methods.mockResolvedValue(available('demo', ['weights']));
    render(
      <RunExplanations run={{ ...RUN, dataset: { kind: 'random', id: null, name: 'demo' } }} />
    );
    expect(await screen.findByText(/Trained on random demo data/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Show/ })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Look inside the network' }));
    expect(
      await screen.findByRole('tab', { name: 'Inside the network', selected: true })
    ).toBeInTheDocument();
    expect(run).toHaveBeenCalledWith(RUN.id, { method: 'weights' }, expect.anything());
  });
});
