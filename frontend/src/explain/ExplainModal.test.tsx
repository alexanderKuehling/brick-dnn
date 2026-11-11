/** The Explain dialog of a finished run, with the server's answers faked. */
import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { vi } from 'vitest';
import type { RunDetail } from '@/api';
import { detail } from '@/training/fixtures';
import { ExplainContent } from './ExplainModal';
import type { ExplainMethods, ExplainRequest } from './types';

const { methods, run } = vi.hoisted(() => ({ methods: vi.fn(), run: vi.fn() }));

vi.mock('./api', async (importOriginal) => {
  const original = await importOriginal<typeof import('./api')>();
  return { ...original, explainApi: { methods, run } };
});
vi.mock('@/api', async (importOriginal) => {
  const original = await importOriginal<typeof import('@/api')>();
  return {
    ...original,
    api: {
      ...original.api,
      layers: () =>
        Promise.resolve({
          categories: [],
          layers: [{ type: 'Linear', description: 'Fully connected layer: y = xWᵀ + b.' }],
        }),
    },
  };
});

const PNG = 'data:image/png;base64,iVBORw0KGgo=';

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

const IRIS: RunDetail = {
  ...detail({ status: 'completed', has_weights: true }),
  plan: {
    task: 'classification',
    summary: '',
    inputs: [{ shape: [2], dtype: 'float', vocab: null }],
    targets: [{ kind: 'class', num_classes: 2 }],
    image: null,
    num_classes: 2,
    class_names: ['setosa', 'virginica'],
    notes: [],
  },
};

const ANSWERS: Record<string, unknown> = {
  samples: {
    method: 'samples',
    kind: 'tabular',
    source: 'validation',
    total: 30,
    columns: ['petal', 'sepal'],
    target: 'species',
    samples: [
      {
        index: 0,
        values: [1.4, 3.5],
        label: 'setosa',
        predicted: 'setosa',
        correct: true,
        probability: 0.9,
      },
      {
        index: 7,
        values: [5.1, 3.0],
        label: 'virginica',
        predicted: 'setosa',
        correct: false,
        probability: 0.6,
      },
    ],
  },
  tabular: {
    method: 'tabular',
    kind: 'tabular',
    sample: { source: 'validation', index: 0, label: 'setosa', values: [1.4, 3.5] },
    prediction: { class: 'setosa', index: 0, probability: 0.9, top: [] },
    explains: "the probability of 'setosa'",
    target: { index: 0, class: 'setosa' },
    base_value: 0.5,
    value: 0.9,
    baseline: 'the mean of the training rows',
    exact: true,
    orders: null,
    features: [
      { name: 'petal', value: 1.4, shapley: 0.35, integrated_gradients: 0.3 },
      { name: 'sepal', value: 3.5, shapley: 0.05, integrated_gradients: 0.06 },
    ],
  },
  permutation: {
    method: 'permutation',
    kind: 'tabular',
    metric: 'accuracy',
    baseline: 0.95,
    rows: 30,
    repeats: 5,
    features: [
      { name: 'petal', mean: 0.42, std: 0.03 },
      { name: 'sepal', mean: 0.01, std: 0.01 },
    ],
  },
  boundary: {
    method: 'boundary',
    kind: 'tabular',
    features: ['petal', 'sepal'],
    x: [0, 7],
    y: [2, 4.5],
    resolution: 64,
    png: PNG,
    classes: ['setosa', 'virginica'],
    colors: ['#4c78a8', '#f58518'],
    grid: [[0]],
    points: [
      [1.4, 3.5, 0],
      [5.0, 3.0, 1],
    ],
  },
  activations: {
    method: 'activations',
    kind: 'tabular',
    sample: { source: 'validation', index: 0 },
    prediction: { class: 'setosa', index: 0, probability: 0.9 },
    nodes: [
      {
        id: 'in',
        type: 'Input',
        name: 'Input',
        shape: [2],
        stats: { mean: 0, std: 1, min: -1, max: 1, zeros: 0 },
        histogram: { edges: [0, 1], counts: [2], count: 2, min: -1, max: 1 },
        view: { kind: 'vector', values: [-1, 1], total: 2 },
      },
      {
        id: 'fc',
        type: 'Linear',
        name: 'classifier',
        shape: [2],
        stats: { mean: 0.5, std: 1, min: -0.5, max: 1.5, zeros: 0 },
        histogram: { edges: [0, 1], counts: [2], count: 2, min: -0.5, max: 1.5 },
        view: { kind: 'vector', values: [1.5, -0.5], total: 2 },
      },
    ],
  },
  weights: {
    method: 'weights',
    filters: null,
    layers: [
      {
        id: 'fc',
        type: 'Linear',
        name: 'classifier',
        params: [
          {
            name: 'weight',
            shape: [2, 2],
            histogram: { edges: [0, 1], counts: [4], count: 4, mean: 0, std: 1 },
            norm: 2,
          },
        ],
      },
    ],
  },
};

beforeEach(() => {
  methods.mockReset();
  run.mockReset();
  run.mockImplementation((_id: string, body: ExplainRequest) =>
    Promise.resolve(ANSWERS[body.method])
  );
});

describe('Explaining a model trained on a CSV file', () => {
  it('explains the first validation row and others on request', async () => {
    methods.mockResolvedValue(
      available('tabular', [
        'samples',
        'tabular',
        'permutation',
        'boundary',
        'activations',
        'weights',
      ])
    );
    render(<ExplainContent run={IRIS} />);

    expect(await screen.findByText('Pick a row')).toBeInTheDocument();
    expect(await screen.findByRole('img', { name: 'petal: +0.35' })).toBeInTheDocument();
    expect(run).toHaveBeenCalledWith(IRIS.id, { method: 'tabular', sample: 0 }, expect.anything());
    expect(screen.getByText(/^The probability of 'setosa':/)).toHaveTextContent(
      "The probability of 'setosa': 0.5 for an average row (the mean of the training rows), 0.9 for this one."
    );

    // Integrated gradients instead of Shapley values.
    await userEvent.click(screen.getByRole('radio', { name: 'Integrated gradients' }));
    expect(screen.getByRole('img', { name: 'petal: +0.3' })).toBeInTheDocument();

    // Another row: the one the model gets wrong.
    const rows = within(screen.getByRole('table', { name: 'Validation rows' })).getAllByRole('row');
    await userEvent.click(rows[2]);
    await waitFor(() =>
      expect(run).toHaveBeenCalledWith(IRIS.id, { method: 'tabular', sample: 7 }, expect.anything())
    );
  });

  it('shows permutation importance, the boundary and the inside of the network', async () => {
    methods.mockResolvedValue(
      available('tabular', [
        'samples',
        'tabular',
        'permutation',
        'boundary',
        'activations',
        'weights',
      ])
    );
    render(<ExplainContent run={IRIS} />);

    await userEvent.click(await screen.findByRole('tab', { name: 'Feature importance' }));
    expect(await screen.findByRole('img', { name: 'petal: 42.0 % ± 3.0 %' })).toBeInTheDocument();
    expect(screen.getByText(/Validation accuracy 95.0 % on 30 rows/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole('tab', { name: 'Decision boundary' }));
    const plot = await screen.findByRole('img', { name: 'Decision boundary over petal and sepal' });
    expect(plot.querySelectorAll('circle')).toHaveLength(2);
    expect(screen.getByText('virginica')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('tab', { name: 'Inside the network' }));
    expect(await screen.findByText('Fully connected layer: y = xWᵀ + b.')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /1\. Input/ }));
    expect(screen.getByRole('img', { name: '2 values from -1 to 1' })).toBeInTheDocument();
    expect(await screen.findByText('Learned weights')).toBeInTheDocument();
    expect(
      screen.getByRole('img', { name: 'Distribution of classifier.weight' })
    ).toBeInTheDocument();
  });

  it('disables what does not work and says why', async () => {
    methods.mockResolvedValue(available('tabular', ['samples', 'tabular', 'weights']));
    render(<ExplainContent run={IRIS} />);
    const boundary = await screen.findByRole('tab', { name: 'Decision boundary' });
    expect(boundary).toBeDisabled();
    expect(boundary).toHaveAttribute('title', 'Not for this run (boundary).');
  });

  it('shows errors with a way to try again', async () => {
    methods.mockResolvedValue(available('tabular', ['samples', 'tabular']));
    run.mockImplementation((_id: string, body: ExplainRequest) =>
      body.method === 'tabular'
        ? Promise.reject(new Error('The server is busy with other predictions.'))
        : Promise.resolve(ANSWERS[body.method])
    );
    render(<ExplainContent run={IRIS} />);
    expect(
      await screen.findByText('The server is busy with other predictions.')
    ).toBeInTheDocument();
    run.mockImplementation((_id: string, body: ExplainRequest) =>
      Promise.resolve(ANSWERS[body.method])
    );
    await userEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(await screen.findByRole('img', { name: 'petal: +0.35' })).toBeInTheDocument();
  });
});

describe('Explaining an image model', () => {
  const MNIST: RunDetail = {
    ...IRIS,
    plan: { ...IRIS.plan!, num_classes: 3, class_names: ['0', '1', '2'] },
    state: {
      ...IRIS.state!,
      preprocessing: {
        kind: 'image',
        channels: 1,
        height: 28,
        width: 28,
        class_names: ['0', '1', '2'],
      },
    },
  };
  const IMAGE = {
    method: 'image',
    kind: 'image',
    sample: { source: 'validation', index: 4, label: '2' },
    image: PNG,
    size: [28, 28],
    prediction: {
      class: '2',
      index: 2,
      probability: 0.97,
      top: [
        { class: '2', index: 2, probability: 0.97 },
        { class: '1', index: 1, probability: 0.02 },
      ],
    },
    target: { index: 2, class: '2' },
    saliency: { png: PNG, grid: [[1]], samples: 16, noise: 0.15 },
    gradcam: {
      png: PNG,
      grid: [[1]],
      layer: { id: 'c2', name: 'conv2', type: 'Conv2d', shape: [16, 14, 14] },
      layers: [
        { id: 'c1', name: 'conv1', type: 'Conv2d', shape: [8, 28, 28] },
        { id: 'c2', name: 'conv2', type: 'Conv2d', shape: [16, 14, 14] },
      ],
    },
    occlusion: { unavailable: 'Skipped: it would take too long for this network.' },
  };

  it('shows the maps over the picked image and explains another class', async () => {
    methods.mockResolvedValue(available('image', ['samples', 'image', 'activations', 'weights']));
    run.mockImplementation((_id: string, body: ExplainRequest) =>
      Promise.resolve(
        body.method === 'samples'
          ? {
              method: 'samples',
              kind: 'image',
              source: 'validation',
              total: 64,
              samples: [
                {
                  index: 4,
                  png: PNG,
                  label: '2',
                  predicted: '2',
                  probability: 0.97,
                  correct: true,
                },
              ],
            }
          : {
              ...IMAGE,
              target: body.target === undefined ? IMAGE.target : { index: 1, class: '1' },
            }
      )
    );
    render(<ExplainContent run={MNIST} />);
    expect(
      await screen.findByRole('button', { name: 'Sample 4: 2, predicted 2' })
    ).toBeInTheDocument();
    expect(await screen.findByText('Grad-CAM')).toBeInTheDocument();
    expect(screen.getByText('at conv2 (16 × 14 × 14)')).toBeInTheDocument();
    expect(
      screen.getByText('Skipped: it would take too long for this network.')
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Draw' })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('combobox', { name: 'Explain the class' }));
    await userEvent.click(await screen.findByRole('option', { name: '1' }));
    await waitFor(() =>
      expect(run).toHaveBeenCalledWith(
        MNIST.id,
        { method: 'image', sample: 4, target: 1, layer: undefined },
        expect.anything()
      )
    );
    expect(await screen.findByText(/The maps show what speaks for “1”/)).toBeInTheDocument();
  });
});
