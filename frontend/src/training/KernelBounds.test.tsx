/** Kernel hyperparameters at a bound: the hint in a run's results and its undoable fix. */
import { render, screen, userEvent } from '@test-utils';
import { vi } from 'vitest';
import type { Analysis, KernelBound } from '@/api/types';
import { BuilderContext, type BuilderContextValue } from '@/builder/BuilderContext';
import { formatBound, KernelBoundsHint } from './KernelBounds';

const NOISE: KernelBound = {
  node: 'noise',
  name: 'white_1',
  type: 'WhiteKernel',
  parameter: 'noise_level',
  label: 'Noise level',
  value: 1e-5,
  bound: 1e-5,
  side: 'lower',
  classes: null,
  suggestion: {
    hyperparameters: 'custom bounds',
    noise_level_min: 0.001,
    noise_level_max: 100000,
    noise_level: 0.1,
  },
  message:
    'Noise level of white_1 ended at its lower bound 1e-05: the model may be overfitting; ' +
    'set custom bounds, e.g. ≥ 0.001.',
};

const ANALYSIS = {
  valid: true,
  errors: [],
  warnings: [],
  nodes: {
    noise: {
      type: 'WhiteKernel',
      output_shape: null,
      dtype: null,
      num_params: 1,
      params: { noise_level: 0.1 },
    },
  },
  inputs: [],
  outputs: [],
  total_params: 1,
} as Analysis;

function builder(overrides: Partial<BuilderContextValue> = {}): BuilderContextValue {
  return {
    specs: {},
    analysis: ANALYSIS,
    errorsByNode: {},
    warningsByNode: {},
    readOnly: false,
    updateNode: vi.fn(),
    patchParams: vi.fn(() => ({ noise_level: 0.1 })),
    deleteNode: vi.fn(),
    duplicateNode: vi.fn(),
    focusNode: vi.fn(),
    ...overrides,
  };
}

function renderHint(bounds: KernelBound[], context: BuilderContextValue | null) {
  return render(
    <BuilderContext.Provider value={context}>
      <KernelBoundsHint bounds={bounds} />
    </BuilderContext.Provider>
  );
}

describe('Kernel hyperparameters at a bound', () => {
  it('names the brick and sets the suggested bounds as a change that can be undone', async () => {
    const context = builder();
    renderHint([NOISE], context);
    expect(screen.getByText('A kernel hyperparameter ended at its bound')).toBeInTheDocument();
    expect(screen.getByText(NOISE.message)).toBeInTheDocument();

    await userEvent.click(
      screen.getByRole('button', { name: 'Set a higher lower bound on white_1' })
    );
    expect(context.patchParams).toHaveBeenCalledWith('noise', 'WhiteKernel', NOISE.suggestion);
    expect(context.focusNode).toHaveBeenCalledWith('noise');
    expect(
      screen.getByText(/Set on white_1: Noise level: custom bounds 0.001 … 1e\+05/)
    ).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
    expect(context.updateNode).toHaveBeenCalledWith('noise', { params: { noise_level: 0.1 } });
    expect(screen.getByRole('button', { name: /Set a higher lower bound/ })).toBeInTheDocument();
  });

  it('only offers the change for a matching brick in an editable builder', () => {
    const { unmount } = renderHint([NOISE], builder({ readOnly: true }));
    expect(screen.getByText(NOISE.message)).toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
    unmount();

    // The network on the canvas has no such brick (another network is open).
    const other = { ...ANALYSIS, nodes: {} } as Analysis;
    renderHint([NOISE], builder({ analysis: other })).unmount();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();

    // Outside the builder, and for the default kernel: the text only.
    renderHint([{ ...NOISE, node: null, suggestion: null, classes: ['a', 'b'] }], null);
    expect(screen.getByText(/\(classes: a, b\)/)).toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('shows nothing without bounds', () => {
    renderHint([], builder());
    expect(screen.queryByText(/ended at/)).not.toBeInTheDocument();
  });

  it('formats bounds like the server', () => {
    expect(formatBound(1e-5)).toBe('1e-05');
    expect(formatBound(100000)).toBe('1e+05');
    expect(formatBound(0.001)).toBe('0.001');
    expect(formatBound(50)).toBe('50');
    expect(formatBound(2.5e-7)).toBe('2.5e-07');
  });
});
