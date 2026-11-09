/** The Inspector of an Input layer and the dataset chosen for training. */
import { render, screen, userEvent } from '@test-utils';
import { vi } from 'vitest';
import type { LayerSpec } from '@/api/types';
import { BuilderContext, type BuilderContextValue, type InputFit } from './BuilderContext';
import type { LayerFlowNode } from './graph';
import { Inspector } from './Inspector';

const INPUT_SPEC = {
  type: 'Input',
  label: 'Input',
  category: 'Input',
  description: 'Entry point of the network.',
  params: [
    {
      name: 'shape',
      label: 'Shape',
      kind: 'ints',
      default: [784],
      help: '',
      options: [],
      min: 1,
      max: null,
      auto: false,
      length: null,
      show_if: null,
    },
  ],
  min_inputs: 0,
  max_inputs: 0,
} as LayerSpec;

const NODE: LayerFlowNode = {
  id: 'in',
  type: 'layer',
  position: { x: 0, y: 0 },
  data: { type: 'Input', params: { shape: [784] } },
};

function renderInspector(inputFit: InputFit | null, readOnly = false) {
  const context: BuilderContextValue = {
    specs: { Input: INPUT_SPEC },
    analysis: null,
    errorsByNode: {},
    warningsByNode: {},
    readOnly,
    updateNode: vi.fn(),
    deleteNode: vi.fn(),
    duplicateNode: vi.fn(),
    focusNode: vi.fn(),
    inputFit,
  };
  render(
    <BuilderContext.Provider value={context}>
      <Inspector node={NODE} nodes={[NODE]} />
    </BuilderContext.Provider>
  );
}

describe('Inspector of an Input layer', () => {
  it('offers to set the Input from the training data', async () => {
    const apply = vi.fn();
    renderInspector({
      nodeId: 'in',
      datasetName: 'iris',
      matches: false,
      summary: 'Input shape 784 → 4',
      apply,
    });
    expect(screen.getByText('Training data “iris”: Input shape 784 → 4')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Set from dataset' }));
    expect(apply).toHaveBeenCalledTimes(1);
  });

  it('shows that the Input matches the training data', () => {
    renderInspector({
      nodeId: 'in',
      datasetName: 'iris',
      matches: true,
      summary: null,
      apply: null,
    });
    expect(screen.getByText('Matches “iris”')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Set from dataset' })).not.toBeInTheDocument();
  });

  it('shows nothing for other layers, without data or to viewers', () => {
    const fit: InputFit = {
      nodeId: 'other',
      datasetName: 'iris',
      matches: false,
      summary: 'Input shape 784 → 4',
      apply: vi.fn(),
    };
    renderInspector(fit);
    expect(screen.queryByText(/iris/)).not.toBeInTheDocument();
    renderInspector({ ...fit, nodeId: 'in' }, true);
    expect(screen.queryByText(/iris/)).not.toBeInTheDocument();
    renderInspector(null);
    expect(screen.queryByText(/iris/)).not.toBeInTheDocument();
  });
});
