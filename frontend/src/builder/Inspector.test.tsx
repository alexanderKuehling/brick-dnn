/** The Inspector of an Input layer and the dataset chosen for training. */
import { render, screen, userEvent } from '@test-utils';
import { vi } from 'vitest';
import type { Analysis, LayerSpec, Params } from '@/api/types';
import { HYBRID_ANALYSIS } from '@/training/hybridFixtures';
import { BuilderContext, type BuilderContextValue, type InputFit } from './BuilderContext';
import { GP_ANALYSIS, SPECS, VOTING_ANALYSIS } from './classicalFixtures';
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

describe('Inspector of classical bricks', () => {
  function renderBrick(
    node: LayerFlowNode | null,
    analysis: Analysis | null = GP_ANALYSIS,
    updateNode = vi.fn(),
    focusNode = vi.fn()
  ) {
    const context: BuilderContextValue = {
      specs: SPECS,
      analysis,
      errorsByNode: {},
      warningsByNode: {},
      readOnly: false,
      updateNode,
      deleteNode: vi.fn(),
      duplicateNode: vi.fn(),
      focusNode,
    };
    const nodes = Object.entries(analysis?.nodes ?? {}).map(([id, info]) => ({
      id,
      type: 'layer' as const,
      position: { x: 0, y: 0 },
      data: { type: info.type, params: info.params ?? {} },
    }));
    return render(
      <BuilderContext.Provider value={context}>
        <Inspector node={node} nodes={nodes} />
      </BuilderContext.Provider>
    );
  }
  const rbf = (params: Params): LayerFlowNode => ({
    id: 'rbf',
    type: 'layer',
    position: { x: 0, y: 0 },
    data: { type: 'RBF', params },
  });

  it('shows the bounds of a kernel only with custom bounds', async () => {
    const updateNode = vi.fn();
    const { unmount } = renderBrick(
      rbf({ length_scale: 1, hyperparameters: 'optimize' }),
      GP_ANALYSIS,
      updateNode
    );
    expect(screen.getByRole('combobox', { name: 'Hyperparameters' })).toHaveValue('Optimise');
    expect(screen.queryByLabelText('Length scale: lower bound')).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('combobox', { name: 'Hyperparameters' }));
    await userEvent.click(await screen.findByRole('option', { name: 'Custom bounds' }));
    expect(updateNode).toHaveBeenCalledWith('rbf', {
      params: { length_scale: 1, hyperparameters: 'custom bounds' },
    });
    unmount();

    renderBrick(rbf({ length_scale: 1, hyperparameters: 'custom bounds' }));
    expect(screen.getByLabelText('Length scale: lower bound')).toHaveValue('0.00001');
    expect(screen.getByLabelText('Length scale: upper bound')).toHaveValue('100000');
  });

  it('shows what a kernel brick puts out and its scikit-learn code', () => {
    renderBrick(rbf({ length_scale: 1, hyperparameters: 'optimize' }));
    expect(screen.getByText('Output')).toBeInTheDocument();
    expect(screen.getByText('kernel')).toBeInTheDocument();
    expect(screen.getByText('Fitted hyperparameters')).toBeInTheDocument();
    expect(screen.getByLabelText('Kernel as scikit-learn code')).toHaveTextContent(
      'RBF(length_scale=1.0)'
    );
  });

  it('counts the hyperparameters of a kernel sum in its kernels', () => {
    renderBrick({
      id: 'sum',
      type: 'layer',
      position: { x: 0, y: 0 },
      data: { type: 'KernelSum', params: {} },
    });
    expect(screen.getByText('Fitted hyperparameters').nextSibling).toHaveTextContent(
      'in its kernels'
    );
    expect(screen.getByLabelText('Kernel as scikit-learn code').textContent).toBe(
      'RBF(length_scale=1.0)\n+ WhiteKernel(noise_level=0.1)'
    );
  });

  it('shows the kernel of a Gaussian process, one term per line', () => {
    renderBrick({
      id: 'gp',
      type: 'layer',
      position: { x: 0, y: 0 },
      data: { type: 'GaussianProcessRegressor', params: {} },
    });
    const code = screen.getByLabelText('Kernel as scikit-learn code');
    expect(code.textContent).toBe('RBF(length_scale=1.0)\n+ WhiteKernel(noise_level=0.1)');
    // The connected kernel's hyperparameters are the ones the fit optimises.
    expect(screen.getByText('Fitted hyperparameters').nextSibling).toHaveTextContent('2');
  });

  it('says that an SVM has no parameters to count before the fit', () => {
    const analysis: Analysis = {
      ...GP_ANALYSIS,
      nodes: {
        svc: {
          type: 'SVC',
          output_shape: null,
          dtype: null,
          num_params: null,
          params: { kernel: 'rbf', C: 1 },
          output_kind: 'tensor',
          kernel: null,
          task: 'classification',
        },
      },
    };
    renderBrick(
      { id: 'svc', type: 'layer', position: { x: 0, y: 0 }, data: { type: 'SVC', params: {} } },
      analysis
    );
    expect(screen.getByText('classes from data')).toBeInTheDocument();
    expect(screen.getByText('after the fit')).toBeInTheDocument();
    expect(screen.queryByLabelText('Kernel as scikit-learn code')).not.toBeInTheDocument();
  });

  it('shows the settings of gradient boosting and its trees after the fit', async () => {
    const analysis: Analysis = {
      ...GP_ANALYSIS,
      nodes: {
        gb: {
          type: 'HistGradientBoostingClassifier',
          output_shape: null,
          dtype: null,
          num_params: null,
          params: { learning_rate: 0.1, max_iter: 200, max_depth: null, class_weight: 'none' },
          output_kind: 'tensor',
          kernel: null,
          task: 'classification',
        },
      },
      total_params: null,
      classical: {
        node: 'gb',
        type: 'HistGradientBoostingClassifier',
        estimator: 'HistGradientBoostingClassifier',
        task: 'classification',
        kernel: null,
      },
    };
    const updateNode = vi.fn();
    const node: LayerFlowNode = {
      id: 'gb',
      type: 'layer',
      position: { x: 0, y: 0 },
      data: { type: 'HistGradientBoostingClassifier', params: { class_weight: 'none' } },
    };
    renderBrick(node, analysis, updateNode);
    expect(screen.getByText('classes from data')).toBeInTheDocument();
    expect(screen.getByText('Decision trees').nextSibling).toHaveTextContent('after the fit');
    expect(screen.getByLabelText('Learning rate')).toHaveAttribute('placeholder', '0.1');
    // Empty max. depth: no limit.
    expect(screen.getByLabelText('Max. depth')).toHaveAttribute('placeholder', 'auto');
    expect(screen.getByRole('combobox', { name: 'Early stopping' })).toHaveValue(
      'Auto (above 10,000 rows)'
    );
    expect(screen.queryByLabelText('Kernel as scikit-learn code')).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('combobox', { name: 'Class weights' }));
    await userEvent.click(await screen.findByRole('option', { name: 'Balanced' }));
    expect(updateNode).toHaveBeenCalledWith('gb', { params: { class_weight: 'balanced' } });
  });

  it('summarises a gradient boosting network', () => {
    renderBrick(null, {
      ...GP_ANALYSIS,
      total_params: null,
      classical: {
        node: 'gb',
        type: 'HistGradientBoostingRegressor',
        estimator: 'HistGradientBoostingRegressor',
        task: 'regression',
        kernel: null,
      },
    });
    expect(screen.getByText('Parameters').nextSibling).toHaveTextContent(
      'decision trees, after the fit'
    );
  });

  it('summarises a classical network', () => {
    renderBrick(null);
    expect(screen.getByText('Gaussian process regression')).toBeInTheDocument();
    expect(screen.getByText('Hyperparameters the fit optimises').nextSibling).toHaveTextContent(
      '2'
    );
    expect(screen.getByLabelText('Kernel of the model')).toHaveTextContent('WhiteKernel');
    expect(screen.queryByText('Trainable parameters')).not.toBeInTheDocument();
    expect(screen.getByText(/Kernels flow along the dashed/)).toBeInTheDocument();
  });

  it('lists the models of an ensemble in the order of their weights', async () => {
    const updateNode = vi.fn();
    const focusNode = vi.fn();
    const vote: LayerFlowNode = {
      id: 'vote',
      type: 'layer',
      position: { x: 0, y: 0 },
      data: { type: 'VotingClassifier', params: { voting: 'soft', weights: [2, 1] } },
    };
    renderBrick(vote, VOTING_ANALYSIS, updateNode, focusNode);
    expect(screen.getByText('Models').nextSibling).toHaveTextContent('2');
    expect(screen.getByText('Models, in the order of their connections and weights')).toBeVisible();
    const models = screen.getByLabelText('Models of the ensemble');
    expect(models).toHaveTextContent('gb Gradient boosting classification · weight 2');
    expect(models).toHaveTextContent('svc Support vector classification · weight 1');
    await userEvent.click(screen.getByText('svc'));
    expect(focusNode).toHaveBeenCalledWith('svc');

    // Its settings: soft or hard voting, and the weights as a list of whole numbers.
    expect(screen.getByLabelText('Weights')).toHaveValue('2, 1');
    await userEvent.clear(screen.getByLabelText('Weights'));
    expect(updateNode).toHaveBeenLastCalledWith('vote', {
      params: { voting: 'soft', weights: null },
    });
    expect(screen.getByRole('combobox', { name: 'Voting' })).toHaveValue(
      'Soft (average the probabilities)'
    );
  });

  it('summarises an ensemble', () => {
    renderBrick(null, VOTING_ANALYSIS);
    expect(screen.getByText('Voting classifier')).toBeInTheDocument();
    expect(screen.getByText('2 models').nextSibling).toHaveTextContent('gb, svc');
    expect(screen.queryByText('Parameters')).not.toBeInTheDocument();
    expect(screen.getByText(/the dotted connections carry the models/)).toBeInTheDocument();
  });

  it('summarises a hybrid network: the learned features and the classical model', () => {
    renderBrick(null, HYBRID_ANALYSIS);
    expect(screen.getByText('Neural layers (PyTorch): learned features')).toBeInTheDocument();
    expect(
      screen.getByText('3 features from Linear, 131 trainable parameters')
    ).toBeInTheDocument();
    expect(screen.getByText('Classical model (scikit-learn)')).toBeInTheDocument();
  });

  it('shows why a neural layer does not fit before a classical model', () => {
    const message =
      'Gaussian process regression needs a flat feature vector, but Conv2d outputs (32, 8, 8). Add Flatten (or a Linear layer) before Gaussian process regression: its output, shape [n], is the feature vector.';
    renderBrick(null, {
      ...GP_ANALYSIS,
      valid: false,
      total_params: null,
      errors: [{ node: 'gp', message }],
    });
    expect(screen.getByText('1 problem needs fixing before export.')).toBeInTheDocument();
    expect(screen.getByText(message)).toBeInTheDocument();
  });
});
