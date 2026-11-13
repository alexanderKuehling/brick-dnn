import { GP_ANALYSIS, VOTING_ANALYSIS } from './classicalFixtures';
import {
  formatCount,
  formatOutput,
  formatShape,
  networkSize,
  optionLabel,
  pyFloat,
  summarize,
} from './format';

describe('format', () => {
  it('formats shapes', () => {
    expect(formatShape([1, 28, 28])).toBe('1 × 28 × 28');
    expect(formatShape(null)).toBe('?');
  });

  it('formats parameter counts', () => {
    expect(formatCount(1290)).toBe('1,290');
    expect(formatCount(235146)).toBe('235.1k');
    expect(formatCount(2_824_706)).toBe('2.8M');
    expect(formatCount(null)).toBe('–');
  });

  it('labels activation functions', () => {
    expect(optionLabel('function', 'leaky_relu')).toBe('LeakyReLU');
    expect(optionLabel('mode', 'mean')).toBe('mean');
  });

  it('labels the options of kernels and SVMs', () => {
    expect(optionLabel('hyperparameters', 'optimize')).toBe('Optimise');
    expect(optionLabel('hyperparameters', 'custom bounds')).toBe('Custom bounds');
    expect(optionLabel('hyperparameters', 'fixed')).toBe('Fixed');
    expect(optionLabel('kernel', 'poly')).toBe('Polynomial');
    expect(optionLabel('class_weight', 'balanced')).toBe('Balanced');
  });

  it('summarizes layers', () => {
    expect(summarize('Linear', { in_features: 784, out_features: 10, bias: true })).toBe(
      '784 → 10'
    );
    expect(summarize('Linear', { in_features: null, out_features: 10, bias: false })).toBe(
      '? → 10 · no bias'
    );
    expect(summarize('Conv2d', { out_channels: 32, kernel_size: [3, 3], stride: [2, 2] })).toBe(
      '32 filters · 3×3 · stride 2'
    );
    expect(
      summarize('ConvTranspose2d', { out_channels: 64, kernel_size: [2, 2], stride: [2, 2] })
    ).toBe('64 filters · 2×2 · stride 2');
    expect(summarize('Input', { shape: [64], dtype: 'long' })).toBe('64 · token ids');
    expect(
      summarize('LSTM', {
        hidden_size: 128,
        num_layers: 2,
        bidirectional: true,
        return_sequences: false,
      })
    ).toBe('hidden 128 · 2 layers · bi');
  });

  it('writes floats like Python', () => {
    expect(pyFloat(1)).toBe('1.0');
    expect(pyFloat(0.1)).toBe('0.1');
    expect(pyFloat(1e-10)).toBe('1e-10');
    expect(pyFloat(null)).toBe('?');
  });

  it('summarizes kernels with their starting values', () => {
    expect(summarize('RBF', { length_scale: 1, hyperparameters: 'optimize' })).toBe('RBF(ℓ=1.0)');
    expect(summarize('RBF', { length_scale: 2.5, hyperparameters: 'fixed' })).toBe(
      'RBF(ℓ=2.5) · fixed'
    );
    expect(summarize('Matern', { length_scale: 1, nu: '0.5' })).toBe('Matérn(ℓ=1.0, ν=0.5)');
    expect(summarize('ExpSineSquared', { length_scale: 1, periodicity: 7 })).toBe(
      'Periodic(ℓ=1.0, p=7.0)'
    );
    expect(summarize('WhiteKernel', { noise_level: 0.1, hyperparameters: 'custom bounds' })).toBe(
      'White(noise=0.1) · bounded'
    );
    expect(summarize('KernelSum', {}, { info: GP_ANALYSIS.nodes.sum })).toBe('RBF + White');
    expect(summarize('KernelProduct', {})).toBe('product of kernels');
  });

  it('summarizes classical models', () => {
    const gp = GP_ANALYSIS.nodes.gp;
    expect(summarize('GaussianProcessRegressor', {}, { info: gp, kernelConnected: true })).toBe(
      'RBF + White'
    );
    expect(summarize('GaussianProcessRegressor', {}, { info: gp })).toBe('default kernel');
    expect(summarize('SVC', { kernel: 'rbf', C: 1, class_weight: 'balanced' })).toBe(
      'RBF kernel · C = 1 · balanced'
    );
    expect(summarize('SVR', { kernel: 'poly', degree: 3, C: 10, epsilon: 0.1 })).toBe(
      'Polynomial kernel (degree 3) · C = 10 · ε = 0.1'
    );
    expect(
      summarize('HistGradientBoostingClassifier', {
        max_iter: 200,
        learning_rate: 0.1,
        max_leaf_nodes: 31,
        max_depth: null,
        class_weight: 'balanced',
      })
    ).toBe('200 iterations · rate 0.1 · 31 leaves · balanced');
    expect(
      summarize('HistGradientBoostingRegressor', {
        max_iter: 500,
        learning_rate: 0.05,
        max_leaf_nodes: 15,
        max_depth: 6,
      })
    ).toBe('500 iterations · rate 0.05 · 15 leaves · depth 6');
    expect(optionLabel('early_stopping', 'auto')).toBe('Auto (above 10,000 rows)');
  });

  it('describes outputs of kernels and classifiers', () => {
    expect(formatOutput(GP_ANALYSIS.nodes.rbf)).toBe('kernel');
    expect(formatOutput(GP_ANALYSIS.nodes.gp)).toBe('1');
    expect(
      formatOutput({ ...GP_ANALYSIS.nodes.gp, output_shape: null, task: 'classification' })
    ).toBe('classes from data');
    expect(formatOutput(null)).toBe('?');
  });

  it('tells the size of a network', () => {
    expect(networkSize({ ...GP_ANALYSIS, family: 'neural', total_params: 1290 })).toBe(
      '1,290 params'
    );
    expect(networkSize(GP_ANALYSIS)).toBe('2 hyperparameters');
    const svm = {
      ...GP_ANALYSIS,
      total_params: null,
      classical: { ...GP_ANALYSIS.classical!, type: 'SVC', task: 'classification' as const },
    };
    expect(networkSize(svm)).toBe('SVM classifier');
    const boosting = {
      ...svm,
      classical: {
        ...svm.classical,
        type: 'HistGradientBoostingRegressor',
        task: 'regression' as const,
      },
    };
    expect(networkSize(boosting)).toBe('Gradient boosting regression');
    expect(
      networkSize({
        ...boosting,
        classical: { ...boosting.classical, task: 'classification' as const },
      })
    ).toBe('Gradient boosting classifier');
    expect(networkSize(VOTING_ANALYSIS)).toBe('Ensemble of 2 models');
  });

  it('summarizes ensembles with their models', () => {
    const info = VOTING_ANALYSIS.nodes.vote;
    expect(summarize('VotingClassifier', { voting: 'soft', weights: [2, 1] }, { info })).toBe(
      'soft voting · weights 2:1 · gb, svc'
    );
    // Before the analysis knows the models.
    expect(summarize('VotingClassifier', { voting: 'hard', weights: null })).toBe('hard voting');
    expect(summarize('VotingRegressor', { weights: null }, { info })).toBe('average · gb, svc');
    expect(summarize('StackingClassifier', { cv: 5, passthrough: false }, { info })).toBe(
      'logistic on 5 folds · gb, svc'
    );
    expect(summarize('StackingRegressor', { cv: 3, passthrough: true })).toBe(
      'ridge on 3 folds · + features'
    );
    expect(optionLabel('voting', 'soft')).toBe('Soft (average the probabilities)');
  });
});
