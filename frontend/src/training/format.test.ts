import type { Dataset } from '@/api/types';
import { detail, summary, TUNING } from './fixtures';
import {
  datasetDetails,
  describeParams,
  formatBytes,
  formatDuration,
  formatMetric,
  formatPercent,
  formatSetting,
  headlineMetric,
  progressLabel,
  resultText,
  searchedSettings,
  searchLabel,
  weightsSnippet,
} from './format';
import {
  DEFAULT_SETTINGS,
  loadSettings,
  plannedTrials,
  saveSettings,
  toConfig,
  toTuning,
  validateSettings,
  type SettingsValues,
} from './settings';

const SEARCH: SettingsValues = { ...DEFAULT_SETTINGS, search: true };

describe('training format', () => {
  it('formats numbers, sizes and durations', () => {
    expect(formatPercent(0.9812)).toBe('98.1%');
    expect(formatPercent(null)).toBe('–');
    expect(formatMetric(0.123456)).toBe('0.1235');
    expect(formatMetric(1234.5)).toBe('1,235');
    expect(formatBytes(512)).toBe('512 bytes');
    expect(formatBytes(4551)).toBe('4.4 KB');
    expect(formatBytes(170 * 1024 * 1024)).toBe('170 MB');
    expect(formatDuration(42)).toBe('42 s');
    expect(formatDuration(192)).toBe('3 min 12 s');
    expect(formatDuration(3900)).toBe('1 h 5 min');
  });

  it('describes runs', () => {
    expect(progressLabel(summary())).toBe('Epoch 4/10');
    expect(progressLabel(summary({ status: 'preparing' }))).toBe('Preparing');
    expect(resultText(summary({ status: 'completed' }))).toBe('validation accuracy 95.0%');
    expect(
      resultText(
        summary({ metrics: { train_loss: 1, val_loss: 2, val_accuracy: null, val_rmse: 0.5 } })
      )
    ).toBe('validation RMSE 0.5');
  });

  it('prefers the test result', () => {
    const metrics = {
      train_loss: 0.1,
      val_loss: 0.2,
      val_accuracy: 0.96,
      val_rmse: null,
      test_loss: 0.25,
      test_accuracy: 0.951,
      test_rmse: null,
    };
    expect(headlineMetric(metrics)).toEqual(['95.1%', 'test accuracy']);
    expect(headlineMetric({ ...metrics, test_accuracy: null, val_accuracy: null })).toEqual([
      '0.25',
      'test loss',
    ]);
    expect(headlineMetric({ ...metrics, test_accuracy: null, test_rmse: 4.2 })).toEqual([
      '4.2',
      'test RMSE',
    ]);
    // Runs from before test sets existed have no test keys.
    expect(
      headlineMetric({ train_loss: 1, val_loss: 0.5, val_accuracy: null, val_rmse: null })
    ).toEqual(['0.5', 'validation loss']);
    expect(headlineMetric(null)).toBeNull();
    expect(resultText(summary({ status: 'completed', metrics }))).toBe('test accuracy 95.1%');
  });

  it('describes searches', () => {
    const search = summary({ tuning: TUNING, trials_done: 2, epochs_done: 1 });
    expect(progressLabel(search)).toBe('Trial 3/4');
    expect(progressLabel({ ...search, trials_done: 4 })).toBe('Trial 4/4');
    expect(searchLabel(TUNING)).toBe('search · 4 trials');
    expect(searchLabel({ ...TUNING, method: 'grid', trials: 1 })).toBe('grid search · 1 trial');
    expect(searchedSettings(TUNING)).toEqual(['learning_rate', 'batch_size']);
    expect(searchedSettings(null)).toEqual([]);
    expect(
      resultText({
        ...search,
        status: 'completed',
        best_trial: { trial: 3, params: {}, val_loss: 0.1, val_accuracy: 0.97, val_rmse: null },
      })
    ).toBe('best trial 3, validation accuracy 95.0%');
  });

  it('formats searched settings', () => {
    expect(formatSetting('learning_rate', 0.0031234)).toBe('0.00312');
    expect(formatSetting('learning_rate', 0.00001)).toBe('0.00001');
    expect(formatSetting('batch_size', 64)).toBe('64');
    expect(formatSetting('optimizer', 'adamw')).toBe('AdamW');
    expect(formatSetting('weight_decay', 0)).toBe('0');
    expect(formatSetting('weight_decay', undefined)).toBe('–');
    expect(describeParams({ optimizer: 'sgd', learning_rate: 0.00312, batch_size: 16 })).toBe(
      'learning rate 0.00312, batch size 16, SGD'
    );
  });

  it('describes uploaded datasets', () => {
    const csv = {
      id: 1,
      name: 'iris',
      kind: 'csv',
      size_bytes: 4551,
      created_at: '',
      info: { num_samples: 150, num_features: 4, num_classes: 3, target_kind: 'text' },
    } as Dataset;
    expect(datasetDetails(csv)).toBe('150 rows · 4 features · 3 classes · 4.4 KB');
  });

  it('writes a snippet that prepares images like the training did', () => {
    const run = detail({ status: 'completed', has_weights: true });
    run.state!.preprocessing = {
      kind: 'image',
      channels: 1,
      height: 28,
      width: 28,
      layout: 'flat',
      mean: [0.1307],
      std: [0.3081],
      class_names: ['zero', 'one'],
    };
    const code = weightsSnippet(run, 'small_cnn.py', 'small_cnn.pt');
    expect(code).toContain('from small_cnn import NeuralNetwork');
    expect(code).toContain('model.load_state_dict(torch.load("small_cnn.pt"');
    expect(code).toContain('model.eval()');
    expect(code).toContain('.convert("L")');
    expect(code).toContain('image.resize((28, 28)');
    expect(code).toContain('TF.normalize(x, mean=[0.1307], std=[0.3081])');
    expect(code).toContain('x = x.flatten()');
    expect(code).toContain('classes = ["zero", "one"]');
  });

  it('writes a snippet for CSV regression', () => {
    const run = detail({
      status: 'completed',
      has_weights: true,
      task: 'regression',
      dataset: { kind: 'upload', id: 3, name: 'houses' },
    });
    run.state!.class_names = null;
    run.state!.preprocessing = {
      kind: 'tabular',
      columns: ['rooms', 'age'],
      target: 'price',
      mean: [3.2, 41.5],
      std: [1.1, 12.25],
      target_mean: 250000,
      target_std: 80000,
      class_names: null,
    };
    const code = weightsSnippet(run, 'm.py', 'm.pt');
    expect(code).toContain('# rooms, age');
    expect(code).toContain('features = [0.0] * 2');
    expect(code).toContain('mean = torch.tensor([3.2, 41.5])');
    expect(code).toContain('output.item() * 80000 + 250000');
    expect(code).not.toContain('classes =');
  });
});

describe('training settings', () => {
  beforeEach(() => localStorage.clear());

  it('checks the limits of the backend', () => {
    expect(validateSettings(DEFAULT_SETTINGS)).toEqual({});
    const errors = validateSettings({
      ...DEFAULT_SETTINGS,
      epochs: 0,
      batch_size: '',
      learning_rate: 0,
      val_split: 60,
      max_samples: 5,
      seed: 1.5,
    });
    expect(Object.keys(errors).sort()).toEqual(
      ['batch_size', 'epochs', 'learning_rate', 'max_samples', 'seed', 'val_split'].sort()
    );
  });

  it('checks the test split only when it is on', () => {
    expect(validateSettings({ ...DEFAULT_SETTINGS, test_split: 90 })).toEqual({});
    const on = { ...DEFAULT_SETTINGS, test_enabled: true };
    expect(validateSettings(on)).toEqual({});
    expect(validateSettings({ ...on, test_split: 2 })).toEqual({
      test_split: 'Between 5% and 50%',
    });
    expect(validateSettings({ ...on, val_split: 50, test_split: 35 })).toEqual({
      test_split: 'Validation and test together: at most 80%',
    });
  });

  it('builds the config for the API', () => {
    expect(toConfig({ ...DEFAULT_SETTINGS, val_split: 15, seed: 7 })).toEqual({
      epochs: 10,
      batch_size: 64,
      learning_rate: 0.001,
      optimizer: 'adam',
      weight_decay: 0,
      val_split: 0.15,
      test_split: 0,
      device: 'auto',
      max_samples: null,
      seed: 7,
    });
    const test = { ...DEFAULT_SETTINGS, test_enabled: true, test_split: 12.5 };
    expect(toConfig(test).test_split).toBe(0.125);
    // Built-in datasets only switch their official test images on.
    expect(toConfig(test, { builtin: true }).test_split).toBe(0.2);
    expect(toConfig({ ...test, test_enabled: false }, { builtin: true }).test_split).toBe(0);
  });

  it('builds a random search', () => {
    expect(toTuning(DEFAULT_SETTINGS)).toBeNull();
    expect(toTuning(SEARCH)).toEqual({
      method: 'random',
      trials: 10,
      space: { learning_rate: { min: 0.0001, max: 0.01 }, batch_size: [32, 64, 128] },
    });
    expect(
      toTuning({
        ...SEARCH,
        tune_learning_rate: false,
        tune_batch_size: false,
        tune_optimizer: true,
        optimizers: ['sgd', 'adam'],
        tune_weight_decay: true,
        weight_decays: [0.001, 0],
      })
    ).toEqual({
      method: 'random',
      trials: 10,
      space: { optimizer: ['adam', 'sgd'], weight_decay: [0, 0.001] },
    });
    // Searched settings are left out of the config; the others stay.
    const config = toConfig(SEARCH);
    expect(config.learning_rate).toBeUndefined();
    expect(config.batch_size).toBeUndefined();
    expect(config.optimizer).toBe('adam');
    expect(config.weight_decay).toBe(0);
  });

  it('builds a grid search', () => {
    const grid: SettingsValues = { ...SEARCH, search_method: 'grid', lr_values: [0.01, 0.0001] };
    expect(toTuning(grid)).toEqual({
      method: 'grid',
      space: { learning_rate: [0.0001, 0.01], batch_size: [32, 64, 128] },
    });
    expect(plannedTrials(grid)).toBe(6);
    expect(validateSettings(grid)).toEqual({});
  });

  it('counts the trials like the server', () => {
    expect(plannedTrials(SEARCH)).toBe(10);
    expect(plannedTrials({ ...SEARCH, search_trials: '' })).toBeNull();
    // Only lists: every combination at most once.
    const lists = { ...SEARCH, tune_learning_rate: false, batch_sizes: [16, 32] };
    expect(plannedTrials(lists)).toBe(2);
    expect(plannedTrials({ ...lists, tune_optimizer: true })).toBe(4);
  });

  it('checks the search settings', () => {
    expect(validateSettings(SEARCH)).toEqual({});
    // Searched settings are not checked in their fixed fields.
    expect(validateSettings({ ...SEARCH, learning_rate: '', batch_size: 0 })).toEqual({});
    expect(validateSettings({ ...SEARCH, search_trials: 0, lr_min: 0.1, lr_max: 0.01 })).toEqual({
      search_trials: 'Between 1 and 100',
      lr_min: 'At most the maximum',
    });
    expect(validateSettings({ ...SEARCH, lr_max: 0 }).lr_max).toBe('Greater than 0, at most 10');
    expect(validateSettings({ ...SEARCH, batch_sizes: [] })).toEqual({
      batch_sizes: 'Choose at least one',
    });
    expect(
      validateSettings({ ...SEARCH, tune_learning_rate: false, tune_batch_size: false }).search
    ).toBe('Choose at least one setting to search.');
    const big: SettingsValues = {
      ...SEARCH,
      search_method: 'grid',
      lr_values: [0.1, 0.01, 0.001, 0.0001, 0.00001],
      batch_sizes: [8, 16, 32, 64, 128],
      tune_weight_decay: true,
      weight_decays: [0, 0.00001, 0.0001, 0.001, 0.01],
    };
    expect(validateSettings(big).search).toMatch(/^The grid has 125 combinations/);
    // The search is off: its fields do not matter.
    expect(validateSettings({ ...big, search: false })).toEqual({});
  });

  it('remembers the settings and ignores broken values', () => {
    saveSettings({ ...DEFAULT_SETTINGS, epochs: 25, max_samples: 1000 });
    expect(loadSettings()).toMatchObject({ epochs: 25, max_samples: 1000 });
    localStorage.setItem(
      'brickdnn:training-settings',
      JSON.stringify({ epochs: 'many', optimizer: 'rmsprop' })
    );
    expect(loadSettings()).toMatchObject({ epochs: 10, optimizer: 'adam' });
  });

  it('remembers the search and the test split', () => {
    saveSettings({
      ...SEARCH,
      test_enabled: true,
      test_split: 15,
      search_method: 'grid',
      batch_sizes: [16, 256],
      optimizers: ['sgd'],
    });
    expect(loadSettings()).toMatchObject({
      test_enabled: true,
      test_split: 15,
      search: true,
      search_method: 'grid',
      batch_sizes: [16, 256],
      optimizers: ['sgd'],
    });
  });

  it('loads settings saved before searches and test sets existed', () => {
    localStorage.setItem(
      'brickdnn:training-settings',
      JSON.stringify({ epochs: 3, val_split: 10, optimizer: 'sgd' })
    );
    const values = loadSettings();
    expect(values).toEqual({ ...DEFAULT_SETTINGS, epochs: 3, val_split: 10, optimizer: 'sgd' });
    expect(toTuning(values)).toBeNull();
    expect(toConfig(values).test_split).toBe(0);
  });

  it('ignores broken search settings', () => {
    localStorage.setItem(
      'brickdnn:training-settings',
      JSON.stringify({
        search: 'yes',
        test_enabled: 1,
        search_method: 'bayes',
        batch_sizes: [64, 'x', 3, 64],
        optimizers: 'adam',
        lr_values: null,
        tune_optimizer: true,
      })
    );
    expect(loadSettings()).toMatchObject({
      search: false,
      test_enabled: false,
      search_method: 'random',
      batch_sizes: [64],
      optimizers: ['adam', 'sgd'],
      lr_values: [0.01, 0.001, 0.0001],
      tune_optimizer: true,
    });
  });
});
