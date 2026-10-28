import type { Dataset } from '@/api/types';
import { detail, summary } from './fixtures';
import {
  datasetDetails,
  formatBytes,
  formatDuration,
  formatMetric,
  formatPercent,
  progressLabel,
  resultText,
  weightsSnippet,
} from './format';
import {
  DEFAULT_SETTINGS,
  loadSettings,
  saveSettings,
  toConfig,
  validateSettings,
} from './settings';

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

  it('builds the config for the API', () => {
    expect(toConfig({ ...DEFAULT_SETTINGS, val_split: 15, seed: 7 })).toEqual({
      epochs: 10,
      batch_size: 64,
      learning_rate: 0.001,
      optimizer: 'adam',
      weight_decay: 0,
      val_split: 0.15,
      device: 'auto',
      max_samples: null,
      seed: 7,
    });
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
});
