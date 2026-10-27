import { formatCount, formatShape, optionLabel, summarize } from './format';

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
});
