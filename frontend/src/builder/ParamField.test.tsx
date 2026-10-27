import { render, screen, userEvent } from '@test-utils';
import { vi } from 'vitest';
import type { ParamSpec } from '@/api/types';
import { ParamField, parseIntList } from './ParamField';

const base: ParamSpec = {
  name: 'x',
  label: 'Field',
  kind: 'int',
  default: 64,
  help: '',
  options: [],
  min: 1,
  max: null,
  auto: false,
  length: null,
  show_if: null,
};

describe('ParamField', () => {
  it('shows the inferred value for auto params', () => {
    render(
      <ParamField
        param={{ ...base, label: 'Input features', auto: true, default: null }}
        value={null}
        inferred={784}
        onChange={() => {}}
      />
    );
    expect(screen.getByPlaceholderText('auto (784)')).toBeInTheDocument();
  });

  it('reports numbers and clears to null', async () => {
    const onChange = vi.fn();
    render(<ParamField param={base} value={64} onChange={onChange} />);
    const input = screen.getByLabelText('Field');
    await userEvent.clear(input);
    expect(onChange).toHaveBeenLastCalledWith(null);
    await userEvent.type(input, '7');
    expect(onChange).toHaveBeenLastCalledWith(7);
  });

  it('renders one input per axis for fixed-length lists', async () => {
    const onChange = vi.fn();
    render(
      <ParamField
        param={{ ...base, label: 'Kernel size', kind: 'ints', length: 2, default: [3, 3] }}
        value={[3, 3]}
        onChange={onChange}
      />
    );
    const width = screen.getByLabelText('Kernel size W');
    await userEvent.clear(width);
    await userEvent.type(width, '5');
    expect(onChange).toHaveBeenLastCalledWith([3, 5]);
  });

  it('keeps the other axis while one axis is being retyped', async () => {
    const onChange = vi.fn();
    render(
      <ParamField
        param={{ ...base, label: 'Kernel size', kind: 'ints', length: 2, default: [3, 3] }}
        value={[5, 7]}
        onChange={onChange}
      />
    );
    const height = screen.getByLabelText('Kernel size H');
    await userEvent.clear(height);
    expect(onChange).not.toHaveBeenCalled();
    await userEvent.type(height, '4');
    expect(onChange).toHaveBeenLastCalledWith([4, 7]);
  });

  it('parses free-form shapes', async () => {
    const onChange = vi.fn();
    render(
      <ParamField
        param={{ ...base, label: 'Shape', kind: 'ints', default: [784] }}
        value={[784]}
        onChange={onChange}
      />
    );
    const input = screen.getByLabelText('Shape');
    await userEvent.clear(input);
    await userEvent.type(input, '1, 28, 28');
    expect(onChange).toHaveBeenLastCalledWith([1, 28, 28]);
    await userEvent.type(input, ',a');
    expect(screen.getByText('Enter whole numbers separated by commas.')).toBeInTheDocument();
  });

  it('toggles booleans', async () => {
    const onChange = vi.fn();
    render(
      <ParamField
        param={{ ...base, label: 'Bias', kind: 'bool', default: true }}
        value
        onChange={onChange}
      />
    );
    await userEvent.click(screen.getByLabelText('Bias'));
    expect(onChange).toHaveBeenCalledWith(false);
  });
});

describe('parseIntList', () => {
  it('accepts commas, spaces and x', () => {
    expect(parseIntList('3, 32 x 32')).toEqual([3, 32, 32]);
    expect(parseIntList('-1, 28')).toEqual([-1, 28]);
    expect(parseIntList('1.5')).toBeNull();
    expect(parseIntList('')).toBeNull();
  });
});
