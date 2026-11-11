import { useState } from 'react';
import { render, screen, userEvent, within } from '@test-utils';
import { vi } from 'vitest';
import type { StepParamSpec } from '@/api/types';
import { CATALOG, OTHER_SOURCE, SOURCE } from '../fixtures';
import { stepSpec } from '../steps';
import { StepParamField } from './StepParamField';

const COLUMNS = [
  { name: 'age', kind: 'numeric' as const },
  { name: 'income', kind: 'numeric' as const },
  { name: 'city', kind: 'categorical' as const },
  { name: 'signup', kind: 'datetime' as const },
];

const catalogParam = (type: string, name: string) =>
  stepSpec(CATALOG, type)!.params.find((p) => p.name === name)!;

const base: StepParamSpec = {
  name: 'x',
  kind: 'string',
  label: 'Field',
  default: null,
  help: null,
  optional: false,
};

let latest: unknown;

function Field({
  param,
  initial,
  params: extra = {},
  onChange = () => {},
  keysAreColumns = false,
  error,
}: {
  param: StepParamSpec;
  initial?: unknown;
  params?: Record<string, unknown>;
  onChange?: (value: unknown) => void;
  keysAreColumns?: boolean;
  error?: string;
}) {
  const [params, setParams] = useState<Record<string, unknown>>({
    ...extra,
    [param.name]: initial === undefined ? param.default : initial,
  });
  latest = params[param.name];
  return (
    <StepParamField
      param={param}
      value={params[param.name]}
      params={params}
      columns={COLUMNS}
      sources={[SOURCE, OTHER_SOURCE]}
      currentSourceId={SOURCE.id}
      mappingKeysAreColumns={keysAreColumns}
      error={error}
      onChange={(value) => {
        onChange(value);
        setParams({ ...params, [param.name]: value });
      }}
    />
  );
}

describe('StepParamField', () => {
  it('column: offers columns of the allowed kinds', async () => {
    const onChange = vi.fn();
    const param = { ...catalogParam('filter_rows', 'column'), column_kinds: ['numeric' as const] };
    render(<Field param={param} onChange={onChange} />);
    await userEvent.click(screen.getByRole('combobox', { name: 'Column' }));
    expect(screen.getByRole('option', { name: 'age' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'city' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('option', { name: 'income' }));
    expect(onChange).toHaveBeenLastCalledWith('income');
  });

  it('column: keeps and flags a column that is not available', () => {
    render(<Field param={catalogParam('filter_rows', 'column')} initial="gone" />);
    expect(screen.getByRole('combobox', { name: 'Column' })).toHaveValue('gone');
    expect(screen.getByText('“gone” does not exist at this step.')).toBeInTheDocument();
  });

  it('columns: shows what an empty choice means and picks columns', async () => {
    const onChange = vi.fn();
    render(<Field param={catalogParam('clip', 'columns')} onChange={onChange} />);
    const input = screen.getByPlaceholderText('All numeric columns');
    await userEvent.click(input);
    expect(screen.queryByRole('option', { name: 'city' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('option', { name: 'age' }));
    expect(onChange).toHaveBeenLastCalledWith(['age']);
  });

  it('columns: selects all columns matching the search', async () => {
    const param = catalogParam('drop_columns', 'columns');
    render(<Field param={param} initial={['city']} />);
    await userEvent.type(screen.getByRole('combobox', { name: 'Columns' }), 'IN');
    await userEvent.click(screen.getByRole('button', { name: 'Select all 1 matching “IN”' }));
    expect(latest).toEqual(['city', 'income']);
    await userEvent.type(screen.getByRole('combobox', { name: 'Columns' }), 'g');
    // age and signup match; nothing chosen twice
    await userEvent.click(screen.getByRole('button', { name: 'Select all 2 matching “g”' }));
    expect(latest).toEqual(['city', 'income', 'age', 'signup']);
    await userEvent.click(screen.getByRole('button', { name: 'Clear (4)' }));
    expect(latest).toEqual([]);
  });

  it('columns: pastes a list of names and reports unknown ones', async () => {
    const param = catalogParam('drop_columns', 'columns');
    render(<Field param={param} initial={['age']} />);
    await userEvent.click(screen.getByRole('button', { name: 'Paste a list' }));
    const box = screen.getByRole('textbox', { name: 'Paste column names for Columns' });
    await userEvent.click(box);
    await userEvent.paste("['income', 'age']\ncity nope");
    await userEvent.click(screen.getByRole('button', { name: 'Add these columns' }));
    expect(latest).toEqual(['age', 'income', 'city']);
    expect(screen.getByRole('status')).toHaveTextContent('Added 2 columns; not found: nope.');
  });

  it('columns: explains columns of the wrong kind', () => {
    render(<Field param={catalogParam('clip', 'columns')} initial={['city']} />);
    expect(
      screen.getByText('“city” is a categorical column; this needs numeric columns.')
    ).toBeInTheDocument();
  });

  it('source: offers the other tables and sends the id', async () => {
    const onChange = vi.fn();
    render(<Field param={catalogParam('join', 'source')} onChange={onChange} />);
    await userEvent.click(screen.getByRole('combobox', { name: 'Table' }));
    expect(screen.queryByRole('option', { name: /customers/ })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('option', { name: 'regions (5 rows × 3 columns)' }));
    expect(onChange).toHaveBeenLastCalledWith(2);
  });

  it('source_columns: uses the columns of the chosen table', async () => {
    const param = catalogParam('join', 'right_on');
    const { unmount } = render(<Field param={param} />);
    expect(screen.getByPlaceholderText('Choose a table first')).toBeDisabled();
    unmount();

    render(<Field param={param} params={{ source: 2 }} />);
    await userEvent.click(screen.getByRole('combobox', { name: 'Key columns there' }));
    expect(screen.getByRole('option', { name: 'region' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'age' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('option', { name: 'city' }));
    expect(latest).toEqual(['city']);
  });

  it('select: picks an option', async () => {
    render(<Field param={catalogParam('impute', 'strategy')} />);
    await userEvent.click(screen.getByRole('combobox', { name: 'Fill with' }));
    await userEvent.click(screen.getByRole('option', { name: 'Most frequent value' }));
    expect(latest).toBe('most_frequent');
  });

  it('multiselect: toggles chips and keeps the catalog order', async () => {
    render(<Field param={catalogParam('datetime_features', 'parts')} />);
    const group = screen.getByRole('group', { name: 'Parts' });
    await userEvent.click(within(group).getByRole('checkbox', { name: 'Day' }));
    expect(latest).toEqual(['year', 'month', 'day', 'weekday']);
    await userEvent.click(within(group).getByRole('checkbox', { name: 'Year' }));
    expect(latest).toEqual(['month', 'day', 'weekday']);
  });

  it('number: reports valid numbers only and explains the range', async () => {
    const onChange = vi.fn();
    render(<Field param={catalogParam('clip', 'lower_q')} onChange={onChange} />);
    const input = screen.getByLabelText('Lower quantile');
    await userEvent.clear(input);
    expect(screen.getByText('Enter a number.')).toBeInTheDocument();
    await userEvent.type(input, '0.7');
    expect(screen.getByText('Must be at most 0.5.')).toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalledWith(0.7);
    await userEvent.clear(input);
    await userEvent.type(input, '0.05');
    expect(onChange).toHaveBeenLastCalledWith(0.05);
    expect(screen.queryByText(/Must be/)).not.toBeInTheDocument();
  });

  it('int: refuses decimals', async () => {
    render(<Field param={catalogParam('impute', 'k')} />);
    const input = screen.getByLabelText('Neighbours');
    await userEvent.clear(input);
    await userEvent.type(input, '7');
    expect(latest).toBe(7);
  });

  it('bool: switches', async () => {
    render(<Field param={catalogParam('impute', 'add_indicator')} />);
    await userEvent.click(screen.getByLabelText('Add a column that marks filled values'));
    expect(latest).toBe(true);
  });

  it('string: sends text, optional empty text as null', async () => {
    render(<Field param={{ ...base, label: 'Suffix', default: '_right' }} />);
    const input = screen.getByLabelText('Suffix');
    await userEvent.clear(input);
    expect(latest).toBe('');
    await userEvent.type(input, '_r');
    expect(latest).toBe('_r');
  });

  it('string: optional fields send null when empty', async () => {
    render(<Field param={{ ...base, label: 'Name', optional: true }} initial="x" />);
    await userEvent.clear(screen.getByLabelText('Name'));
    expect(latest).toBeNull();
  });

  it('value: sends numbers as numbers and text as text', async () => {
    render(<Field param={catalogParam('filter_rows', 'value')} />);
    const input = screen.getByLabelText('Value');
    await userEvent.type(input, '42');
    expect(latest).toBe(42);
    expect(screen.getByText('number')).toBeInTheDocument();
    await userEvent.clear(input);
    await userEvent.type(input, 'Berlin');
    expect(latest).toBe('Berlin');
    expect(screen.getByText('text')).toBeInTheDocument();
    await userEvent.clear(input);
    await userEvent.type(input, '"007"');
    expect(latest).toBe('007');
    await userEvent.clear(input);
    expect(latest).toBeNull();
  });

  it('value: keeps unfinished numbers as typed', async () => {
    render(<Field param={catalogParam('filter_rows', 'value')} />);
    const input = screen.getByLabelText('Value');
    await userEvent.type(input, '1.');
    expect(input).toHaveValue('1.');
    expect(latest).toBe(1);
    await userEvent.type(input, '5');
    expect(latest).toBe(1.5);
  });

  it('list: adds tags and parses them', async () => {
    render(<Field param={catalogParam('filter_rows', 'values')} />);
    const input = screen.getByRole('combobox', { name: 'Values' });
    await userEvent.type(input, '-999{Enter}unknown{Enter}');
    expect(latest).toEqual([-999, 'unknown']);
  });

  it('mapping: edits rows of column names', async () => {
    render(<Field param={catalogParam('rename_columns', 'mapping')} keysAreColumns />);
    await userEvent.click(screen.getByRole('button', { name: 'Add column' }));
    await userEvent.type(screen.getByRole('combobox', { name: 'Column 1' }), 'age');
    expect(latest).toEqual({ age: '' });
    await userEvent.type(screen.getByLabelText('New name 1'), '2024');
    // Column names stay text, even when they look like numbers.
    expect(latest).toEqual({ age: '2024' });

    await userEvent.click(screen.getByRole('button', { name: 'Add column' }));
    await userEvent.type(screen.getByRole('combobox', { name: 'Column 2' }), 'nope');
    expect(screen.getByText('No such column at this step')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Remove row 2' }));
    expect(latest).toEqual({ age: '2024' });
    await userEvent.click(screen.getByRole('button', { name: 'Remove row 1' }));
    expect(latest).toEqual({});
  });

  it('mapping: parses new cell values and flags duplicate keys', async () => {
    const param = { ...catalogParam('rename_columns', 'mapping'), label: 'Replace' };
    render(<Field param={param} initial={{ yes: 1 }} />);
    expect(screen.getByLabelText('Value 1')).toHaveValue('yes');
    expect(screen.getByLabelText('Replace with 1')).toHaveValue('1');
    await userEvent.click(screen.getByRole('button', { name: 'Add value' }));
    await userEvent.type(screen.getByLabelText('Value 2'), 'yes');
    expect(screen.getByText('Listed twice')).toBeInTheDocument();
    await userEvent.clear(screen.getByLabelText('Value 2'));
    await userEvent.type(screen.getByLabelText('Value 2'), 'no');
    await userEvent.type(screen.getByLabelText('Replace with 2'), '0');
    expect(latest).toEqual({ yes: 1, no: 0 });
  });

  it('shows help texts and errors from the preview', () => {
    render(<Field param={catalogParam('join', 'how')} error="Choose how to join." />);
    expect(screen.getByText('Which rows to keep when a key has no match.')).toBeInTheDocument();
    expect(screen.getByText('Choose how to join.')).toBeInTheDocument();
  });
});
