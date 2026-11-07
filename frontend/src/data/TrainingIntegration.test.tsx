/** How datasets made in the Data tab appear in the training panel. */
import { render, screen } from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { vi } from 'vitest';
import type { Dataset } from '@/api';
import { DataSourcePicker } from '@/training/DataSourcePicker';
import { STATUS } from '@/training/fixtures';
import { formatSplit, splitPercents } from '@/training/format';
import { DEFAULT_SETTINGS } from '@/training/settings';
import { SettingsForm } from '@/training/SettingsForm';

const PREPARED: Dataset = {
  id: 42,
  name: 'Churn model data',
  kind: 'csv',
  size_bytes: 51_200,
  created_at: '2026-11-07T11:00:00Z',
  info: {
    kind: 'csv',
    name: 'Churn model data',
    num_samples: 1200,
    builtin: null,
    sample_shape: null,
    train_size: null,
    val_size: null,
    num_features: 14,
    columns: [],
    has_header: true,
    delimiter: ',',
    target_kind: 'text',
    target_range: null,
    class_names: ['no', 'yes'],
    class_counts: null,
    subdir: null,
    num_classes: 2,
    predefined_split: [840, 180, 180],
  },
  prepared_from: { id: 10, name: 'Churn model data' },
};

const UPLOADED: Dataset = {
  ...PREPARED,
  id: 7,
  name: 'iris',
  info: { ...PREPARED.info, name: 'iris', predefined_split: null },
  prepared_from: null,
};

function renderPicker() {
  render(
    <MemoryRouter>
      <DataSourcePicker
        status={STATUS}
        choice={{ tab: 'upload', builtin: null, upload: 42 }}
        onChange={vi.fn()}
        datasets={[PREPARED, UPLOADED]}
        datasetsError={null}
        onReloadDatasets={vi.fn()}
        uploading={null}
        uploadError={null}
        onDismissUploadError={vi.fn()}
        onUpload={vi.fn()}
        onDelete={vi.fn()}
      />
    </MemoryRouter>
  );
}

function renderForm(predefinedSplit: number[] | null) {
  render(
    <SettingsForm
      values={{ ...DEFAULT_SETTINGS, val_split: 25 }}
      onChange={vi.fn()}
      errors={{}}
      name=""
      namePlaceholder="Tiny MLP"
      onNameChange={vi.fn()}
      devices={[]}
      defaultDevice={null}
      builtin={false}
      predefinedSplit={predefinedSplit}
    />
  );
}

describe('prepared datasets in the training panel', () => {
  it('marks prepared datasets and shows their split', () => {
    renderPicker();
    const prepared = screen.getByRole('radio', { name: /Churn model data/ });
    expect(prepared).toBeChecked();
    expect(screen.getAllByText('Prepared')).toHaveLength(1);
    expect(
      screen.getByText('Split 840 / 180 / 180 (training / validation / test)')
    ).toBeInTheDocument();
  });

  it('points to the Data tab', () => {
    renderPicker();
    expect(screen.getByRole('link', { name: 'Data tab' })).toHaveAttribute('href', '/data');
    expect(screen.getByText(/Messy CSV\? Clean, encode and analyse it/)).toBeInTheDocument();
  });

  it('fixes the validation and test shares of a predefined split', () => {
    renderForm([840, 180, 180]);
    const validation = screen.getByLabelText('Validation split');
    expect(validation).toBeDisabled();
    expect(validation).toHaveValue('15%');
    expect(screen.getByLabelText('Test split')).toHaveValue('15%');
    const test = screen.getByRole('switch', { name: /Hold out a test set/ });
    expect(test).toBeChecked();
    expect(test).toBeDisabled();
    expect(
      screen.getByText(/brings its own split from the Data tab: 840 \/ 180 \/ 180 rows/)
    ).toBeInTheDocument();
  });

  it('shows no test set when the split has none', () => {
    renderForm([1000, 200, 0]);
    expect(screen.getByRole('switch', { name: /Hold out a test set/ })).not.toBeChecked();
    expect(screen.queryByLabelText('Test split')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Validation split')).toHaveValue('16.7%');
  });

  it('keeps the share fields editable for other datasets', () => {
    renderForm(null);
    expect(screen.getByLabelText('Validation split')).toBeEnabled();
    expect(screen.getByLabelText('Validation split')).toHaveValue('25%');
    expect(screen.getByRole('switch', { name: /Hold out a test set/ })).toBeEnabled();
  });

  it('formats splits', () => {
    expect(formatSplit([84_000, 18_000, 0])).toBe('84,000 / 18,000 / 0');
    expect(splitPercents([840, 180, 180])).toEqual([70, 15, 15]);
    expect(splitPercents([0, 0, 0])).toEqual([0, 0, 0]);
  });
});
