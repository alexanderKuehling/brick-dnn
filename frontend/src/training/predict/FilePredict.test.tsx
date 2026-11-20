/** "Predict a file" in the prediction panel: upload, options, summary, preview, download. */
import {
  act,
  axeViolations,
  fireEvent,
  render,
  screen,
  userEvent,
  waitFor,
  within,
} from '@test-utils';
import { afterEach, beforeEach, vi } from 'vitest';
import { api, ApiError, type FilePrediction, type PredictionForm } from '@/api';
import { downloadBlob } from '@/builder/download';
import { setLanguage } from '@/i18n';
import { PredictionPanel } from './PredictionPanel';
import { fileColumns, hasMoreLines } from './rows';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      training: {
        runs: {
          predictForm: vi.fn(),
          predict: vi.fn(),
          predictFile: vi.fn(),
          downloadPredictedFile: vi.fn(),
          discardPredictedFile: vi.fn(),
        },
      },
    },
  };
});
vi.mock('@/builder/download', () => ({ downloadBlob: vi.fn(), downloadText: vi.fn() }));

const runs = vi.mocked(api.training.runs);

const FORM: PredictionForm = {
  available: true,
  reason: null,
  kind: 'tabular',
  task: 'classification',
  class_names: ['no', 'yes'],
  max_rows: 64,
  max_form_rows: 300,
  max_top_k: 10,
  budget_note: null,
  target: 'Survived',
  file: { max_mb: 200, max_rows: 1000000, keep_minutes: 60 },
  prepared: {
    columns: [
      { name: 'Age', mean: 30, std: 12 },
      { name: 'Fare', mean: 32, std: 40 },
    ],
    samples: [],
  },
};

const RESULT: FilePrediction = {
  id: 'abc123',
  filename: 'test-predictions.csv',
  expires_at: '2026-11-19T12:00:00+00:00',
  keep_minutes: 60,
  size_bytes: 2048,
  rows: 418,
  predicted: 416,
  failed: 2,
  problems: ["Row 7, 'Age': 'n/a' is not a number.", "Row 90, 'Fare': no value."],
  warnings: [],
  warning_count: 0,
  columns: ['PassengerId', 'Survived'],
  preview: [
    ['892', 'no'],
    ['893', 'yes'],
  ],
  task: 'classification',
  output: 'label',
  input_format: 'prepared',
  class_counts: { no: 266, yes: 150 },
  target: null,
  metrics: null,
  metrics_note: null,
  ignored_columns: ['PassengerId', 'Name'],
  decision_threshold: 0.5,
};

const TEST_CSV = 'PassengerId,Name,Age,Fare\n892,"Kelly, Mr. James",34.5,7.8\n893,Else,47,7\n';

function csvFile(text = TEST_CSV, name = 'test.csv') {
  return new File([text], name, { type: 'text/csv' });
}

async function show(form: PredictionForm = FORM) {
  runs.predictForm.mockResolvedValue(form);
  render(<PredictionPanel runId="r1" runName="Titanic" />);
  await waitFor(() => expect(screen.queryByLabelText('Loading')).not.toBeInTheDocument());
}

/** Choose a file with the button "Predict a file" (its hidden file input). */
async function choose(user: ReturnType<typeof userEvent.setup>, file: File) {
  const input = document.querySelector<HTMLInputElement>('input[type="file"]');
  expect(input).not.toBeNull();
  await user.upload(input as HTMLInputElement, file);
}

beforeEach(() => {
  vi.clearAllMocks();
  runs.discardPredictedFile.mockResolvedValue(undefined);
});

afterEach(async () => {
  await setLanguage('en', { save: false });
});

describe('FilePredict', () => {
  it('shows one button and one sentence until a file is chosen', async () => {
    await show();
    expect(screen.getByRole('button', { name: 'Predict a file' })).toBeInTheDocument();
    expect(
      screen.getByText('Upload a CSV file and get it back with a prediction for every row.')
    ).toBeInTheDocument();
    expect(screen.queryByText('Columns of the result')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Name of the prediction column')).not.toBeInTheDocument();
    expect(screen.queryByText(/deleted after the prediction/)).not.toBeInTheDocument();
    expect(await axeViolations()).toEqual([]);
  });

  it('uploads the file with the options of a submission and shows the result', async () => {
    const user = userEvent.setup();
    await show();
    const file = csvFile();
    await choose(user, file);
    // The options appear now, with the columns of the file; the file itself is not shown.
    expect(await screen.findByText('Columns of the result')).toBeInTheDocument();
    expect(screen.getByText('test.csv')).toBeInTheDocument();
    expect(screen.queryByDisplayValue(/Kelly/)).not.toBeInTheDocument();
    expect(screen.getByText(/The file is deleted after the prediction/)).toHaveTextContent(
      'at the latest after 60 minutes'
    );
    // The file has no target column: the prediction gets its name.
    const name = screen.getByLabelText('Name of the prediction column');
    await waitFor(() => expect(name).toHaveValue('Survived'));
    // Only chosen columns: the first one that is no feature (the id) is suggested.
    await user.click(screen.getByRole('radio', { name: 'Only chosen' }));
    expect(screen.getByLabelText('Columns to keep', { selector: 'input' })).toBeInTheDocument();
    expect(screen.getAllByText('PassengerId').length).toBeGreaterThan(0);
    expect(await axeViolations()).toEqual([]);

    let finish: (value: FilePrediction) => void = () => undefined;
    runs.predictFile.mockImplementation(
      (_id, _file, _options, upload) =>
        new Promise<FilePrediction>((resolve) => {
          upload?.onUploadProgress?.(512, 1024);
          finish = resolve;
        })
    );
    await user.click(screen.getByRole('button', { name: 'Predict the file' }));
    expect(await screen.findByText(/Uploading: /)).toBeInTheDocument();
    expect(screen.getByRole('progressbar', { name: 'Progress' })).toBeInTheDocument();
    await act(async () => finish(RESULT));

    expect(runs.predictFile).toHaveBeenCalledWith(
      'r1',
      file,
      { columns: ['PassengerId'], prediction_column: 'Survived', output: 'label' },
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    );
    expect(await screen.findByText('416 of 418 rows predicted')).toBeInTheDocument();
    expect(screen.getByText('Predicted: no 266 · yes 150')).toBeInTheDocument();
    expect(screen.getByText('2 rows without a prediction')).toBeInTheDocument();
    expect(screen.getByText("Row 7, 'Age': 'n/a' is not a number.")).toBeInTheDocument();
    const preview = screen.getByRole('table', { name: 'First rows of the result' });
    expect(
      within(preview)
        .getAllByRole('columnheader')
        .map((h) => h.textContent)
    ).toEqual(['PassengerId', 'Survived']);
    expect(within(preview).getByText('893')).toBeInTheDocument();
    expect(screen.getByText(/The first 2 of 418 rows/)).toBeInTheDocument();
    expect(await axeViolations()).toEqual([]);

    // The download fetches the result once (the server deletes it) and saves it again
    // from memory.
    const blob = new Blob(['PassengerId,Survived\n892,no\n']);
    runs.downloadPredictedFile.mockResolvedValue({ blob, filename: 'test-predictions.csv' });
    await user.click(screen.getByRole('button', { name: 'Download result' }));
    await waitFor(() => expect(downloadBlob).toHaveBeenCalledWith('test-predictions.csv', blob));
    await user.click(screen.getByRole('button', { name: 'Download result' }));
    await waitFor(() => expect(downloadBlob).toHaveBeenCalledTimes(2));
    expect(runs.downloadPredictedFile).toHaveBeenCalledTimes(1);
    expect(runs.downloadPredictedFile).toHaveBeenCalledWith('r1', 'abc123');
    expect(runs.discardPredictedFile).not.toHaveBeenCalled();
  });

  it('writes probabilities and shows the metrics of a file with the target', async () => {
    const user = userEvent.setup();
    await show();
    await choose(user, csvFile('PassengerId,Age,Fare,Survived\n1,22,7.25,no\n', 'train.csv'));
    // The file has the target column: the prediction keeps the neutral name.
    const name = await screen.findByLabelText('Name of the prediction column');
    await waitFor(() => expect(name).toHaveValue('prediction'));
    await user.clear(name);
    await user.type(name, 'p_yes');
    await user.click(screen.getByLabelText('What is written', { selector: 'input' }));
    await user.click(await screen.findByRole('option', { name: 'Probability of one class' }));
    // The positive class of a binary classification is preselected.
    expect(screen.getByLabelText('Class', { selector: 'input' })).toHaveValue('yes');

    runs.predictFile.mockResolvedValue({
      ...RESULT,
      id: 'first',
      failed: 0,
      predicted: 418,
      problems: [],
      output: 'probability',
      class: 'yes',
      values: { min: 0.021, max: 0.987, mean: 0.384 },
      target: 'Survived',
      metrics: { rows: 418, skipped: 0, accuracy: 0.8134, log_loss: 0.4321, roc_auc: 0.8765 },
      warnings: ["Row 2, 'Sex': 'x' does not occur in the training data."],
      warning_count: 31,
    });
    await user.click(screen.getByRole('button', { name: 'Predict the file' }));
    expect(await screen.findByText('418 rows predicted')).toBeInTheDocument();
    expect(runs.predictFile).toHaveBeenCalledWith(
      'r1',
      expect.any(File),
      { columns: 'all', prediction_column: 'p_yes', output: 'probability', class: 'yes' },
      expect.anything()
    );
    expect(
      screen.getByText('Probability of “yes” from 2.1 % to 98.7 %, mean 38.4 %')
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'Measured on “Survived” in the file (418 rows): accuracy 81.3 %, log loss 0.4321, AUC 0.8765'
      )
    ).toBeInTheDocument();
    expect(screen.queryByText(/without a prediction/)).not.toBeInTheDocument();
    expect(screen.getByText('… and 30 more')).toBeInTheDocument();

    // All probabilities need no column name; predicting again discards the result that
    // was not downloaded.
    await user.click(screen.getByLabelText('What is written', { selector: 'input' }));
    await user.click(await screen.findByRole('option', { name: 'Probabilities of all classes' }));
    expect(screen.queryByLabelText('Name of the prediction column')).not.toBeInTheDocument();
    runs.predictFile.mockResolvedValue({ ...RESULT, id: 'second', columns: ['no', 'yes'] });
    await user.click(screen.getByRole('button', { name: 'Predict the file' }));
    await waitFor(() => expect(runs.discardPredictedFile).toHaveBeenCalledWith('r1', 'first'));
    expect(runs.predictFile).toHaveBeenLastCalledWith(
      'r1',
      expect.any(File),
      { columns: 'all', prediction_column: 'p_yes', output: 'probabilities' },
      expect.anything()
    );
  });

  it('shows why a file cannot be predicted', async () => {
    const user = userEvent.setup();
    await show({ ...FORM, task: 'regression', class_names: null });
    await choose(user, csvFile());
    await screen.findByText('Columns of the result');
    // A regression has no choice of what is written.
    expect(
      screen.queryByLabelText('What is written', { selector: 'input' })
    ).not.toBeInTheDocument();
    const detail = "The file has no column 'Fare'. The model needs: Age, Fare.";
    runs.predictFile.mockRejectedValue(new ApiError(400, { detail }, detail));
    await user.click(screen.getByRole('button', { name: 'Predict the file' }));
    expect(await screen.findByText('Cannot predict the file')).toBeInTheDocument();
    expect(screen.getByText(detail)).toBeInTheDocument();
    expect(screen.queryByRole('table', { name: 'First rows of the result' })).toBeNull();
    expect(await axeViolations()).toEqual([]);

    // A used-up budget is named as such.
    runs.predictFile.mockRejectedValue(
      new ApiError(403, { detail: 'No compute left.', code: 'budget_exceeded' }, 'No compute left.')
    );
    await user.click(screen.getByRole('button', { name: 'Predict the file' }));
    expect(await screen.findByText('No compute left')).toBeInTheDocument();

    // A file above the limit is not uploaded at all.
    const large = csvFile();
    Object.defineProperty(large, 'size', { value: 201 * 1024 * 1024 });
    await choose(user, large);
    expect(await screen.findByText(/a file to predict may have 200 MB/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Predict the file' })).not.toBeInTheDocument();
  });

  it('takes a dropped file and is German in German', async () => {
    await setLanguage('de', { save: false });
    await show();
    const button = screen.getByRole('button', { name: 'Datei vorhersagen' });
    expect(
      screen.getByText(
        'CSV-Datei hochladen und mit einer Vorhersage für jede Zeile zurückbekommen.'
      )
    ).toBeInTheDocument();
    fireEvent.drop(button, { dataTransfer: { files: [csvFile()] } });
    expect(await screen.findByText('Spalten des Ergebnisses')).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Alle Spalten' })).toBeChecked();
    expect(screen.getByLabelText('Name der Vorhersagespalte')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Andere Datei wählen' })).toBeInTheDocument();
    expect(screen.getByText(/spätestens nach 60 Minuten/)).toBeInTheDocument();
    runs.predictFile.mockResolvedValue({ ...RESULT, failed: 1, predicted: 417 });
    await userEvent.setup().click(screen.getByRole('button', { name: 'Vorhersage starten' }));
    expect(await screen.findByText('417 von 418 Zeilen vorhergesagt')).toBeInTheDocument();
    expect(screen.getByText('1 Zeile ohne Vorhersage')).toBeInTheDocument();
    expect(screen.getByText('Vorhergesagt: no 266 · yes 150')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Ergebnis herunterladen' })).toBeInTheDocument();
    expect(screen.getByRole('table', { name: 'Erste Zeilen des Ergebnisses' })).toBeInTheDocument();
    expect(await axeViolations()).toEqual([]);
  });

  it('caps the text field for pasted rows and points to the file', async () => {
    const user = userEvent.setup();
    await show();
    await user.click(screen.getByRole('button', { name: 'Paste CSV rows' }));
    const field = screen.getByLabelText('CSV rows');
    expect(
      screen.getByText(/up to 300 rows .* For more rows: “Predict a file”\./)
    ).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Upload a CSV file' })).not.toBeInTheDocument();
    const rows = Array.from({ length: 400 }, (_, i) => `${i},7.5`);
    fireEvent.change(field, { target: { value: ['Age,Fare', ...rows].join('\n') } });
    expect(field).toHaveValue('');
    expect(
      screen.getByText('The text field takes up to 300 rows. Use “Predict a file” for more.')
    ).toBeInTheDocument();
    fireEvent.change(field, { target: { value: ['Age,Fare', ...rows.slice(0, 300)].join('\n') } });
    expect((field as HTMLTextAreaElement).value.split('\n')).toHaveLength(301);
  });
});

describe('fileColumns', () => {
  it('reads the names of the columns from the start of a file', async () => {
    expect(await fileColumns(csvFile())).toEqual(['PassengerId', 'Name', 'Age', 'Fare']);
    expect(await fileColumns(csvFile('\ufeffsep=;\r\nid;"a;b"; c \r\n1;2;3\r\n'))).toEqual([
      'id',
      'a;b',
      'c',
    ]);
    // Windows-1252 text, and a file without a header line.
    const latin = new File(
      [new Uint8Array([0x47, 0x72, 0xf6, 0xdf, 0x65, 0x2c, 0x78, 0x0a])],
      'a.csv'
    );
    expect(await fileColumns(latin)).toEqual(['Größe', 'x']);
    expect(await fileColumns(csvFile('1.5,2,3\n4,5,6\n'))).toEqual([]);
    // Only the start of a large file is read.
    const large = csvFile(`id,value\n${'1,2\n'.repeat(40000)}`);
    const slice = vi.spyOn(large, 'slice');
    expect(await fileColumns(large)).toEqual(['id', 'value']);
    expect(slice).toHaveBeenCalledWith(0, 64 * 1024);
  });

  it('counts lines without reading a long text to its end', () => {
    expect(hasMoreLines('a\nb\nc', 3)).toBe(false);
    expect(hasMoreLines('a\nb\nc\n', 3)).toBe(false);
    expect(hasMoreLines('a\nb\nc\nd', 3)).toBe(true);
    expect(hasMoreLines('', 0)).toBe(false);
  });
});
