/** The prediction panel of the run view: forms from the columns, samples, CSV rows, images. */
import { fireEvent, render, screen, userEvent, waitFor, within } from '@test-utils';
import { beforeEach, vi } from 'vitest';
import { api, ApiError, type PredictionForm, type PredictionResult } from '@/api';
import { downloadText } from '@/builder/download';
import { imagePixels } from './loadImage';
import { PredictionPanel } from './PredictionPanel';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: { training: { runs: { predictForm: vi.fn(), predict: vi.fn() } } },
  };
});
vi.mock('@/builder/download', () => ({ downloadBlob: vi.fn(), downloadText: vi.fn() }));
vi.mock('./loadImage', () => ({ imagePixels: vi.fn() }));

const predictForm = vi.mocked(api.training.runs.predictForm);
const predict = vi.mocked(api.training.runs.predict);

const BASE: PredictionForm = {
  available: true,
  reason: null,
  kind: 'tabular',
  task: 'regression',
  class_names: null,
  max_rows: 64,
  max_form_rows: 500,
  max_top_k: 10,
  budget_note: null,
};

const SALES: PredictionForm = {
  ...BASE,
  target: 'sales',
  prepared: {
    columns: [
      { name: 'promo', mean: 0.5, std: 0.5 },
      { name: 'temp', mean: 12, std: 4 },
    ],
    samples: [],
  },
  raw: {
    prep: { id: 3, name: 'Sales prep' },
    source: 'sales',
    target: 'sales',
    columns: [
      { name: 'date', kind: 'datetime', min: '2024-01-01', max: '2024-02-12' },
      { name: 'store', kind: 'categorical', categories: ['Berlin', 'Hamburg'] },
      { name: 'promo', kind: 'boolean' },
      { name: 'temp', kind: 'numeric', min: 5, max: 24.5, missing: true },
      { name: 'note', kind: 'text', used: false },
    ],
    samples: [
      {
        values: { date: '2024-01-09', store: 'Hamburg', promo: true, temp: 14.5, note: 'x' },
        target: 134,
      },
      { values: { date: '2024-01-15', store: 'Berlin', promo: false, temp: null }, target: 118 },
    ],
    skipped_steps: ['Remove duplicates'],
    joined: ['stores'],
  },
};

const IRIS: PredictionForm = {
  ...BASE,
  task: 'classification',
  class_names: ['setosa', 'versicolor'],
  target: 'species',
  prepared: {
    columns: [
      { name: 'sepal_length', mean: 5.5, std: 0.5 },
      { name: 'petal_length', mean: 1.3, std: 0.2 },
    ],
    samples: [{ values: { sepal_length: '5.1', petal_length: '1.4' }, target: 'setosa' }],
  },
};

const MNIST: PredictionForm = {
  ...BASE,
  kind: 'image',
  task: 'classification',
  class_names: ['0', '1'],
  image: { channels: 1, height: 28, width: 28 },
};

function answer(
  predictions: PredictionResult['predictions'],
  extra: Partial<PredictionResult> = {}
) {
  return {
    run: { id: 'r1', name: 'Net' },
    task: 'regression',
    predictions,
    target: 'sales',
    ...extra,
  } as PredictionResult;
}

beforeEach(() => {
  vi.clearAllMocks();
});

async function show(form: PredictionForm) {
  predictForm.mockResolvedValue(form);
  render(<PredictionPanel runId="r1" runName="Sales net" />);
  await waitFor(() => expect(screen.queryByLabelText('Loading')).not.toBeInTheDocument());
}

describe('PredictionPanel', () => {
  it('builds the form from the raw columns and predicts a sample', async () => {
    const user = userEvent.setup();
    await show(SALES);
    expect(predictForm).toHaveBeenCalledWith('r1', expect.any(AbortSignal));
    // One input per used raw column, by kind; the unused column is only mentioned.
    expect(screen.getByLabelText('date')).toHaveAttribute('type', 'date');
    expect(screen.getAllByLabelText('store').length).toBeGreaterThan(0);
    expect(screen.getByLabelText('temp')).toHaveAttribute('inputmode', 'decimal');
    expect(screen.queryByLabelText('note')).not.toBeInTheDocument();
    expect(screen.getByText(/Not needed: note/)).toBeInTheDocument();
    expect(screen.getByText(/Not applied to new rows .*: Remove duplicates/)).toBeInTheDocument();
    expect(screen.getByText(/stored copy of “stores”/)).toBeInTheDocument();
    expect(screen.getByText('5 … 24.5')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Use a sample' }));
    expect(screen.getByDisplayValue('Hamburg')).toBeInTheDocument();
    expect(screen.getByLabelText('temp')).toHaveValue('14.5');
    expect(screen.getByRole('button', { name: 'Sample 1 of 2' })).toBeInTheDocument();

    predict.mockResolvedValue(answer([{ value: 131.25 }]));
    await user.click(screen.getByRole('button', { name: 'Predict' }));
    const card = await screen.findByLabelText('Prediction');
    expect(within(card).getByText('Predicted sales')).toBeInTheDocument();
    expect(within(card).getByText('131.25')).toBeInTheDocument();
    expect(within(card).getByText('actual: 134')).toBeInTheDocument();
    expect(predict).toHaveBeenCalledWith(
      'r1',
      {
        inputs: [{ date: '2024-01-09', store: 'Hamburg', promo: true, temp: 14.5 }],
        input_format: 'raw',
        top_k: undefined,
      },
      undefined
    );
  });

  it('shows the errors of the server next to the inputs', async () => {
    const user = userEvent.setup();
    await show(SALES);
    await user.type(screen.getByLabelText('temp'), 'warm');
    predict.mockRejectedValue(
      new ApiError(
        400,
        {
          detail: "The row, 'temp': 'warm' is not a number.",
          errors: ["The row, 'temp': 'warm' is not a number."],
        },
        "The row, 'temp': 'warm' is not a number."
      )
    );
    await user.click(screen.getByRole('button', { name: 'Predict' }));
    expect(await screen.findByText("'warm' is not a number.")).toBeInTheDocument();
    expect(screen.getByLabelText('temp')).toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByText('Cannot predict')).toBeInTheDocument();
  });

  it('predicts pasted CSV rows in parts and downloads the results', async () => {
    const user = userEvent.setup();
    await show(SALES);
    await user.click(screen.getByRole('button', { name: 'Paste CSV rows' }));
    const lines = ['date;store;promo;temp;sales'];
    for (let i = 0; i < 70; i += 1) {
      lines.push(`2024-03-${String(1 + (i % 28)).padStart(2, '0')};Berlin;true;${i},5;${100 + i}`);
    }
    fireEvent.change(screen.getByLabelText('CSV rows'), { target: { value: lines.join('\n') } });
    predict.mockImplementation(async (_id, body) => {
      const rows = body.inputs as unknown[];
      return answer(
        rows.map((_, i) => ({ value: i })),
        { warnings: rows.length === 6 ? ["Row 2, 'store': unknown."] : [] }
      );
    });
    await user.click(screen.getByRole('button', { name: 'Predict rows' }));
    expect(await screen.findByText('70 predictions')).toBeInTheDocument();
    expect(predict).toHaveBeenCalledTimes(2);
    expect((predict.mock.calls[0][1].inputs as unknown[]).length).toBe(64);
    expect((predict.mock.calls[1][1].inputs as unknown[]).length).toBe(6);
    expect(predict.mock.calls[0][1].inputs).toContainEqual({
      date: '2024-03-01',
      store: 'Berlin',
      promo: true,
      temp: '0,5', // the server reads the decimal comma like the import
    });
    // The warning of the second request is numbered from the first row of the file.
    expect(screen.getByText("Row 66, 'store': unknown.")).toBeInTheDocument();
    const table = screen.getByRole('table', { name: 'Predictions' });
    expect(within(table).getAllByRole('row')).toHaveLength(71);
    expect(within(table).getByText('Actual')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Results (.csv)' }));
    const [name, text] = vi.mocked(downloadText).mock.calls[0];
    expect(name).toBe('sales_net-predictions.csv');
    expect(text.split('\n')[0]).toBe('date,store,promo,temp,predicted_sales');
    expect(text.split('\n')[1]).toBe('2024-03-01,Berlin,true,"0,5",0');
  });

  it('refuses CSV text without the needed columns', async () => {
    const user = userEvent.setup();
    await show(SALES);
    await user.click(screen.getByRole('button', { name: 'Paste CSV rows' }));
    fireEvent.change(screen.getByLabelText('CSV rows'), { target: { value: 'date,x\n1,2' } });
    await user.click(screen.getByRole('button', { name: 'Predict rows' }));
    expect(
      await screen.findByText('The header line lacks columns: store, promo, temp.')
    ).toBeInTheDocument();
    expect(predict).not.toHaveBeenCalled();
  });

  it('predicts classes from the prepared columns with the top classes', async () => {
    const user = userEvent.setup();
    await show(IRIS);
    expect(screen.queryByRole('radiogroup')).not.toBeInTheDocument(); // no raw columns
    await user.click(screen.getByRole('button', { name: 'Use a sample' }));
    predict.mockResolvedValue(
      answer(
        [
          {
            class: 'setosa',
            index: 0,
            probability: 0.924,
            top: [
              { class: 'setosa', index: 0, probability: 0.924 },
              { class: 'versicolor', index: 1, probability: 0.076 },
            ],
          },
        ],
        { task: 'classification' }
      )
    );
    await user.click(screen.getByRole('button', { name: 'Predict' }));
    const card = await screen.findByLabelText('Prediction');
    expect(within(card).getByText('Predicted species')).toBeInTheDocument();
    expect(within(card).getByText('actual: setosa')).toBeInTheDocument();
    expect(within(card).getAllByText('92.4 %')).toHaveLength(2);
    expect(within(card).getByText('7.6 %')).toBeInTheDocument();
    expect(predict.mock.calls[0][1]).toEqual({
      inputs: [{ sepal_length: 5.1, petal_length: 1.4 }],
      input_format: undefined,
      top_k: 3,
    });
  });

  it('switches to the prepared columns', async () => {
    const user = userEvent.setup();
    await show(SALES);
    await user.click(screen.getByText('Prepared columns'));
    expect(screen.getByLabelText('promo')).toBeInTheDocument();
    expect(screen.queryByLabelText('date')).not.toBeInTheDocument();
    predict.mockResolvedValue(answer([{ value: 1, std: 0.25 }]));
    await user.type(screen.getByLabelText('promo'), '1');
    await user.type(screen.getByLabelText('temp'), '20');
    await user.click(screen.getByRole('button', { name: 'Predict' }));
    expect(await screen.findByText('± 0.25')).toBeInTheDocument();
    expect(predict.mock.calls[0][1].input_format).toBe('prepared');
  });

  it('predicts an uploaded image', async () => {
    const user = userEvent.setup();
    await show({ ...MNIST, image: { channels: 1, height: 2, width: 2 } });
    vi.mocked(imagePixels).mockResolvedValue(
      new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255])
    );
    predict.mockResolvedValue(
      answer([{ class: '1', probability: 0.8, top: [{ class: '1', probability: 0.8 }] }], {
        task: 'classification',
      })
    );
    const input = document.querySelector('input[type=file]') as HTMLInputElement;
    await user.upload(input, new File(['png'], 'digit.png', { type: 'image/png' }));
    const card = await screen.findByLabelText('Prediction');
    expect(within(card).getAllByText('80.0 %').length).toBeGreaterThan(0);
    expect(imagePixels).toHaveBeenCalledWith(expect.any(File), 2, 2);
    expect(predict.mock.calls[0][1]).toEqual({
      inputs: [
        [
          [
            [255, 0],
            [0, 255],
          ],
        ],
      ],
      input_scale: 'raw',
      top_k: 3,
    });
    expect(screen.getByText('digit.png – choose another')).toBeInTheDocument();
  });

  it('predicts a drawn digit', async () => {
    const user = userEvent.setup();
    await show(MNIST);
    await user.click(screen.getByText('Draw'));
    await user.click(screen.getByRole('button', { name: 'Predict' }));
    expect(await screen.findByText('Draw something first.')).toBeInTheDocument();

    const pad = screen.getByLabelText('Drawing pad');
    pad.getBoundingClientRect = () =>
      ({ left: 0, top: 0, width: 280, height: 280, right: 280, bottom: 280 }) as DOMRect;
    fireEvent.pointerDown(pad, { clientX: 140, clientY: 40, pointerId: 1 });
    fireEvent.pointerMove(pad, { clientX: 140, clientY: 160, pointerId: 1 });
    fireEvent.pointerMove(pad, { clientX: 140, clientY: 240, pointerId: 1 });
    fireEvent.pointerUp(pad, { pointerId: 1 });
    predict.mockResolvedValue(
      answer([{ class: '1', probability: 0.99, top: [] }], { task: 'classification' })
    );
    await user.click(screen.getByRole('button', { name: 'Predict' }));
    await screen.findByLabelText('Prediction');
    const [image] = predict.mock.calls[0][1].inputs as number[][][][];
    expect(image).toHaveLength(1);
    expect(image[0]).toHaveLength(28);
    expect(image[0][14][14]).toBeGreaterThan(200); // a vertical line through the centre
    expect(image[0][14][4]).toBe(0);
    expect(predict.mock.calls[0][1].input_scale).toBe('raw');

    await user.click(screen.getByRole('button', { name: 'Clear' }));
    expect(screen.queryByLabelText('Prediction')).not.toBeInTheDocument();
  });

  it('says why a run cannot predict, and when the budget is used up', async () => {
    await show({ ...BASE, available: false, reason: 'This run has no trained weights.' });
    expect(screen.getByText('This run has no trained weights.')).toBeInTheDocument();
  });

  it('notes demo data and used-up budgets', async () => {
    await show({ ...BASE, kind: 'demo', budget_note: 'The monthly budget is used up.' });
    expect(screen.getByText(/random demo data/)).toBeInTheDocument();
    expect(screen.getByText('The monthly budget is used up.')).toBeInTheDocument();
    expect(screen.getByText('No compute left')).toBeInTheDocument();
  });

  it('shows a budget refusal of the server', async () => {
    const user = userEvent.setup();
    await show(IRIS);
    predict.mockRejectedValue(
      new ApiError(403, { detail: 'Used up.', code: 'budget_exceeded' }, 'Used up.')
    );
    await user.click(screen.getByRole('button', { name: 'Predict' }));
    expect(await screen.findByText('Used up.')).toBeInTheDocument();
    expect(screen.getByText('No compute left')).toBeInTheDocument();
  });
});
