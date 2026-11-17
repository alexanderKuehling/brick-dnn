/**
 * The built-in text datasets in the training panel: the data picker (what they hold, their test
 * texts, also in German), fitting a network to them (the fixes, a first network) and trying a
 * trained run with typed texts (the words the network read).
 */
import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { beforeEach, vi } from 'vitest';
import { api, type CompatibilityFix, type PredictionForm, type TrainingStatus } from '@/api';
import { i18n } from '@/i18n';
import { DataSourcePicker } from './DataSourcePicker';
import {
  describeFixes,
  inputUnit,
  isImageSource,
  isSegmentationSource,
  isTextSource,
  onlyInputAndOutput,
  starterNetwork,
} from './fit';
import { TEXT_STATUS } from './fixtures';
import { PredictionPanel } from './predict/PredictionPanel';
import { effectiveSource, type SourceChoice } from './settings';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: { training: { runs: { predictForm: vi.fn(), predict: vi.fn() } } },
  };
});

const predictForm = vi.mocked(api.training.runs.predictForm);
const predict = vi.mocked(api.training.runs.predict);

beforeEach(() => {
  vi.clearAllMocks();
});

function Picker({
  choice,
  status = TEXT_STATUS,
}: {
  choice: SourceChoice;
  status?: TrainingStatus;
}) {
  return (
    <DataSourcePicker
      status={status}
      choice={choice}
      onChange={vi.fn()}
      datasets={[]}
      datasetsError={null}
      onReloadDatasets={vi.fn()}
      uploading={null}
      uploadProgress={null}
      uploadError={null}
      onDismissUploadError={vi.fn()}
      onUpload={vi.fn()}
      onCancelUpload={vi.fn()}
      onDelete={vi.fn()}
    />
  );
}

const LOOKUP: SourceChoice = { tab: 'builtin', builtin: 'lookup', upload: null };

describe('The data picker', () => {
  it('lists the text datasets with their size, classes, vocabulary and length', () => {
    render(<Picker choice={LOOKUP} />);
    const lookup = screen.getByRole('radio', { name: /Key lookup/ });
    expect(lookup).toBeChecked();
    expect(lookup).toHaveTextContent('Attention finds the asked pair');
    expect(lookup).toHaveTextContent(
      '12,000 training and 2,000 test texts · 10 classes (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) · ' +
        'vocabulary of 289, 10 tokens per text · no download'
    );
    expect(screen.getByRole('radio', { name: /Sentiment/ })).toHaveTextContent(
      '6,000 training and 1,000 test texts · 2 classes (negative, positive) · vocabulary of ' +
        '151, 24 tokens per text · no download'
    );
  });

  it('shows test texts with their class for the chosen text dataset only', () => {
    render(<Picker choice={LOOKUP} />);
    const examples = within(screen.getByRole('radio', { name: /Key lookup/ })).getByLabelText(
      'Test texts'
    );
    expect(examples).toHaveTextContent('g7 f4 i8 l9 t0 h6 u3 k0 ? h → 6');
    expect(examples).toHaveTextContent('m5 n4 f3 e1 b0 s2 c3 p6 ? f → 3');
    // Not for the others, and not for image datasets.
    const sentiment = screen.getByRole('radio', { name: /Sentiment/ });
    expect(within(sentiment).queryByLabelText('Test texts')).not.toBeInTheDocument();
    expect(sentiment).not.toHaveTextContent('the soup was not very tasty');
  });

  it('describes the text datasets in German (the descriptions come from the server)', async () => {
    await i18n.changeLanguage('de');
    const german = 'Text: kurze Bewertungen, positiv oder negativ.';
    const status = {
      ...TEXT_STATUS,
      builtin_datasets: TEXT_STATUS.builtin_datasets.map((d) =>
        d.id === 'sentiment' ? { ...d, description: `${german} Generiert.` } : d
      ),
    };
    render(
      <Picker choice={{ tab: 'builtin', builtin: 'sentiment', upload: null }} status={status} />
    );
    const sentiment = screen.getByRole('radio', { name: /Sentiment/ });
    expect(sentiment).toHaveTextContent(german);
    expect(sentiment).toHaveTextContent(
      '6.000 Trainings- und 1.000 Testtexte · 2 Klassen (negative, positive) · Vokabular von ' +
        '151, 24 Tokens pro Text · kein Download'
    );
    // The texts and class names stay as they are.
    expect(within(sentiment).getByLabelText('Testtexte')).toHaveTextContent(
      'the soup was not very tasty but the staff was friendly → positive'
    );
  });
});

describe('Fitting a network to a text dataset', () => {
  const lookup = effectiveSource(TEXT_STATUS, LOOKUP, []);
  const mnist = effectiveSource(
    TEXT_STATUS,
    { tab: 'builtin', builtin: 'mnist', upload: null },
    []
  );

  it('knows that the texts are token ids, not images', () => {
    expect(isTextSource(lookup)).toBe(true);
    expect(isImageSource(lookup)).toBe(false);
    expect(isSegmentationSource(lookup)).toBe(false);
    expect(inputUnit(lookup)).toBe('tokens');
    expect(isTextSource(mnist)).toBe(false);
    expect(inputUnit(mnist)).toBe('images');
  });

  it('applies the fixes of the Input, the Embedding and the outputs on its own', () => {
    const fixes: CompatibilityFix[] = [
      {
        node: 'tokens',
        type: 'Input',
        params: { shape: [10] },
        previous: { shape: [64] },
        summary: 'Input shape 64 → 10',
      },
      {
        node: 'embed',
        type: 'Embedding',
        params: { num_embeddings: 289 },
        previous: { num_embeddings: 10000 },
        summary: 'Vocabulary size of the Embedding 10000 → 289',
      },
      {
        node: 'fc',
        type: 'Linear',
        params: { out_features: 10 },
        previous: { out_features: 4 },
        summary: 'Outputs of the last layer 4 → 10',
      },
    ];
    expect(onlyInputAndOutput(fixes)).toBe(true);
    expect(describeFixes(fixes, 'tokens')).toBe(
      'Input set to 10 tokens, last layer to 10 outputs, Embedding set to 289 words'
    );
    expect(describeFixes([fixes[1]], 'tokens')).toBe('Embedding set to 289 words');
    // Other parameters of the Embedding are never changed without asking.
    expect(onlyInputAndOutput([{ ...fixes[1], params: { embedding_dim: 8 } }])).toBe(false);
  });

  it('starts an empty canvas with a small Transformer for the texts', () => {
    const spec = (type: string) => ({ type, params: [] }) as never;
    const types = [
      'Input',
      'Embedding',
      'PositionalEmbedding',
      'TransformerEncoder',
      'SequencePool',
      'Linear',
    ];
    const specs = Object.fromEntries(types.map((type) => [type, spec(type)]));
    const network = starterNetwork(lookup, specs)!;
    expect(network.nodes.map((node) => node.type)).toEqual(types);
    expect(network.nodes[0].params).toEqual({ shape: [10], dtype: 'long' });
    expect(network.nodes[1].params).toEqual({ num_embeddings: 289, embedding_dim: 64 });
    expect(network.nodes[2].params).toEqual({ max_len: 10 });
    expect(network.nodes[5].params).toEqual({ out_features: 10 });
    expect(network.edges).toHaveLength(5);
    expect(starterNetwork(lookup, { Input: spec('Input') })).toBeNull();
  });
});

const SENTIMENT_FORM: PredictionForm = {
  available: true,
  reason: null,
  kind: 'tokens',
  task: 'classification',
  class_names: ['negative', 'positive'],
  max_rows: 64,
  max_form_rows: 500,
  max_top_k: 10,
  budget_note: null,
  text: {
    length: 24,
    vocab_size: 151,
    vocabulary: ['<pad>', '<unk>', 'the'],
    dataset: 'sentiment',
    examples: [
      { text: 'the soup was not very tasty but the staff was friendly', class: 'positive' },
      { text: 'honestly my phone seemed awful', class: 'negative' },
    ],
  },
};

describe('Predicting texts', () => {
  async function show() {
    predictForm.mockResolvedValue(SENTIMENT_FORM);
    render(<PredictionPanel runId="r1" runName="Reviews" />);
    await waitFor(() => expect(screen.queryByLabelText('Loading')).not.toBeInTheDocument());
  }

  it('predicts the typed texts and shows the words the network read', async () => {
    const user = userEvent.setup();
    predict.mockResolvedValue({
      run: { id: 'r1', name: 'Reviews' },
      task: 'classification',
      predictions: [
        {
          class: 'negative',
          probability: 0.97,
          top: [
            { class: 'negative', probability: 0.97 },
            { class: 'positive', probability: 0.03 },
          ],
          tokens: ['the', 'coffee', 'was', 'not', '<unk>'],
          unknown_words: ['yummy'],
        },
        {
          class: 'negative',
          probability: 0.9,
          top: [{ class: 'negative', probability: 0.9 }],
          tokens: ['honestly', 'my', 'phone', 'seemed', 'awful'],
        },
      ],
    });
    await show();
    expect(
      screen.getByText(/in the vocabulary of the training texts \(151 entries\) become <unk>/)
    ).toBeVisible();
    const box = screen.getByRole('textbox', { name: 'Texts' });
    // The first test text is there to start with; a test text can be picked.
    expect(box).toHaveValue('the soup was not very tasty but the staff was friendly');
    await user.clear(box);
    await user.type(box, 'The coffee was not yummy');
    await user.click(screen.getByRole('button', { name: 'Text 2' }));
    expect(box).toHaveValue('honestly my phone seemed awful');
    await user.clear(box);
    await user.type(box, 'The coffee was not yummy{enter}honestly my phone seemed awful');
    await user.click(screen.getByRole('button', { name: 'Predict' }));
    await waitFor(() => expect(predict).toHaveBeenCalledTimes(1));
    expect(predict).toHaveBeenCalledWith(
      'r1',
      { inputs: ['The coffee was not yummy', 'honestly my phone seemed awful'], top_k: 3 },
      undefined
    );
    const [first, second] = await screen.findAllByLabelText('Tokens');
    expect(
      within(first)
        .getAllByText(/./)
        .map((badge) => badge.textContent)
    ).toEqual(['the', 'coffee', 'was', 'not', '<unk>']);
    expect(screen.getByText('Unknown words: yummy')).toBeVisible();
    expect(second).toHaveTextContent('honestly');
    const results = screen.getAllByLabelText('Prediction');
    expect(results[0]).toHaveTextContent('negative');
    // A test text says which class it really has.
    expect(results[1]).toHaveTextContent('actual: negative');
  });

  it('shows its hint in German', async () => {
    await i18n.changeLanguage('de');
    await show();
    expect(
      screen.getByText(/im Vokabular der Trainingstexte \(151 Einträge\) stehen/)
    ).toBeVisible();
    expect(screen.getByRole('textbox', { name: 'Texte' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Vorhersagen' })).toBeVisible();
  });
});
