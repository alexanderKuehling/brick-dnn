/**
 * Segmentation in the training panel: the data picker (Shapes, uploads of images with masks,
 * how to pack the ZIP), the settings (batch size, loss), the run view (IoU, Dice and pixel
 * accuracy, per class, the gallery of predictions) and the run list.
 */
import { useState } from 'react';
import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { beforeEach, vi } from 'vitest';
import { api, ApiError, type CompatibilityFix, type RunDetail } from '@/api';
import { NO_SCOPE } from './access';
import { DataSourcePicker } from './DataSourcePicker';
import {
  describeFixes,
  isImageSource,
  isSegmentationSource,
  onlyInputAndOutput,
  starterNetwork,
} from './fit';
import {
  CONFIG,
  dataset,
  detail,
  DOC,
  previews,
  SEGMENTATION_STATUS,
  segmentationDataset,
  segmentationDetail,
  STATUS,
  summary,
  trial,
  TUNING,
} from './fixtures';
import {
  classShares,
  datasetDetails,
  hasTestResult,
  headlineMetric,
  isClassification,
  isSegmentation,
  resultText,
  segmentationNotes,
  weightsSnippet,
} from './format';
import { HistoryView } from './HistoryView';
import { RunView } from './RunView';
import {
  DEFAULT_SETTINGS,
  effectiveSource,
  maxSegmentationBatch,
  toConfig,
  validateSettings,
  type SegmentationScope,
  type SettingsValues,
  type SourceChoice,
} from './settings';
import { SettingsForm } from './SettingsForm';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      generate: vi.fn(),
      training: { runs: { previews: vi.fn(), downloadWeights: vi.fn() } },
    },
  };
});

const loadPreviews = vi.mocked(api.training.runs.previews);

beforeEach(() => {
  loadPreviews.mockReset();
  loadPreviews.mockResolvedValue(previews());
});

function show(run: RunDetail) {
  return render(
    <RunView
      run={run}
      doc={DOC}
      onBack={vi.fn()}
      onNewTraining={vi.fn()}
      onUpdate={vi.fn()}
      onDeleted={vi.fn()}
      onLoadNetwork={vi.fn()}
    />
  );
}

/** The tile or card whose first label is `label`. */
const card = (label: string) => screen.getAllByText(label)[0].closest('.mantine-Paper-root');

function Picker({
  choice,
  datasets,
}: {
  choice: SourceChoice;
  datasets: ReturnType<typeof dataset>[];
}) {
  return (
    <DataSourcePicker
      status={SEGMENTATION_STATUS}
      choice={choice}
      onChange={vi.fn()}
      datasets={datasets}
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

describe('The data picker', () => {
  it('offers Shapes among the built-in datasets, without a download', () => {
    render(<Picker choice={{ tab: 'builtin', builtin: 'shapes', upload: null }} datasets={[]} />);
    const shapes = screen.getByRole('radio', { name: /Shapes/ });
    expect(shapes).toBeChecked();
    expect(shapes).toHaveTextContent(
      '400 training and 100 test images with masks · 4 classes (background, square, circle, ' +
        'triangle) · no download'
    );
    expect(screen.getByRole('radio', { name: /MNIST/ })).toHaveTextContent(
      '60,000 training and 10,000 test images · 10 classes · 12 MB download'
    );
  });

  it('lists a segmentation upload with its pairs, classes and what was left out', () => {
    const folders = segmentationDataset({ id: 10, name: 'roads' });
    folders.info = { ...folders.info, predefined_split: [80, 20, 20] };
    folders.info.segmentation = {
      ...folders.info.segmentation!,
      ignore_index: 255,
      unmatched_images: 0,
      mask_size_mismatches: 3,
    };
    render(
      <Picker
        choice={{ tab: 'upload', builtin: null, upload: 9 }}
        datasets={[segmentationDataset(), folders, dataset()]}
      />
    );
    const cells = screen.getByRole('radio', { name: 'cells' });
    expect(cells).toBeChecked();
    expect(cells).toHaveTextContent('Segmentation · 120 images with masks · 3 classes · 2.3 MB');
    expect(cells).toHaveTextContent('Pixels: background 81%, cell 15%, nucleus 4%');
    expect(cells).toHaveTextContent('2 images without a mask');
    const roads = screen.getByRole('radio', { name: 'roads' });
    expect(roads).toHaveTextContent('Split 80 / 20 / 20 (training / validation / test)');
    expect(roads).toHaveTextContent(
      '3 masks resized to their images · pixels with the value 255 are ignored'
    );
    // The split of the folders is not a preparation of the Data tab.
    expect(screen.queryByText('Prepared')).not.toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'iris' })).toHaveTextContent('CSV · 150 rows');
  });

  it('explains how to pack images and masks', async () => {
    render(<Picker choice={{ tab: 'upload', builtin: null, upload: null }} datasets={[]} />);
    expect(screen.getByText(/or for segmentation the folders/)).toBeInTheDocument();
    const toggle = screen.getByRole('button', { name: 'Segmentation: how to pack the ZIP' });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    await userEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText(/images\/001\.jpg/)).toHaveTextContent('masks/001.png');
    expect(screen.getByText(/grey values 0, 1, 2/)).toHaveTextContent(
      'Up to 32 classes; JPEG masks and smoothed edges do not work.'
    );
    expect(screen.getByText(/the folders are the split/)).toHaveTextContent('at most 512×512');
  });

  it('says nothing about segmentation on a server without it', () => {
    render(
      <DataSourcePicker
        status={STATUS}
        choice={{ tab: 'upload', builtin: null, upload: null }}
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
    expect(screen.queryByText(/segmentation/i)).not.toBeInTheDocument();
  });
});

describe('Segmentation runs', () => {
  it('show IoU, Dice and pixel accuracy in the tiles and the test card', async () => {
    show(segmentationDetail());
    expect(card('Validation IoU')).toHaveTextContent(/^Validation IoU78.2%mean over the classes$/);
    expect(card('Validation Dice')).toHaveTextContent(/^Validation Dice87.0%mean/);
    expect(card('Pixel accuracy')).toHaveTextContent(/^Pixel accuracy97.2%validation$/);
    expect(card('Best epoch')).toHaveTextContent(/^Best epoch3IoU 78.2%$/);
    expect(card('Test IoU')).toHaveTextContent(
      /^Test IoU77.8%Dice86.7%Pixel acc.96.9%loss 0.09 · 100 test samples/
    );
    expect(screen.queryByText('Validation accuracy')).not.toBeInTheDocument();
    expect(screen.queryByText('Validation RMSE')).not.toBeInTheDocument();
    await screen.findByRole('region', { name: 'Predictions' });
  });

  it('draw the validation IoU per epoch and list it', async () => {
    show(segmentationDetail());
    const curve = screen.getByRole('slider', { name: /^Validation IoU/ });
    expect(curve).toHaveAttribute('aria-valuenow', '3');
    expect(screen.getByText('Mean IoU')).toBeInTheDocument();
    expect(screen.getAllByText('Dice').length).toBeGreaterThan(0);
    expect(screen.queryByRole('slider', { name: /^Accuracy/ })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Values per epoch' }));
    const tables = await screen.findAllByRole('table');
    const rows = within(tables[0]).getAllByRole('row');
    expect(rows.map((row) => row.textContent)).toEqual([
      'EpochTrain lossVal. lossVal. IoUVal. DicePixel acc.Time',
      '10.520.4621.6%23.2%86.5%8 s',
      '20.250.1954.0%63.7%93.3%8 s',
      '30.140.0878.2%87.0%97.2%8 s',
    ]);
  });

  it('show the IoU of every class with its colour', async () => {
    show(segmentationDetail());
    const table = screen.getByRole('table', { name: 'Results per class' });
    const rows = within(table).getAllByRole('row');
    expect(rows.map((row) => row.textContent)).toEqual([
      'ClassVal. IoUTest IoU',
      'background98.4%98.1%',
      'square80.2%74.9%',
      'circle78.0%80.9%',
      'triangle56.3%57.3%',
      'Mean78.2%77.8%',
    ]);
    expect(screen.getByText('Weights of epoch 3')).toBeInTheDocument();
    await screen.findByRole('region', { name: 'Predictions' });
  });

  it('show a gallery of predictions with a legend and the IoU of every sample', async () => {
    show(segmentationDetail());
    const gallery = await screen.findByRole('region', { name: 'Predictions' });
    expect(loadPreviews).toHaveBeenCalledWith('r1', expect.any(AbortSignal));
    await within(gallery).findByText('2 validation images · weights of epoch 3');
    const legend = within(gallery).getByRole('list', { name: 'Classes' });
    expect(
      within(legend)
        .getAllByRole('listitem')
        .map((item) => item.textContent)
    ).toEqual(['background', 'square', 'circle', 'triangle']);
    const first = within(gallery).getByRole('figure', { name: 'Validation image 1' });
    const pictures = within(first).getAllByRole('img');
    expect(pictures.map((img) => img.getAttribute('alt'))).toEqual([
      'Image of validation image 1',
      'Mask of validation image 1',
      'Prediction of validation image 1',
      'Overlay of validation image 1',
    ]);
    expect(pictures[0]).toHaveAttribute('src', expect.stringMatching(/^data:image\/png;base64,/));
    expect(first).toHaveTextContent(
      'IoU 92.1% · pixel accuracy 97.4% · background, square, triangle'
    );
    const second = within(gallery).getByRole('figure', { name: 'Validation image 2' });
    expect(second).toHaveTextContent('IoU 45.1% · pixel accuracy 90.1% · background, circle');
  });

  it('say so when the predictions cannot be loaded, and show none without previews', async () => {
    loadPreviews.mockRejectedValue(new ApiError(404, null, 'This run has no preview images.'));
    const { unmount } = show(segmentationDetail());
    expect(await screen.findByText('Cannot load the predictions')).toBeInTheDocument();
    expect(screen.getByText('This run has no preview images.')).toBeInTheDocument();
    unmount();

    loadPreviews.mockClear();
    const base = segmentationDetail();
    const running: RunDetail = {
      ...base,
      status: 'running',
      has_weights: false,
      state: { ...base.state!, status: 'running', segmentation: null, test: null },
    };
    show(running);
    expect(card('Validation IoU')).toHaveTextContent('78.2%');
    expect(screen.queryByRole('region', { name: 'Predictions' })).not.toBeInTheDocument();
    expect(screen.queryByRole('table', { name: 'Results per class' })).not.toBeInTheDocument();
    await waitFor(() => expect(loadPreviews).not.toHaveBeenCalled());
  });

  it('show the IoU of every trial of a search', async () => {
    const base = segmentationDetail();
    const trials = [
      trial({
        val_loss: 0.2,
        val_accuracy: null,
        val_iou: 0.5123,
        val_dice: 0.6,
        val_pixel_accuracy: 0.93,
      }),
      trial({
        trial: 2,
        params: { learning_rate: 0.0011, batch_size: 32 },
        val_loss: 0.08,
        val_accuracy: null,
        val_iou: 0.7821,
        val_dice: 0.87,
        val_pixel_accuracy: 0.97,
      }),
    ];
    show({
      ...base,
      tuning: TUNING,
      trials_done: 2,
      best_trial: { ...trials[1], weights_epoch: 3 },
      config: { ...CONFIG, epochs: 3 },
      state: { ...base.state!, tuning: TUNING, trials_total: 2, trials, best_trial: 2 },
    });
    const rows = within(screen.getByRole('table', { name: 'Trials' })).getAllByRole('row');
    expect(rows[0]).toHaveTextContent('Val. lossVal. IoUStatus');
    expect(rows[1]).toHaveTextContent('0.251.2%Done');
    expect(rows[2]).toHaveTextContent('0.0878.2%Best');
    expect(card('Best trial')).toHaveTextContent(/^Best trial2IoU 78.2%$/);
    await screen.findByRole('region', { name: 'Predictions' });
  });

  it('are listed with their test IoU as the headline', () => {
    const finished = summary({ ...segmentationDetail(), id: 'unet' });
    const noTest = summary({
      ...segmentationDetail(),
      id: 'unet-2',
      name: 'U-Net on cells',
      metrics: { ...segmentationDetail().metrics!, test_iou: null, test_loss: null },
    });
    render(
      <HistoryView
        runs={[finished, noTest]}
        error={null}
        onRetry={vi.fn()}
        onOpen={vi.fn()}
        onDeleted={vi.fn()}
        onNewTraining={vi.fn()}
        scope={NO_SCOPE}
        filter="mine"
        onFilterChange={vi.fn()}
      />
    );
    expect(screen.getByText('77.8%')).toBeInTheDocument();
    expect(screen.getByText('test mean IoU')).toBeInTheDocument();
    expect(screen.getByText('78.2%')).toBeInTheDocument();
    expect(screen.getByText('val. mean IoU')).toBeInTheDocument();
  });
});

describe('Segmentation in the texts', () => {
  it('names the mean IoU as the result of a run', () => {
    const run = segmentationDetail();
    expect(headlineMetric(run.metrics)).toEqual(['77.8%', 'test mean IoU']);
    expect(resultText(run)).toBe('test mean IoU 77.8%');
    const validation = { ...run.metrics!, test_iou: null, test_loss: null };
    expect(headlineMetric(validation)).toEqual(['78.2%', 'validation mean IoU']);
    expect(hasTestResult(run.metrics)).toBe(true);
    expect(hasTestResult(validation)).toBe(false);
  });

  it('knows a segmentation from its task or its metrics', () => {
    const run = segmentationDetail();
    expect(isSegmentation(run, run.state!.history)).toBe(true);
    expect(isClassification(run, run.state!.history)).toBe(false);
    expect(isSegmentation({ ...run, task: null }, [])).toBe(true); // from the metrics
    expect(isSegmentation(detail(), detail().state!.history)).toBe(false);
  });

  it('describes a segmentation dataset', () => {
    const cells = segmentationDataset();
    expect(datasetDetails(cells)).toBe('120 images with masks · 3 classes · 2.3 MB');
    expect(classShares(cells)).toBe('background 81%, cell 15%, nucleus 4%');
    expect(classShares(cells, 2)).toBe('background 81%, cell 15% …');
    expect(segmentationNotes(cells)).toBe('2 images without a mask');
    const tiny = segmentationDataset();
    tiny.info.segmentation = {
      ...tiny.info.segmentation!,
      class_pixel_share: [0.997, 0.003, 0],
      unmatched_images: 0,
      unmatched_masks: 1,
    };
    expect(classShares(tiny)).toBe('background 100%, cell <1%, nucleus 0%');
    expect(segmentationNotes(tiny)).toBe('1 mask without an image');
    tiny.info.segmentation.unmatched_masks = 0;
    expect(segmentationNotes(tiny)).toBeNull();
    expect(classShares(dataset())).toBeNull();
    expect(segmentationNotes(dataset())).toBeNull();
  });

  it('shows how to turn the output into a mask', () => {
    const snippet = weightsSnippet(segmentationDetail(), 'u_net.py', 'u_net.pt');
    expect(snippet).toContain('image = image.resize((64, 64), Image.BILINEAR)');
    expect(snippet).toContain('classes = ["background", "square", "circle", "triangle"]');
    expect(snippet).toContain('mask = output.argmax(dim=1)[0].to(torch.uint8)');
    expect(snippet).toContain('Image.fromarray(mask.numpy()).save("mask.png")');
    expect(snippet).not.toContain('output.argmax(dim=1).item()');
    const binary = segmentationDetail();
    binary.plan = {
      ...binary.plan!,
      targets: [{ kind: 'mask', num_classes: 2, binary: true }],
    };
    expect(weightsSnippet(binary, 'u_net.py', 'u_net.pt')).toContain(
      'mask = (output[0, 0] > 0).to(torch.uint8)'
    );
  });
});

describe('Fitting a network to segmentation data', () => {
  const shapes = effectiveSource(
    SEGMENTATION_STATUS,
    { tab: 'builtin', builtin: 'shapes', upload: null },
    []
  );
  const cells = effectiveSource(SEGMENTATION_STATUS, { tab: 'upload', builtin: null, upload: 9 }, [
    segmentationDataset(),
  ]);

  it('knows which sources have a class for every pixel', () => {
    expect(isSegmentationSource(shapes)).toBe(true);
    expect(isSegmentationSource(cells)).toBe(true);
    expect(isImageSource(cells)).toBe(true);
    const mnist = effectiveSource(
      SEGMENTATION_STATUS,
      { tab: 'builtin', builtin: 'mnist', upload: null },
      []
    );
    expect(isSegmentationSource(mnist)).toBe(false);
    const iris = effectiveSource(SEGMENTATION_STATUS, { tab: 'upload', builtin: null, upload: 7 }, [
      dataset(),
    ]);
    expect(isSegmentationSource(iris)).toBe(false);
    expect(isSegmentationSource(null)).toBe(false);
  });

  it('applies fixes of the Input and the output channels on its own', () => {
    const fixes: CompatibilityFix[] = [
      {
        node: 'input',
        type: 'Input',
        params: { shape: [3, 64, 64] },
        previous: { shape: [3, 128, 128] },
        summary: 'Input shape 3×128×128 → 3×64×64',
      },
      {
        node: 'mask',
        type: 'Conv2d',
        params: { out_channels: 4 },
        previous: { out_channels: 1 },
        summary: 'Output channels of the last layer 1 → 4',
      },
    ];
    expect(onlyInputAndOutput(fixes)).toBe(true);
    expect(describeFixes(fixes, true)).toBe(
      'Input set to 3×64×64, last layer to 4 output channels'
    );
    expect(describeFixes([fixes[1]], true)).toBe('Last layer set to 4 output channels');
    const transposed = { ...fixes[1], type: 'ConvTranspose2d' };
    expect(onlyInputAndOutput([transposed])).toBe(true);
    // Other parameters of a convolution are never changed without asking.
    const kernel = { ...fixes[1], params: { kernel_size: [1, 1] } };
    expect(onlyInputAndOutput([kernel])).toBe(false);
  });

  it('starts an empty canvas with a small network that keeps the image size', () => {
    const spec = (type: string) => ({ type, params: [] }) as never;
    const specs = { Input: spec('Input'), Conv2d: spec('Conv2d'), Activation: spec('Activation') };
    const network = starterNetwork(shapes, specs)!;
    expect(network.nodes.map((node) => node.type)).toEqual([
      'Input',
      'Conv2d',
      'Activation',
      'Conv2d',
      'Activation',
      'Conv2d',
    ]);
    expect(network.nodes[0].params).toEqual({ shape: [3, 64, 64] });
    expect(network.nodes[5].params).toEqual({ out_channels: 4, kernel_size: [1, 1] });
    expect(network.edges).toHaveLength(5);
    const upload = starterNetwork(cells, specs)!;
    expect(upload.nodes[0].params).toEqual({ shape: [3, 64, 64] });
    expect(upload.nodes[5].params).toEqual({ out_channels: 3, kernel_size: [1, 1] });
    expect(starterNetwork(shapes, { Input: spec('Input') })).toBeNull();
  });
});

describe('Settings of a segmentation', () => {
  const scope: SegmentationScope = { maxBatch: 50 };

  it('have a batch size of their own and a loss', () => {
    const values: SettingsValues = { ...DEFAULT_SETTINGS, loss: 'cross_entropy_dice' };
    expect(toConfig(values)).toMatchObject({ batch_size: 64 });
    expect(toConfig(values)).not.toHaveProperty('loss');
    const config = toConfig(values, { segmentation: true, builtin: true });
    expect(config).toMatchObject({ batch_size: 8, loss: 'cross_entropy_dice', test_split: 0 });
  });

  it('refuse batches that do not fit into memory', () => {
    expect(maxSegmentationBatch({ fixed_bytes: 30_811_200, sample_bytes: 17_118_003 }, 900)).toBe(
      53
    );
    expect(maxSegmentationBatch(null, 900)).toBeNull();
    expect(maxSegmentationBatch({ fixed_bytes: 1, sample_bytes: 1 }, undefined)).toBeNull();
    expect(validateSettings(DEFAULT_SETTINGS, { segmentation: scope })).toEqual({});
    const large = { ...DEFAULT_SETTINGS, segmentation_batch_size: 64 };
    expect(validateSettings(large, { segmentation: scope })).toEqual({
      segmentation_batch_size: 'At most 50: larger batches do not fit into memory',
    });
    expect(validateSettings(large, { segmentation: { maxBatch: null } })).toEqual({});
    expect(validateSettings(large, { segmentation: { maxBatch: 0 } })).toEqual({
      segmentation_batch_size: 'No batch fits into memory: make the Input smaller',
    });
    // The batch size of other data is not checked against it.
    expect(validateSettings({ ...DEFAULT_SETTINGS, batch_size: 512 })).toEqual({});
    expect(
      validateSettings({ ...large, segmentation_batch_size: '' }, { segmentation: scope })
    ).toEqual({
      segmentation_batch_size: 'Required',
    });
    // A search over batch sizes: the largest one has to fit.
    const search = { ...DEFAULT_SETTINGS, search: true };
    expect(validateSettings(search, { segmentation: scope })).toEqual({
      batch_sizes: 'At most 50: larger batches do not fit into memory',
    });
    expect(validateSettings({ ...search, batch_sizes: [8, 16] }, { segmentation: scope })).toEqual(
      {}
    );
  });

  function Form({ segmentation }: { segmentation: SegmentationScope | null }) {
    const [values, setValues] = useState<SettingsValues>(DEFAULT_SETTINGS);
    return (
      <>
        <SettingsForm
          values={values}
          onChange={setValues}
          errors={validateSettings(values, { segmentation })}
          name=""
          namePlaceholder="U-Net"
          onNameChange={() => {}}
          devices={[]}
          defaultDevice={null}
          builtin
          segmentation={segmentation}
        />
        <output>{JSON.stringify(toConfig(values, { segmentation: segmentation !== null }))}</output>
      </>
    );
  }

  it('show the batch size for segmentation, what fits and the loss', async () => {
    render(<Form segmentation={scope} />);
    const batch = screen.getByLabelText('Batch size') as HTMLInputElement;
    expect(batch.value).toBe('8');
    expect(
      screen.getByText('Images per optimizer step; up to 50 fit into memory')
    ).toBeInTheDocument();
    await userEvent.clear(batch);
    await userEvent.type(batch, '64');
    expect(
      await screen.findByText('At most 50: larger batches do not fit into memory')
    ).toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent('"batch_size":64');
    const loss = screen.getByRole('combobox', { name: 'Loss' }) as HTMLInputElement;
    expect(loss.value).toBe('Cross-entropy');
    await userEvent.click(loss);
    await userEvent.click(await screen.findByRole('option', { name: 'Cross-entropy + Dice' }));
    expect(screen.getByRole('status')).toHaveTextContent('"loss":"cross_entropy_dice"');
  });

  it('keep the usual batch size and no loss for other data', () => {
    render(<Form segmentation={null} />);
    expect((screen.getByLabelText('Batch size') as HTMLInputElement).value).toBe('64');
    expect(screen.queryByRole('combobox', { name: 'Loss' })).not.toBeInTheDocument();
    expect(screen.getByRole('status')).not.toHaveTextContent('loss');
  });
});
