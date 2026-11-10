/** The training panel together with the real useTraining hook (only the API is mocked). */
import { act, render, screen, userEvent, waitFor } from '@test-utils';
import { beforeEach, vi } from 'vitest';
import { ModalsProvider } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type Dataset } from '@/api';
import { ANALYSIS, detail, DOC, STATUS, summary } from './fixtures';
import TrainingDrawer from './TrainingDrawer';
import { useTraining } from './useTraining';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      generate: vi.fn(),
      training: {
        status: vi.fn(),
        datasets: { list: vi.fn(), upload: vi.fn(), remove: vi.fn() },
        check: vi.fn(),
        estimate: vi.fn(),
        runs: {
          list: vi.fn(),
          get: vi.fn(),
          create: vi.fn(),
          stop: vi.fn(),
          remove: vi.fn(),
          downloadWeights: vi.fn(),
        },
      },
    },
  };
});

const IRIS: Dataset = {
  id: 7,
  name: 'iris',
  kind: 'csv',
  size_bytes: 4551,
  created_at: '2026-10-28T10:00:00Z',
  info: {
    kind: 'csv',
    name: 'iris',
    num_samples: 150,
    builtin: null,
    sample_shape: null,
    train_size: null,
    test_size: null,
    num_features: 4,
    columns: ['a', 'b', 'c', 'd', 'species'],
    has_header: true,
    delimiter: ',',
    target_kind: 'text',
    target_range: null,
    class_names: ['setosa', 'versicolor', 'virginica'],
    class_counts: null,
    subdir: null,
    num_classes: 3,
  },
  owner: 'ada',
  mine: true,
  shared_in: [],
};

function Panel() {
  const training = useTraining(true);
  return (
    <ModalsProvider>
      <button type="button" onClick={training.open}>
        open panel
      </button>
      <button type="button" onClick={training.close}>
        close panel
      </button>
      <TrainingDrawer
        training={training}
        loggedIn
        doc={DOC}
        networkName="Tiny MLP"
        analysis={ANALYSIS}
        onLogin={vi.fn()}
        onShowProblems={vi.fn()}
        onLoadNetwork={vi.fn()}
      />
    </ModalsProvider>
  );
}

const fileInput = () => document.querySelector('input[type=file]') as HTMLInputElement;
const csv = () => new File(['a,b\n1,2\n'], 'data.csv', { type: 'text/csv' });

async function openUploads() {
  await userEvent.click(screen.getByRole('button', { name: 'open panel' }));
  await userEvent.click(await screen.findByText('Your data'));
  await screen.findByText(/not uploaded any data yet/);
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  vi.spyOn(notifications, 'show').mockImplementation(() => 'id');
  vi.mocked(api.training.status).mockResolvedValue(STATUS);
  vi.mocked(api.training.runs.list).mockResolvedValue([]);
  vi.mocked(api.training.datasets.list).mockResolvedValue([]);
  vi.mocked(api.training.check).mockResolvedValue({
    ok: true,
    errors: [],
    warnings: [],
    plan: null,
  });
});

describe('training panel', () => {
  it('keeps an upload going when another tab is shown', async () => {
    let reject!: (error: unknown) => void;
    vi.mocked(api.training.datasets.upload).mockReturnValue(
      new Promise((_, fail) => (reject = fail))
    );
    render(<Panel />);
    await openUploads();
    await userEvent.upload(fileInput(), csv());
    expect(await screen.findByText('Uploading “data.csv” …')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('tab', { name: 'Runs' }));
    await userEvent.click(screen.getByRole('tab', { name: 'New training' }));
    expect(screen.getByText('Uploading “data.csv” …')).toBeInTheDocument();
    // A second pick while the first upload runs does not start another one.
    await userEvent.upload(fileInput(), csv());
    expect(api.training.datasets.upload).toHaveBeenCalledTimes(1);

    await act(async () => reject(new ApiError(400, null, 'No class folders found.')));
    expect(await screen.findByText('No class folders found.')).toBeInTheDocument();
    expect(notifications.show).not.toHaveBeenCalled();
  });

  it('reports an upload that finishes while the panel is closed', async () => {
    let resolve!: (dataset: Dataset) => void;
    vi.mocked(api.training.datasets.upload).mockReturnValue(
      new Promise((done) => (resolve = done))
    );
    render(<Panel />);
    await openUploads();
    await userEvent.upload(fileInput(), csv());
    await userEvent.click(screen.getByRole('button', { name: 'close panel' }));

    await act(async () => resolve(IRIS));
    expect(notifications.show).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Upload finished', color: 'green' })
    );

    vi.mocked(api.training.datasets.list).mockResolvedValue([IRIS]);
    await userEvent.click(screen.getByRole('button', { name: 'open panel' }));
    expect(await screen.findByRole('radio', { name: /iris/ })).toBeChecked();
  });

  /** An upload that runs until `finish` is called or it is aborted, like the real one. */
  function slowUpload() {
    const upload: {
      finish?: (dataset: Dataset) => void;
      progress?: (sent: number, total: number) => void;
      signal?: AbortSignal;
    } = {};
    vi.mocked(api.training.datasets.upload).mockImplementation((_file, _name, transfer) => {
      upload.progress = transfer?.onUploadProgress;
      upload.signal = transfer?.signal;
      return new Promise((resolve, reject) => {
        upload.finish = resolve;
        transfer?.signal?.addEventListener('abort', () =>
          reject(new DOMException('The operation was aborted.', 'AbortError'))
        );
      });
    });
    return upload;
  }

  const MB = 1024 * 1024;

  it('shows the progress of an upload, then that the server checks the file', async () => {
    const upload = slowUpload();
    render(<Panel />);
    await openUploads();
    await userEvent.upload(fileInput(), csv());
    const bar = await screen.findByRole('progressbar', { name: 'Upload of “data.csv”' });

    act(() => upload.progress?.(60 * MB, 240 * MB));
    await waitFor(() => expect(bar).toHaveAttribute('aria-valuenow', '25'));
    expect(screen.getByText('60.0 MB of 240.0 MB')).toBeInTheDocument();

    act(() => upload.progress?.(240 * MB, 240 * MB));
    expect(screen.getByRole('status')).toHaveTextContent('Processing the file on the server …');
    expect(bar).not.toHaveAttribute('aria-valuenow');
    // Too late to cancel: the server has the file.
    expect(screen.queryByRole('button', { name: 'Cancel upload' })).not.toBeInTheDocument();

    await act(async () => upload.finish?.(IRIS));
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /iris/ })).toBeChecked();
  });

  it('cancels an upload', async () => {
    const upload = slowUpload();
    render(<Panel />);
    await openUploads();
    await userEvent.upload(fileInput(), csv());
    await userEvent.click(await screen.findByRole('button', { name: 'Cancel upload' }));

    expect(upload.signal?.aborted).toBe(true);
    await waitFor(() => expect(screen.queryByRole('progressbar')).not.toBeInTheDocument());
    expect(screen.queryByText('Upload failed')).not.toBeInTheDocument();
    expect(notifications.show).not.toHaveBeenCalled();
    // A new upload can start right away.
    await userEvent.upload(fileInput(), csv());
    expect(api.training.datasets.upload).toHaveBeenCalledTimes(2);
  });

  it('lets the same file be picked again after a failed upload', async () => {
    vi.mocked(api.training.datasets.upload).mockRejectedValue(
      new ApiError(400, null, 'The CSV file needs at least 4 rows.')
    );
    render(<Panel />);
    await openUploads();
    const file = csv();
    await userEvent.upload(fileInput(), file);
    expect(await screen.findByText(/at least 4 rows/)).toBeInTheDocument();
    // The input is emptied after each pick, so choosing the same file fires a change again.
    expect(fileInput().files).toHaveLength(0);
    await userEvent.upload(fileInput(), file);
    await waitFor(() => expect(api.training.datasets.upload).toHaveBeenCalledTimes(2));
  });

  it('keeps past and running trainings usable when PyTorch is missing', async () => {
    vi.mocked(api.training.status).mockResolvedValue({
      ...STATUS,
      available: false,
      reason: 'PyTorch is not installed on the server.',
    });
    vi.mocked(api.training.runs.list).mockResolvedValue([summary()]);
    vi.mocked(api.training.runs.get).mockResolvedValue(detail());
    render(<Panel />);
    await waitFor(() => expect(api.training.runs.list).toHaveBeenCalled());

    // A running training opens directly, with its Stop button.
    await userEvent.click(screen.getByRole('button', { name: 'open panel' }));
    expect(await screen.findByRole('button', { name: 'Stop' })).toBeInTheDocument();

    // Only new trainings are blocked, with the reason.
    await userEvent.click(screen.getByRole('tab', { name: 'New training' }));
    expect(screen.getByText('PyTorch is not installed on the server.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Start training' })).not.toBeInTheDocument();
  });
});
