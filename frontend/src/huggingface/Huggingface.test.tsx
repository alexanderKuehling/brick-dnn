/** The Hugging Face Hub in the app: publishing a run and importing a file of a dataset. */
import { axeViolations, render, screen, userEvent, waitFor, within } from '@test-utils';
import { beforeEach, vi } from 'vitest';
import {
  api,
  ApiError,
  type DataSource,
  type HubCapabilities,
  type HubDatasetFiles,
  type HubPublishPreview,
  type HubPublishResult,
} from '@/api';
import { SOURCE } from '@/data/fixtures';
import { ImportDialog } from '@/data/ImportDialog';
import { i18n } from '@/i18n';
import { detail, DOC } from '@/training/fixtures';
import { RunView } from '@/training/RunView';
import { isHubDatasetLink } from './HubImportPanel';
import { PublishDialog } from './PublishDialog';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      generate: vi.fn(),
      training: { runs: { hubPreview: vi.fn(), publishToHub: vi.fn(), downloadWeights: vi.fn() } },
      data: {
        hubFiles: vi.fn(),
        sources: { upload: vi.fn(), importUrl: vi.fn(), importFromHub: vi.fn(), remove: vi.fn() },
      },
    },
  };
});

const TOKEN = `hf_${'A'.repeat(34)}`;
const HUB: HubCapabilities = {
  publish: true,
  max_upload_mb: 250,
  licenses: ['apache-2.0', 'mit', 'cc-by-4.0', 'other'],
  default_license: 'apache-2.0',
  url: 'https://huggingface.co',
};
const CARD =
  '---\nlibrary_name: pytorch\nlicense: apache-2.0\ntags:\n- "brickdnn"\n---\n\n# Small CNN\n';
const PREVIEW: HubPublishPreview = {
  repo: 'small-cnn',
  default_repo: 'small-cnn',
  files: [
    { path: 'README.md', size: 4100 },
    { path: 'model.safetensors', size: 1_500_000 },
    { path: 'small_cnn.py', size: 20_800 },
    { path: 'network.json', size: 677 },
    { path: 'preprocessing.json', size: 528 },
  ],
  size: 1_526_105,
  max_upload_mb: 250,
  too_large: false,
  model_card: CARD,
  can_publish: true,
};
const RESULT: HubPublishResult = {
  repo_id: 'ada-hf/small-cnn',
  url: 'https://huggingface.co/ada-hf/small-cnn',
  created: true,
  private: true,
  license: 'apache-2.0',
  commit_url: 'https://huggingface.co/ada-hf/small-cnn/commit/abc123',
  commit: 'abc123',
  files: PREVIEW.files,
  size: PREVIEW.size,
};
const LISTING: HubDatasetFiles = {
  repo_id: 'scikit-learn/iris',
  revision: 'main',
  converted: false,
  page: 'https://huggingface.co/datasets/scikit-learn/iris',
  private: false,
  gated: false,
  files: [
    {
      path: 'data/train.csv',
      size: 3900,
      format: 'csv',
      config: 'default',
      split: 'train',
      ref: 'hf://datasets/scikit-learn/iris/data/train.csv',
    },
    {
      path: 'data/test.parquet',
      size: 2100,
      format: 'parquet',
      config: 'default',
      split: 'test',
      ref: 'hf://datasets/scikit-learn/iris/data/test.parquet',
    },
  ],
  total: 2,
  truncated: false,
  configs: ['default'],
  splits: ['train', 'test'],
  config: null,
  split: null,
  max_upload_mb: 500,
};

function hubError(status: number, code: string, message: string): ApiError {
  return new ApiError(status, { detail: message, code }, message);
}

function renderPublish(onClose = vi.fn()) {
  render(
    <PublishDialog opened onClose={onClose} run={{ id: 'r1', name: 'Small CNN' }} hub={HUB} />
  );
  return onClose;
}

function renderImport(onImported = vi.fn()) {
  render(
    <ImportDialog
      opened
      onClose={vi.fn()}
      initialFile={null}
      maxUploadMb={500}
      urlImport
      hubImport
      currentPrepName={null}
      onImported={onImported}
    />
  );
  return onImported;
}

beforeEach(async () => {
  vi.clearAllMocks();
  await i18n.changeLanguage('en');
  vi.mocked(api.training.runs.hubPreview).mockResolvedValue(PREVIEW);
  vi.mocked(api.training.runs.publishToHub).mockResolvedValue(RESULT);
  vi.mocked(api.data.hubFiles).mockResolvedValue(LISTING);
});

describe('PublishDialog', () => {
  it('shows what would be uploaded and publishes with the token, name, visibility and licence', async () => {
    renderPublish();
    const dialog = await screen.findByRole('dialog', { name: 'Publish to Hugging Face' });
    // The files and the model card, before anything is sent.
    expect(await within(dialog).findByText('model.safetensors')).toBeInTheDocument();
    expect(within(dialog).getByText('small_cnn.py')).toBeInTheDocument();
    expect(api.training.runs.hubPreview).toHaveBeenCalledWith('r1', '', expect.anything());
    const name = within(dialog).getByRole('textbox', { name: 'Repository name' });
    await waitFor(() => expect(name).toHaveValue('small-cnn'));
    await userEvent.click(within(dialog).getByRole('button', { name: /Show the model card/ }));
    expect(within(dialog).getByLabelText('Model card (README.md)')).toHaveTextContent(
      'library_name: pytorch'
    );
    expect(within(dialog).getByText(/Hugging Face, Inc. is in the USA/)).toBeInTheDocument();

    // Private by default; nothing can be sent without a token or with a bad name.
    const submit = within(dialog).getByRole('button', { name: 'Publish' });
    expect(submit).toBeDisabled();
    expect(within(dialog).getByRole('radio', { name: /^Private/ })).toBeChecked();
    const token = within(dialog).getByLabelText(/Hugging Face access token/);
    expect(token).toHaveAttribute('type', 'password');
    expect(token).toHaveAttribute('autocomplete', 'off');
    await userEvent.type(token, ` ${TOKEN} `);
    expect(submit).toBeEnabled();
    await userEvent.clear(name);
    await userEvent.type(name, 'bad name!');
    expect(within(dialog).getByText(/Letters, digits/)).toBeInTheDocument();
    expect(submit).toBeDisabled();
    await userEvent.clear(name);
    await userEvent.type(name, 'team/iris-net');

    await userEvent.click(within(dialog).getByRole('radio', { name: /Public/ }));
    await userEvent.click(within(dialog).getByRole('combobox', { name: 'Licence' }));
    await userEvent.click(await screen.findByRole('option', { name: 'MIT' }));
    await userEvent.click(within(dialog).getByRole('checkbox', { name: /Update the repository/ }));
    await userEvent.click(submit);

    expect(api.training.runs.publishToHub).toHaveBeenCalledWith('r1', {
      hf_token: TOKEN,
      repo: 'team/iris-net',
      private: false,
      license: 'mit',
      update_existing: true,
    });
    const done = await screen.findByRole('dialog', { name: 'Published to Hugging Face' });
    expect(
      within(done).getByText(/“ada-hf\/small-cnn” is on Hugging Face with 5 files/)
    ).toBeInTheDocument();
    expect(within(done).getByRole('link', { name: 'Open on Hugging Face' })).toHaveAttribute(
      'href',
      RESULT.url
    );
    // The token is gone from the page.
    expect(document.body.innerHTML).not.toContain(TOKEN);
  });

  it('explains what the Hub refused and keeps the form for another try', async () => {
    vi.mocked(api.training.runs.publishToHub).mockRejectedValue(
      hubError(409, 'name_taken', 'A model repository “ada-hf/small-cnn” exists already.')
    );
    renderPublish();
    const dialog = await screen.findByRole('dialog', { name: 'Publish to Hugging Face' });
    await within(dialog).findByText('model.safetensors');
    await userEvent.type(within(dialog).getByLabelText(/Hugging Face access token/), TOKEN);
    await userEvent.click(within(dialog).getByRole('button', { name: 'Publish' }));
    const alert = await within(dialog).findByText('Publishing failed');
    expect(alert).toBeInTheDocument();
    expect(within(dialog).getByText(/“ada-hf\/small-cnn” exists already/)).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Publish' })).toBeEnabled();
  });

  it('does not offer to publish what is too large or not the user’s to publish', async () => {
    vi.mocked(api.training.runs.hubPreview).mockResolvedValue({ ...PREVIEW, too_large: true });
    renderPublish();
    const dialog = await screen.findByRole('dialog', { name: 'Publish to Hugging Face' });
    expect(await within(dialog).findByText(/larger than 250 MB/)).toBeInTheDocument();
    await userEvent.type(within(dialog).getByLabelText(/Hugging Face access token/), TOKEN);
    expect(within(dialog).getByRole('button', { name: 'Publish' })).toBeDisabled();
  });

  it('says why nothing can be shown when the preview fails', async () => {
    vi.mocked(api.training.runs.hubPreview).mockRejectedValue(
      new ApiError(400, null, 'Only finished runs can be published.')
    );
    renderPublish();
    expect(await screen.findByText('Only finished runs can be published.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Publish' })).toBeDisabled();
  });

  describe.each(['en', 'de'] as const)('in %s', (language) => {
    beforeEach(async () => {
      await i18n.changeLanguage(language);
    });

    it('finds nothing serious with axe', async () => {
      vi.mocked(api.training.runs.publishToHub).mockRejectedValue(
        hubError(400, 'token_invalid', 'Hugging Face did not accept the access token (HTTP 401).')
      );
      renderPublish();
      const title =
        language === 'de' ? 'Auf Hugging Face veröffentlichen' : 'Publish to Hugging Face';
      const dialog = await screen.findByRole('dialog', { name: title });
      await within(dialog).findByText('model.safetensors');
      const submit = language === 'de' ? 'Veröffentlichen' : 'Publish';
      if (language === 'de') {
        expect(within(dialog).getByLabelText(/Hugging-Face-Zugriffstoken/)).toBeInTheDocument();
        expect(within(dialog).getByRole('radio', { name: /^Privat/ })).toBeChecked();
        expect(within(dialog).getByRole('combobox', { name: 'Lizenz' })).toHaveValue('Apache 2.0');
        expect(within(dialog).getByText(/sitzt in den USA/)).toBeInTheDocument();
      }
      expect(await axeViolations()).toEqual([]);
      // An error of the Hub is shown in the language of the app.
      await userEvent.type(
        within(dialog).getByLabelText(/Hugging.Face.(access token|Zugriffstoken)/),
        TOKEN
      );
      await userEvent.click(within(dialog).getByRole('button', { name: submit }));
      expect(
        await within(dialog).findByText(
          language === 'de'
            ? /Hugging Face hat das Zugriffstoken nicht angenommen/
            : /did not accept the access token \(HTTP 401\)/
        )
      ).toBeInTheDocument();
      expect(await axeViolations()).toEqual([]);
    });
  });
});

describe('RunView', () => {
  const finished = detail({ status: 'completed', has_weights: true, message: 'Finished' });
  const props = {
    doc: DOC,
    onBack: vi.fn(),
    onNewTraining: vi.fn(),
    onUpdate: vi.fn(),
    onDeleted: vi.fn(),
    onLoadNetwork: vi.fn(),
  };

  it('offers to publish a finished run with weights where the server can', async () => {
    render(<RunView run={finished} hub={HUB} {...props} />);
    await userEvent.click(screen.getByRole('button', { name: 'Publish to Hugging Face' }));
    const dialog = await screen.findByRole('dialog', { name: 'Publish to Hugging Face' });
    expect(await within(dialog).findByText('model.safetensors')).toBeInTheDocument();
    expect(api.training.runs.hubPreview).toHaveBeenCalledWith(finished.id, '', expect.anything());
  });

  it('does not offer it without the server’s support, without weights or to viewers', () => {
    const name = 'Publish to Hugging Face';
    const { unmount } = render(<RunView run={finished} {...props} />);
    expect(screen.queryByRole('button', { name })).not.toBeInTheDocument();
    unmount();
    const off = render(<RunView run={finished} hub={{ ...HUB, publish: false }} {...props} />);
    expect(screen.queryByRole('button', { name })).not.toBeInTheDocument();
    off.unmount();
    const running = render(<RunView run={detail({ has_weights: true })} hub={HUB} {...props} />);
    expect(screen.queryByRole('button', { name })).not.toBeInTheDocument();
    running.unmount();
    render(
      <RunView
        run={{ ...finished, started_by: 'bob', project: 3 }}
        hub={HUB}
        scope={{ username: 'cy', project: { id: 3, role: 'viewer', shared: true } }}
        {...props}
      />
    );
    expect(screen.queryByRole('button', { name })).not.toBeInTheDocument();
  });
});

describe('Import from Hugging Face', () => {
  it('recognises dataset references', () => {
    expect(isHubDatasetLink('hf://datasets/scikit-learn/iris')).toBe(true);
    expect(isHubDatasetLink(' https://huggingface.co/datasets/stanfordnlp/imdb ')).toBe(true);
    expect(isHubDatasetLink('https://huggingface.co/datasets/a/b/viewer/default/train')).toBe(true);
    // The link of a file is imported as it is.
    expect(isHubDatasetLink('https://huggingface.co/datasets/a/b/resolve/main/x.csv')).toBe(false);
    expect(isHubDatasetLink('https://example.org/datasets/a/b')).toBe(false);
  });

  it('lists the files of a dataset and imports the chosen one', async () => {
    const source: DataSource = { ...SOURCE, id: 9, name: 'iris · train' };
    vi.mocked(api.data.sources.importFromHub).mockResolvedValue(source);
    const onImported = renderImport();
    const dialog = await screen.findByRole('dialog', { name: 'Import data' });
    await userEvent.click(within(dialog).getByRole('tab', { name: 'Hugging Face' }));
    const field = within(dialog).getByRole('textbox', { name: 'Dataset on Hugging Face' });
    await userEvent.type(field, 'hf://datasets/scikit-learn/iris{Enter}');
    expect(api.data.hubFiles).toHaveBeenCalledWith(
      'hf://datasets/scikit-learn/iris',
      null,
      expect.anything()
    );
    expect(api.data.sources.importFromHub).not.toHaveBeenCalled(); // Enter lists, not imports

    const train = await within(dialog).findByRole('radio', { name: /data\/train\.csv/ });
    expect(within(dialog).getByText('default · train · 3.8 KB')).toBeInTheDocument();
    expect(within(dialog).getByRole('link', { name: /Open on Hugging Face/ })).toHaveAttribute(
      'href',
      LISTING.page
    );
    const submit = within(dialog).getByRole('button', { name: 'Import' });
    expect(submit).toBeDisabled();
    // The split narrows the list.
    await userEvent.click(within(dialog).getByRole('combobox', { name: 'Split' }));
    await userEvent.click(await screen.findByRole('option', { name: 'test' }));
    expect(
      within(dialog).queryByRole('radio', { name: /data\/train\.csv/ })
    ).not.toBeInTheDocument();
    await userEvent.click(within(dialog).getByRole('combobox', { name: 'Split' }));
    await userEvent.click(await screen.findByRole('option', { name: 'All' }));

    await userEvent.click(
      train.isConnected ? train : within(dialog).getByRole('radio', { name: /data\/train\.csv/ })
    );
    expect(submit).toBeEnabled();
    await userEvent.click(submit);
    await waitFor(() =>
      expect(api.data.sources.importFromHub).toHaveBeenCalledWith(
        'hf://datasets/scikit-learn/iris/data/train.csv',
        {},
        null,
        expect.anything()
      )
    );
    await waitFor(() => expect(onImported).toHaveBeenCalledWith(source, true, expect.anything()));
  });

  it('sends the access token of a gated dataset with the listing and the import only', async () => {
    vi.mocked(api.data.hubFiles).mockResolvedValue({ ...LISTING, gated: 'manual' });
    vi.mocked(api.data.sources.importFromHub).mockResolvedValue({ ...SOURCE, id: 9 });
    renderImport();
    const dialog = await screen.findByRole('dialog', { name: 'Import data' });
    await userEvent.click(within(dialog).getByRole('tab', { name: 'Hugging Face' }));
    await userEvent.type(
      within(dialog).getByRole('textbox', { name: 'Dataset on Hugging Face' }),
      'scikit-learn/iris'
    );
    const token = within(dialog).getByLabelText(/Access token/);
    expect(token).toHaveAttribute('type', 'password');
    await userEvent.type(token, TOKEN);
    await userEvent.click(within(dialog).getByRole('button', { name: 'Show files' }));
    expect(api.data.hubFiles).toHaveBeenCalledWith('scikit-learn/iris', TOKEN, expect.anything());
    expect(await within(dialog).findByText(/This dataset is gated/)).toBeInTheDocument();
    await userEvent.click(within(dialog).getByRole('radio', { name: /data\/test\.parquet/ }));
    await userEvent.click(within(dialog).getByRole('button', { name: 'Import' }));
    await waitFor(() =>
      expect(api.data.sources.importFromHub).toHaveBeenCalledWith(
        'hf://datasets/scikit-learn/iris/data/test.parquet',
        {},
        TOKEN,
        expect.anything()
      )
    );
  });

  it('leads a dataset link pasted as a link to the list of its files', async () => {
    renderImport();
    const dialog = await screen.findByRole('dialog', { name: 'Import data' });
    await userEvent.click(within(dialog).getByRole('tab', { name: 'From a link' }));
    await userEvent.type(
      within(dialog).getByRole('textbox', { name: 'Link to the file' }),
      'https://huggingface.co/datasets/scikit-learn/iris'
    );
    await userEvent.click(
      within(dialog).getByRole('button', { name: /A dataset on Hugging Face/ })
    );
    expect(api.data.hubFiles).toHaveBeenCalledWith(
      'https://huggingface.co/datasets/scikit-learn/iris',
      null,
      expect.anything()
    );
    expect(
      await within(dialog).findByRole('radio', { name: /data\/train\.csv/ })
    ).toBeInTheDocument();
  });

  it('says what the Hub did not find', async () => {
    vi.mocked(api.data.hubFiles).mockRejectedValue(
      hubError(
        404,
        'not_found',
        'The dataset “a/b” was not found on Hugging Face, or it is private.'
      )
    );
    renderImport();
    const dialog = await screen.findByRole('dialog', { name: 'Import data' });
    await userEvent.click(within(dialog).getByRole('tab', { name: 'Hugging Face' }));
    await userEvent.type(
      within(dialog).getByRole('textbox', { name: 'Dataset on Hugging Face' }),
      'a/b'
    );
    await userEvent.click(within(dialog).getByRole('button', { name: 'Show files' }));
    expect(await within(dialog).findByText('Cannot list the files')).toBeInTheDocument();
    expect(within(dialog).getByText(/was not found on Hugging Face/)).toBeInTheDocument();
  });

  describe.each(['en', 'de'] as const)('in %s', (language) => {
    beforeEach(async () => {
      await i18n.changeLanguage(language);
    });

    it('finds nothing serious with axe', async () => {
      renderImport();
      const dialog = await screen.findByRole('dialog');
      await userEvent.click(within(dialog).getByRole('tab', { name: 'Hugging Face' }));
      const field = within(dialog).getByRole('textbox', {
        name: language === 'de' ? 'Datensatz auf Hugging Face' : 'Dataset on Hugging Face',
      });
      expect(await axeViolations()).toEqual([]);
      await userEvent.type(field, 'scikit-learn/iris');
      await userEvent.click(
        within(dialog).getByRole('button', {
          name: language === 'de' ? 'Dateien zeigen' : 'Show files',
        })
      );
      await within(dialog).findByRole('radio', { name: /data\/train\.csv/ });
      if (language === 'de') {
        expect(within(dialog).getByRole('combobox', { name: 'Aufteilung' })).toBeInTheDocument();
        expect(
          within(dialog).getByText(/Der Server fragt Hugging Face \(USA\)/)
        ).toBeInTheDocument();
        expect(
          within(dialog).getByRole('link', { name: /Auf Hugging Face öffnen/ })
        ).toBeInTheDocument();
      }
      expect(await axeViolations()).toEqual([]);
    });
  });
});
