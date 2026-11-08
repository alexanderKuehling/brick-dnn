import { render, screen, userEvent, waitFor, within } from '@test-utils';
import { beforeEach, vi } from 'vitest';
import { ModalsProvider } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type ApiToken, type NewApiToken } from '@/api';
import { ApiTokensPage } from './ApiTokensPage';
import { claudeCodeCommand, curlCommand, mcpJsonConfig, mcpUrl } from './setup';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: { apiTokens: { list: vi.fn(), create: vi.fn(), revoke: vi.fn() } },
  };
});

const LAPTOP: ApiToken = {
  id: 2,
  name: 'Claude on my laptop',
  prefix: 'bdnn_Ab3dEf9h',
  read_only: false,
  created_at: '2026-11-01T10:00:00Z',
  last_used_at: '2026-11-07T18:30:00Z',
  expires_at: '2099-01-30T10:00:00Z',
};

const SCRIPT: ApiToken = {
  id: 1,
  name: 'Nightly report',
  prefix: 'bdnn_Zy8xWv7u',
  read_only: true,
  created_at: '2026-01-01T10:00:00Z',
  last_used_at: null,
  expires_at: '2026-02-01T10:00:00Z',
};

const SECRET = `bdnn_${'s'.repeat(43)}`;

const CREATED: NewApiToken = {
  id: 3,
  name: 'Cursor',
  prefix: 'bdnn_ssssssss',
  read_only: true,
  created_at: '2026-11-08T09:00:00Z',
  last_used_at: null,
  expires_at: '2026-12-08T09:00:00Z',
  token: SECRET,
};

function renderPage() {
  return render(
    <ModalsProvider>
      <ApiTokensPage />
    </ModalsProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(notifications, 'show').mockImplementation(() => 'id');
  vi.mocked(api.apiTokens.list).mockResolvedValue([LAPTOP, SCRIPT]);
});

describe('setup snippets', () => {
  it('point the assistant at the MCP endpoint with the token', () => {
    const url = mcpUrl('https://brick.example.com');
    expect(url).toBe('https://brick.example.com/api/mcp/');
    expect(claudeCodeCommand(url, 'T')).toBe(
      'claude mcp add --transport http brick-dnn https://brick.example.com/api/mcp/ ' +
        '--header "Authorization: Bearer T"'
    );
    expect(JSON.parse(mcpJsonConfig(url, 'T'))).toEqual({
      mcpServers: {
        'brick-dnn': {
          type: 'http',
          url: 'https://brick.example.com/api/mcp/',
          headers: { Authorization: 'Bearer T' },
        },
      },
    });
    expect(curlCommand('https://brick.example.com', 'T')).toBe(
      'curl -H "Authorization: Bearer T" https://brick.example.com/api/projects/'
    );
  });
});

describe('ApiTokensPage', () => {
  it('lists the tokens', async () => {
    renderPage();
    const laptop = await screen.findByRole('listitem', { name: 'Claude on my laptop' });
    expect(within(laptop).getByText('bdnn_Ab3dEf9h…')).toBeInTheDocument();
    expect(within(laptop).getByText('Full access')).toBeInTheDocument();
    expect(within(laptop).queryByText('Expired')).not.toBeInTheDocument();

    const script = screen.getByRole('listitem', { name: 'Nightly report' });
    expect(within(script).getByText('Read only')).toBeInTheDocument();
    expect(within(script).getByText('Never')).toBeInTheDocument();
    // The badge and the label of the date.
    expect(within(script).getAllByText('Expired')).toHaveLength(2);
  });

  it('shows how to connect an assistant, with a placeholder for the token', async () => {
    renderPage();
    await screen.findByRole('listitem', { name: 'Claude on my laptop' });
    const url = `${window.location.origin}/api/mcp/`;
    expect(screen.getByText(url)).toBeInTheDocument();
    expect(screen.getByText(claudeCodeCommand(url, '<YOUR_TOKEN>'))).toBeInTheDocument();
    expect(
      screen.getByText(curlCommand(window.location.origin, '<YOUR_TOKEN>'))
    ).toBeInTheDocument();
  });

  it('offers to create the first token', async () => {
    vi.mocked(api.apiTokens.list).mockResolvedValue([]);
    renderPage();
    expect(await screen.findByText(/You have no API tokens yet/)).toBeInTheDocument();
    expect(screen.queryByRole('list', { name: 'Your tokens' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Create token' }));
    expect(await screen.findByRole('dialog', { name: 'Create an API token' })).toBeInTheDocument();
  });

  it('shows an error if the tokens cannot be loaded', async () => {
    vi.mocked(api.apiTokens.list).mockRejectedValue(new ApiError(500, null, 'Server error.'));
    renderPage();
    expect(await screen.findByText('Server error.')).toBeInTheDocument();
  });

  it('creates a token and shows it only once', async () => {
    vi.mocked(api.apiTokens.create).mockResolvedValue(CREATED);
    // With a stub of the clipboard.
    const user = userEvent.setup();
    renderPage();
    await screen.findByRole('listitem', { name: 'Claude on my laptop' });

    await user.click(screen.getByRole('button', { name: 'Create token' }));
    const dialog = await screen.findByRole('dialog', { name: 'Create an API token' });
    expect(within(dialog).getByRole('radio', { name: 'Full access' })).toBeChecked();
    expect(within(dialog).getByRole('combobox', { name: 'Expires after' })).toHaveValue('90 days');

    await user.type(within(dialog).getByRole('textbox', { name: 'Name' }), '  Cursor ');
    await user.click(within(dialog).getByRole('radio', { name: 'Read only' }));
    await user.click(within(dialog).getByRole('combobox', { name: 'Expires after' }));
    await user.click(await screen.findByRole('option', { name: '30 days' }));
    await user.click(within(dialog).getByRole('button', { name: 'Create token' }));

    expect(api.apiTokens.create).toHaveBeenCalledWith({
      name: 'Cursor',
      read_only: true,
      expires_in_days: 30,
    });
    const done = await screen.findByRole('dialog', { name: 'Token created' });
    expect(within(done).getByText(/It will not be shown again/)).toBeInTheDocument();
    expect(within(done).getByTestId('new-token')).toHaveTextContent(SECRET);
    // The snippets in the dialog use the new token.
    const url = `${window.location.origin}/api/mcp/`;
    expect(within(done).getByText(claudeCodeCommand(url, SECRET))).toBeInTheDocument();

    await user.click(within(done).getByRole('button', { name: 'Copy token' }));
    expect(await navigator.clipboard.readText()).toBe(SECRET);
    expect(within(done).getByRole('button', { name: 'Copied' })).toBeInTheDocument();

    // The list has the new token, but not its secret.
    expect(screen.getByRole('listitem', { name: 'Cursor' })).toBeInTheDocument();
    await user.click(within(done).getByRole('button', { name: 'Done' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(screen.queryByText(new RegExp(SECRET))).not.toBeInTheDocument();
    expect(screen.getAllByRole('listitem').map((item) => item.getAttribute('aria-label'))).toEqual([
      'Cursor',
      'Claude on my laptop',
      'Nightly report',
    ]);

    // The next token starts with an empty form.
    await user.click(screen.getByRole('button', { name: 'Create token' }));
    const again = await screen.findByRole('dialog', { name: 'Create an API token' });
    expect(within(again).getByRole('textbox', { name: 'Name' })).toHaveValue('');
    expect(within(again).getByRole('radio', { name: 'Full access' })).toBeChecked();
  });

  it('sends "never" expiring tokens without an expiry', async () => {
    vi.mocked(api.apiTokens.create).mockResolvedValue({ ...CREATED, expires_at: null });
    renderPage();
    await screen.findByRole('listitem', { name: 'Claude on my laptop' });
    await userEvent.click(screen.getByRole('button', { name: 'Create token' }));
    const dialog = await screen.findByRole('dialog', { name: 'Create an API token' });
    await userEvent.type(within(dialog).getByRole('textbox', { name: 'Name' }), 'Forever');
    await userEvent.click(within(dialog).getByRole('combobox', { name: 'Expires after' }));
    await userEvent.click(await screen.findByRole('option', { name: 'Never' }));
    expect(within(dialog).getByText('The token works until you revoke it.')).toBeInTheDocument();
    await userEvent.click(within(dialog).getByRole('button', { name: 'Create token' }));
    expect(api.apiTokens.create).toHaveBeenCalledWith({
      name: 'Forever',
      read_only: false,
      expires_in_days: null,
    });
  });

  it('shows server errors on the fields or as a notification', async () => {
    vi.mocked(api.apiTokens.create)
      .mockRejectedValueOnce(
        new ApiError(400, { name: ['Ensure this field has no more than 100 characters.'] }, 'x')
      )
      .mockRejectedValueOnce(
        new ApiError(
          400,
          { detail: 'You can have at most 20 API tokens.' },
          'You can have at most 20 API tokens.'
        )
      );
    renderPage();
    await screen.findByRole('listitem', { name: 'Claude on my laptop' });
    await userEvent.click(screen.getByRole('button', { name: 'Create token' }));
    const dialog = await screen.findByRole('dialog', { name: 'Create an API token' });

    // Without a name, nothing is sent.
    await userEvent.type(within(dialog).getByRole('textbox', { name: 'Name' }), '   ');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Create token' }));
    expect(within(dialog).getByText('Give the token a name.')).toBeInTheDocument();
    expect(api.apiTokens.create).not.toHaveBeenCalled();

    await userEvent.type(within(dialog).getByRole('textbox', { name: 'Name' }), 'Too long');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Create token' }));
    expect(
      await within(dialog).findByText('Ensure this field has no more than 100 characters.')
    ).toBeInTheDocument();
    expect(notifications.show).not.toHaveBeenCalled();

    await userEvent.click(within(dialog).getByRole('button', { name: 'Create token' }));
    await waitFor(() =>
      expect(notifications.show).toHaveBeenCalledWith(
        expect.objectContaining({ color: 'red', message: 'You can have at most 20 API tokens.' })
      )
    );
    expect(
      within(dialog).queryByText('Ensure this field has no more than 100 characters.')
    ).not.toBeInTheDocument();
    expect(screen.getByRole('dialog', { name: 'Create an API token' })).toBeInTheDocument();
  });

  it('revokes a token after confirmation', async () => {
    vi.mocked(api.apiTokens.revoke).mockResolvedValue(undefined);
    renderPage();
    const laptop = await screen.findByRole('listitem', { name: 'Claude on my laptop' });

    await userEvent.click(
      within(laptop).getByRole('button', { name: 'Revoke “Claude on my laptop”' })
    );
    let confirm = await screen.findByRole('dialog', { name: 'Revoke “Claude on my laptop”?' });
    expect(within(confirm).getByText(/stop working immediately/)).toBeInTheDocument();
    await userEvent.click(within(confirm).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(api.apiTokens.revoke).not.toHaveBeenCalled();

    await userEvent.click(
      within(laptop).getByRole('button', { name: 'Revoke “Claude on my laptop”' })
    );
    confirm = await screen.findByRole('dialog', { name: 'Revoke “Claude on my laptop”?' });
    await userEvent.click(within(confirm).getByRole('button', { name: 'Revoke' }));
    expect(api.apiTokens.revoke).toHaveBeenCalledWith(2);
    await waitFor(() =>
      expect(
        screen.queryByRole('listitem', { name: 'Claude on my laptop' })
      ).not.toBeInTheDocument()
    );
    expect(screen.getByRole('listitem', { name: 'Nightly report' })).toBeInTheDocument();
  });

  it('keeps the token if revoking fails', async () => {
    vi.mocked(api.apiTokens.revoke).mockRejectedValue(new ApiError(0, null, 'Cannot reach it.'));
    renderPage();
    const laptop = await screen.findByRole('listitem', { name: 'Claude on my laptop' });
    await userEvent.click(within(laptop).getByRole('button', { name: /Revoke/ }));
    const confirm = await screen.findByRole('dialog', { name: 'Revoke “Claude on my laptop”?' });
    await userEvent.click(within(confirm).getByRole('button', { name: 'Revoke' }));
    await waitFor(() =>
      expect(notifications.show).toHaveBeenCalledWith(
        expect.objectContaining({ color: 'red', message: 'Cannot reach it.' })
      )
    );
    expect(screen.getByRole('listitem', { name: 'Claude on my laptop' })).toBeInTheDocument();
  });
});
