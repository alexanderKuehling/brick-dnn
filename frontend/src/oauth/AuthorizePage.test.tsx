import { render, screen, userEvent, waitFor } from '@test-utils';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, vi } from 'vitest';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type OAuthRequest } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { AuthorizePage } from './AuthorizePage';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return { ...actual, api: { oauth: { validate: vi.fn(), decide: vi.fn() } } };
});

vi.mock('@/auth/AuthContext', () => ({ useAuth: vi.fn() }));

// Passed on exactly as received, encoding included.
const QUERY =
  'response_type=code&client_id=c1&redirect_uri=https%3A%2F%2Fclaude.ai%2Fapi%2Fmcp%2Fauth_callback' +
  '&code_challenge=abc&code_challenge_method=S256&state=s%2B1&scope=read+write';

const REQUEST: OAuthRequest = {
  client: { name: 'Claude', uri: 'https://www.anthropic.com/claude' },
  redirect_host: 'claude.ai',
  write_requested: true,
};

const CALLBACK = 'https://claude.ai/api/mcp/auth_callback';

const logout = vi.fn();
const assign = vi.fn();

function LoginStub() {
  const location = useLocation();
  return <p data-testid="login">{(location.state as { from?: string } | null)?.from}</p>;
}

function renderPage(query = QUERY) {
  return render(
    <MemoryRouter initialEntries={[`/oauth/authorize?${query}`]}>
      <Routes>
        <Route path="/oauth/authorize" element={<AuthorizePage />} />
        <Route path="/login" element={<LoginStub />} />
      </Routes>
    </MemoryRouter>
  );
}

const allowButton = () => screen.getByRole('button', { name: 'Allow' });
const denyButton = () => screen.getByRole('button', { name: 'Deny' });

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(notifications, 'show').mockImplementation(() => 'id');
  vi.mocked(useAuth).mockReturnValue({
    user: { id: 1, username: 'alex', email: '', date_joined: '2026-01-01T00:00:00Z' },
    ready: true,
    login: vi.fn(),
    register: vi.fn(),
    logout,
  });
  vi.mocked(api.oauth.validate).mockResolvedValue(REQUEST);
  vi.mocked(api.oauth.decide).mockResolvedValue({
    redirect_to: `${CALLBACK}?code=xyz&state=s%2B1`,
  });
  // jsdom cannot navigate (and its location cannot be spied on).
  vi.stubGlobal('location', { ...window.location, assign });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('AuthorizePage', () => {
  it('shows who asks for what', async () => {
    renderPage();
    expect(screen.getByLabelText('Checking the sign-in request')).toBeInTheDocument();
    expect(
      await screen.findByRole('heading', { name: 'Allow Claude to use your Brick DNN account?' })
    ).toBeInTheDocument();
    expect(api.oauth.validate).toHaveBeenCalledWith(QUERY);
    // The host of the app's homepage, and where the browser returns to.
    expect(screen.getByText('www.anthropic.com')).toBeInTheDocument();
    expect(screen.getByText(/After you decide, you will return to/)).toHaveTextContent(
      'After you decide, you will return to claude.ai.'
    );
    expect(screen.getByText(/Signed in as/)).toHaveTextContent('Signed in as alex');
    expect(screen.getByText(/Look at your networks, layer types and training runs/)).toBeVisible();
    expect(screen.getByText(/disconnect it at any time on the API tokens page/)).toBeVisible();
    expect(screen.getByRole('switch', { name: /Also allow changes/ })).toBeChecked();
  });

  it('allows full access and returns to the app with the code', async () => {
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Allow' }));
    expect(api.oauth.decide).toHaveBeenCalledWith(QUERY, true, true);
    expect(assign).toHaveBeenCalledWith(`${CALLBACK}?code=xyz&state=s%2B1`);
    expect(await screen.findByText('Access allowed')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Allow' })).not.toBeInTheDocument();
  });

  it('allows read-only access when changes are switched off', async () => {
    renderPage();
    const write = await screen.findByRole('switch', { name: /Also allow changes/ });
    await userEvent.click(write);
    expect(write).not.toBeChecked();
    expect(screen.getByText('Read only: it changes nothing.')).toBeInTheDocument();
    await userEvent.click(allowButton());
    expect(api.oauth.decide).toHaveBeenCalledWith(QUERY, true, false);
    expect(assign).toHaveBeenCalledTimes(1);
  });

  it('offers no switch if the app only asks to read', async () => {
    vi.mocked(api.oauth.validate).mockResolvedValue({ ...REQUEST, write_requested: false });
    renderPage();
    expect(await screen.findByText(/asks for read-only access/)).toBeInTheDocument();
    expect(screen.queryByRole('switch')).not.toBeInTheDocument();
    await userEvent.click(allowButton());
    expect(api.oauth.decide).toHaveBeenCalledWith(QUERY, true, false);
  });

  it('denies access', async () => {
    const denied = `${CALLBACK}?error=access_denied&state=s%2B1`;
    vi.mocked(api.oauth.decide).mockResolvedValue({ redirect_to: denied });
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Deny' }));
    expect(api.oauth.decide).toHaveBeenCalledWith(QUERY, false, false);
    expect(assign).toHaveBeenCalledWith(denied);
    expect(await screen.findByText('Access denied')).toBeInTheDocument();
  });

  it('disables both buttons while sending the answer', async () => {
    let answer: (value: { redirect_to: string }) => void = () => undefined;
    vi.mocked(api.oauth.decide).mockReturnValue(
      new Promise((resolve) => {
        answer = resolve;
      })
    );
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Allow' }));
    expect(allowButton()).toBeDisabled();
    expect(denyButton()).toBeDisabled();
    expect(screen.getByRole('switch')).toBeDisabled();
    expect(assign).not.toHaveBeenCalled();
    answer({ redirect_to: `${CALLBACK}?code=xyz` });
    await waitFor(() => expect(assign).toHaveBeenCalledWith(`${CALLBACK}?code=xyz`));
  });

  it('keeps the page if the answer cannot be sent', async () => {
    vi.mocked(api.oauth.decide).mockRejectedValue(new ApiError(0, null, 'Cannot reach it.'));
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Allow' }));
    await waitFor(() =>
      expect(notifications.show).toHaveBeenCalledWith(
        expect.objectContaining({ color: 'red', message: 'Cannot reach it.' })
      )
    );
    expect(assign).not.toHaveBeenCalled();
    expect(allowButton()).toBeEnabled();
    expect(denyButton()).toBeEnabled();
  });

  it('shows an invalid link and never redirects', async () => {
    vi.mocked(api.oauth.validate).mockRejectedValue(
      new ApiError(
        400,
        { error: 'invalid_request', error_description: 'Unknown redirect_uri.' },
        'x'
      )
    );
    renderPage();
    expect(
      await screen.findByRole('heading', { name: 'This sign-in link is not valid' })
    ).toBeInTheDocument();
    expect(screen.getByText('Unknown redirect_uri.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Allow' })).not.toBeInTheDocument();
    expect(assign).not.toHaveBeenCalled();
  });

  it('sends other errors back to the app', async () => {
    const back = `${CALLBACK}?error=invalid_scope&error_description=Unknown+scope&state=s%2B1`;
    vi.mocked(api.oauth.validate).mockRejectedValue(
      new ApiError(
        400,
        { error: 'invalid_scope', error_description: 'Unknown scope.', redirect_to: back },
        'x'
      )
    );
    renderPage();
    await waitFor(() => expect(assign).toHaveBeenCalledWith(back));
    expect(screen.getByText(/Taking you back to/)).toHaveTextContent(
      'Taking you back to claude.ai…'
    );
    expect(screen.queryByRole('button', { name: 'Allow' })).not.toBeInTheDocument();
  });

  it('shows an invalid request found while deciding', async () => {
    vi.mocked(api.oauth.decide).mockRejectedValue(
      new ApiError(400, { error: 'invalid_request', error_description: 'Expired.' }, 'x')
    );
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Allow' }));
    expect(
      await screen.findByRole('heading', { name: 'This sign-in link is not valid' })
    ).toBeInTheDocument();
    expect(screen.getByText('Expired.')).toBeInTheDocument();
    expect(assign).not.toHaveBeenCalled();
  });

  it('never follows a redirect that would run code on this site', async () => {
    // eslint-disable-next-line no-script-url
    vi.mocked(api.oauth.decide).mockResolvedValue({ redirect_to: 'javascript:alert(1)' });
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Allow' }));
    expect(
      await screen.findByRole('heading', { name: 'This sign-in link is not valid' })
    ).toBeInTheDocument();
    expect(assign).not.toHaveBeenCalled();
  });

  it('can check the request again after a server error', async () => {
    vi.mocked(api.oauth.validate).mockRejectedValueOnce(
      new ApiError(500, null, 'Request failed (HTTP 500).')
    );
    renderPage();
    expect(await screen.findByText('Request failed (HTTP 500).')).toBeInTheDocument();
    expect(assign).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(await screen.findByRole('button', { name: 'Allow' })).toBeInTheDocument();
    expect(api.oauth.validate).toHaveBeenCalledTimes(2);
  });

  it('logs out and comes back to this page after the next login', async () => {
    renderPage();
    await userEvent.click(await screen.findByRole('button', { name: 'Log out' }));
    expect(logout).toHaveBeenCalled();
    expect(await screen.findByTestId('login')).toHaveTextContent(`/oauth/authorize?${QUERY}`);
    expect(api.oauth.decide).not.toHaveBeenCalled();
  });
});
