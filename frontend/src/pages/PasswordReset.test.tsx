import { render, screen, userEvent, waitFor } from '@test-utils';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, vi } from 'vitest';
import { api, ApiError } from '@/api';
import { ForgotPasswordPage, RESET_UNAVAILABLE } from './ForgotPasswordPage';
import { ResetPasswordPage } from './ResetPasswordPage';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      legal: { info: vi.fn(), page: vi.fn() },
      passwordReset: {
        available: vi.fn(),
        request: vi.fn(),
        check: vi.fn(),
        confirm: vi.fn(),
      },
    },
  };
});

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/forgot-password" element={<ForgotPasswordPage />} />
        <Route path="/reset-password" element={<ResetPasswordPage />} />
        <Route path="/login" element={<p>Login page</p>} />
      </Routes>
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.legal.info).mockResolvedValue({
    pages: { imprint: false, privacy: false, terms: false, refund: false },
    terms_version: null,
    privacy_version: null,
    accept_terms_required: false,
    retention: { unused_uploads_days: 0, inactive_accounts_days: 0 },
  });
});

describe('forgot password', () => {
  test('sends a link and says the same for every account', async () => {
    vi.mocked(api.passwordReset.available).mockResolvedValue({ available: true });
    vi.mocked(api.passwordReset.request).mockResolvedValue({
      detail: 'If an account with this username or e-mail address exists, we sent a link.',
    });
    renderAt('/forgot-password');
    await userEvent.type(await screen.findByLabelText(/Username or e-mail/), ' ada@example.com ');
    await userEvent.click(screen.getByRole('button', { name: 'Send me a link' }));
    expect(await screen.findByText('Check your e-mail')).toBeInTheDocument();
    expect(screen.getByText(/we sent a link/)).toBeInTheDocument();
    expect(api.passwordReset.request).toHaveBeenCalledWith('ada@example.com');
  });

  test('says so when the server sends no e-mail', async () => {
    vi.mocked(api.passwordReset.available).mockResolvedValue({ available: false });
    renderAt('/forgot-password');
    expect(await screen.findByText(RESET_UNAVAILABLE)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Send me a link' })).not.toBeInTheDocument();
  });

  test('shows errors such as the rate limit', async () => {
    vi.mocked(api.passwordReset.available).mockResolvedValue({ available: true });
    vi.mocked(api.passwordReset.request).mockRejectedValue(
      new ApiError(429, { detail: 'Request was throttled.' }, 'Request was throttled.')
    );
    renderAt('/forgot-password');
    await userEvent.type(await screen.findByLabelText(/Username or e-mail/), 'ada');
    await userEvent.click(screen.getByRole('button', { name: 'Send me a link' }));
    expect(await screen.findByText('Request was throttled.')).toBeInTheDocument();
  });
});

describe('reset password', () => {
  test('sets a new password with the link', async () => {
    vi.mocked(api.passwordReset.check).mockResolvedValue({ valid: true, username: 'ada' });
    vi.mocked(api.passwordReset.confirm)
      .mockRejectedValueOnce(
        new ApiError(400, { new_password: ['This password is too common.'] }, 'x')
      )
      .mockResolvedValueOnce({ detail: 'Changed.', username: 'ada' });
    renderAt('/reset-password#uid=MQ&token=abc-123');
    expect(await screen.findByText('ada')).toBeInTheDocument();
    expect(api.passwordReset.check).toHaveBeenCalledWith('MQ', 'abc-123');

    await userEvent.type(screen.getByLabelText(/^New password/), 'password1');
    await userEvent.type(screen.getByLabelText(/Repeat/), 'password2');
    await userEvent.click(screen.getByRole('button', { name: 'Change password' }));
    expect(await screen.findByText('The two passwords are not the same.')).toBeInTheDocument();
    expect(api.passwordReset.confirm).not.toHaveBeenCalled();

    await userEvent.clear(screen.getByLabelText(/Repeat/));
    await userEvent.type(screen.getByLabelText(/Repeat/), 'password1');
    await userEvent.click(screen.getByRole('button', { name: 'Change password' }));
    expect(await screen.findByText('This password is too common.')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Change password' }));
    expect(await screen.findByText('Password changed')).toBeInTheDocument();
    expect(api.passwordReset.confirm).toHaveBeenLastCalledWith('MQ', 'abc-123', 'password1');
    await userEvent.click(screen.getByRole('link', { name: 'Log in' }));
    expect(await screen.findByText('Login page')).toBeInTheDocument();
  });

  test('explains an invalid or used link', async () => {
    vi.mocked(api.passwordReset.check).mockResolvedValue({ valid: false, username: null });
    renderAt('/reset-password?uid=MQ&token=old');
    expect(await screen.findByText('This link does not work')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Ask for a new link' })).toHaveAttribute(
      'href',
      '/forgot-password'
    );
  });

  test('a link without its parts is invalid', async () => {
    renderAt('/reset-password');
    expect(await screen.findByText('This link does not work')).toBeInTheDocument();
    expect(api.passwordReset.check).not.toHaveBeenCalled();
  });

  test('a link that expires while typing', async () => {
    vi.mocked(api.passwordReset.check).mockResolvedValue({ valid: true, username: 'ada' });
    vi.mocked(api.passwordReset.confirm).mockRejectedValue(
      new ApiError(400, { detail: 'This link is invalid or has expired.' }, 'x')
    );
    renderAt('/reset-password#uid=MQ&token=t');
    await userEvent.type(await screen.findByLabelText(/^New password/), 'password1');
    await userEvent.type(screen.getByLabelText(/Repeat/), 'password1');
    await userEvent.click(screen.getByRole('button', { name: 'Change password' }));
    await waitFor(() => expect(screen.getByText('This link does not work')).toBeInTheDocument());
  });
});
