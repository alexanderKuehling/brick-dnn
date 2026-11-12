import { render, screen, userEvent, within } from '@test-utils';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, vi } from 'vitest';
import { api, ApiError, setTokens, type LegalInfo } from '@/api';
import { AuthProvider } from '@/auth/AuthContext';
import { RegisterPage } from '@/pages/RegisterPage';
import { LegalLinks } from './LegalLinks';
import { LegalPage } from './LegalPage';
import { resetLegalInfo } from './useLegalInfo';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      legal: { info: vi.fn(), page: vi.fn() },
      register: vi.fn(),
      me: vi.fn(),
      logout: vi.fn(),
    },
  };
});

function info(pages: Partial<LegalInfo['pages']>, retention = 0): LegalInfo {
  const all = { imprint: false, privacy: false, terms: false, refund: false, ...pages };
  return {
    pages: all,
    terms_version: all.terms ? 'abc123' : null,
    privacy_version: all.privacy ? 'def456' : null,
    accept_terms_required: all.terms,
    retention: { unused_uploads_days: retention, inactive_accounts_days: 0 },
  };
}

function renderAt(path: string) {
  return render(
    <AuthProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/imprint" element={<LegalPage page="imprint" />} />
          <Route path="/privacy" element={<LegalPage page="privacy" />} />
          <Route path="/terms" element={<LegalPage page="terms" />} />
          <Route path="/refund" element={<LegalPage page="refund" />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/" element={<LegalLinks />} />
        </Routes>
      </MemoryRouter>
    </AuthProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  resetLegalInfo();
  vi.mocked(api.me).mockRejectedValue(new ApiError(401, null, 'Not logged in.'));
});

afterEach(() => {
  setTokens(null);
});

describe('LegalPage', () => {
  it('shows the Markdown of the page', async () => {
    vi.mocked(api.legal.info).mockResolvedValue(info({ imprint: true, privacy: true }));
    vi.mocked(api.legal.page).mockResolvedValue({
      page: 'imprint',
      title: 'Imprint',
      content: '# Impressum\n\nMax Mustermann\nMusterstraße 1\n\n<script>alert(1)</script>',
      version: 'abc',
      updated_at: '2026-11-01T10:00:00Z',
    });
    const { container } = renderAt('/imprint');
    expect(await screen.findByRole('heading', { name: 'Impressum' })).toBeInTheDocument();
    expect(api.legal.page).toHaveBeenCalledWith('imprint');
    expect(screen.getByText(/Musterstraße 1/)).toBeInTheDocument();
    expect(container.querySelector('script')).toBeNull();
    expect(screen.getByText(/Last changed/)).toBeInTheDocument();
    // Links to the pages the server has, not to the terms.
    const nav = await screen.findByRole('navigation', { name: 'Legal' });
    const links = within(nav).getAllByRole('link');
    // Contact leads to the imprint, which has the operator's address.
    expect(links.map((link) => link.getAttribute('href'))).toEqual([
      '/imprint',
      '/privacy',
      '/imprint',
    ]);
    expect(links.map((link) => link.textContent)).toEqual(['Imprint', 'Privacy', 'Contact']);
  });

  it('shows the refund policy and links it with the other pages', async () => {
    vi.mocked(api.legal.info).mockResolvedValue(info({ terms: true, refund: true }));
    vi.mocked(api.legal.page).mockResolvedValue({
      page: 'refund',
      title: 'Refund policy',
      content: '# Refund policy\n\nFull refund within 14 days.',
      version: 'abc',
      updated_at: '2026-11-12T10:00:00Z',
    });
    renderAt('/refund');
    expect(await screen.findByText('Full refund within 14 days.')).toBeInTheDocument();
    expect(api.legal.page).toHaveBeenCalledWith('refund');
    const nav = await screen.findByRole('navigation', { name: 'Legal' });
    const links = within(nav).getAllByRole('link');
    // Without an imprint there is no contact link.
    expect(links.map((link) => link.getAttribute('href'))).toEqual(['/terms', '/refund']);
    expect(links.map((link) => link.textContent)).toEqual(['Terms', 'Refunds']);
  });

  it('says when the page has not been set up', async () => {
    vi.mocked(api.legal.info).mockResolvedValue(info({}));
    vi.mocked(api.legal.page).mockRejectedValue(new ApiError(404, null, 'Not found.'));
    renderAt('/terms');
    expect(
      await screen.findByText('This page has not been set up on this server.')
    ).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Terms of use' })).toBeInTheDocument();
  });

  it('shows no links when the server has no legal pages', async () => {
    vi.mocked(api.legal.info).mockResolvedValue(info({}));
    renderAt('/');
    await vi.waitFor(() => expect(api.legal.info).toHaveBeenCalled());
    expect(screen.queryByRole('navigation', { name: 'Legal' })).toBeNull();
  });
});

describe('Sign-up with terms of use', () => {
  const fill = async () => {
    await userEvent.type(screen.getByLabelText(/Username/), 'alex');
    await userEvent.type(screen.getByLabelText(/Email/), 'alex@example.com');
    await userEvent.type(screen.getByLabelText(/^Password/), 'secret-password');
  };

  it('asks to accept the terms when the server has some', async () => {
    vi.mocked(api.legal.info).mockResolvedValue(info({ terms: true, privacy: true }));
    vi.mocked(api.register).mockResolvedValue({
      user: { id: 1, username: 'alex', email: 'alex@example.com', date_joined: '' },
      access: 'a',
      refresh: 'r',
    });
    renderAt('/register');
    const checkbox = await screen.findByRole('checkbox', { name: /I agree to the terms of use/ });
    expect(checkbox).toBeRequired();
    expect(screen.getByRole('link', { name: 'terms of use' })).toHaveAttribute('href', '/terms');
    expect(screen.getByRole('link', { name: 'privacy policy' })).toHaveAttribute(
      'href',
      '/privacy'
    );
    await fill();
    await userEvent.click(checkbox);
    await userEvent.click(screen.getByRole('button', { name: 'Create account' }));
    expect(api.register).toHaveBeenCalledWith('alex', 'alex@example.com', 'secret-password', true);
  });

  it('has no checkbox without terms', async () => {
    vi.mocked(api.legal.info).mockResolvedValue(info({ privacy: true }));
    renderAt('/register');
    expect(await screen.findByRole('link', { name: 'privacy policy' })).toBeInTheDocument();
    expect(screen.queryByRole('checkbox')).toBeNull();
  });
});
