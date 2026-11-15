/**
 * axe-core on the main pages and components (docs/accessibility.md): no serious or critical
 * WCAG 2.2 A/AA violations, in English and German. Colours and sizes need a browser:
 * scripts/axe-smoke.mjs checks those against a running app.
 */
import { axeViolations, render, screen, userEvent } from '@test-utils';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, vi } from 'vitest';
import { api, ApiError, setTokens, type LegalInfo } from '@/api';
import { AuthProvider } from '@/auth/AuthContext';
import { PricingPage } from '@/billing/PricingPage';
import { BILLING } from '@/billing/testData';
import { resetBillingInfo } from '@/billing/useBillingInfo';
import { AppHeader } from '@/components/AppHeader';
import { ChartFrame } from '@/components/charts/ChartFrame';
import { LineChart } from '@/components/LineChart';
import { SkipLink } from '@/components/SkipLink';
import { i18n } from '@/i18n';
import { LegalPage } from '@/legal/LegalPage';
import { resetLegalInfo } from '@/legal/useLegalInfo';
import { ForgotPasswordPage } from '@/pages/ForgotPasswordPage';
import { LoginPage } from '@/pages/LoginPage';
import { NotFoundPage } from '@/pages/NotFoundPage';
import { RegisterPage } from '@/pages/RegisterPage';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      me: vi.fn(),
      logout: vi.fn(),
      login: vi.fn(),
      register: vi.fn(),
      legal: { info: vi.fn(), page: vi.fn() },
      billing: { info: vi.fn(), state: vi.fn() },
      passwordReset: { available: vi.fn(), request: vi.fn() },
    },
  };
});

const LEGAL: LegalInfo = {
  pages: { imprint: true, privacy: true, terms: true, refund: true, accessibility: true },
  terms_version: 'abc',
  privacy_version: 'def',
  accept_terms_required: true,
  retention: { unused_uploads_days: 180, inactive_accounts_days: 730 },
};

const STATEMENT = `# Erklärung zur Barrierefreiheit / Accessibility statement

## English

Brick DNN is **partially accessible**. Write to [the operator](mailto:a@example.com).

## Deutsch

Brick DNN ist **teilweise barrierefrei**. Schreiben Sie [dem Betreiber](mailto:a@example.com).`;

function renderAt(path: string, element: React.ReactNode, routes: string[] = []) {
  return render(
    <AuthProvider>
      <MemoryRouter initialEntries={[path]}>
        <SkipLink />
        <header>
          <AppHeader />
        </header>
        <main id="main" tabIndex={-1}>
          <Routes>
            <Route path={path} element={element} />
            {routes.map((route) => (
              <Route key={route} path={route} element={<p>{route}</p>} />
            ))}
          </Routes>
        </main>
      </MemoryRouter>
    </AuthProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  setTokens(null);
  resetLegalInfo();
  resetBillingInfo();
  vi.mocked(api.me).mockRejectedValue(new ApiError(401, null, 'Not logged in.'));
  vi.mocked(api.legal.info).mockResolvedValue(LEGAL);
  vi.mocked(api.billing.info).mockResolvedValue(BILLING);
});

describe.each(['en', 'de'] as const)('axe in %s', (language) => {
  beforeEach(async () => {
    await i18n.changeLanguage(language);
  });

  it('finds nothing serious on the login page', async () => {
    renderAt('/login', <LoginPage />);
    await screen.findAllByRole('link', { name: language === 'de' ? 'Impressum' : 'Imprint' });
    expect(await axeViolations()).toEqual([]);
  });

  it('finds nothing serious on the sign-up page, with the terms to accept', async () => {
    renderAt('/register', <RegisterPage />);
    await screen.findByRole('checkbox');
    expect(await axeViolations()).toEqual([]);
  });

  it('finds nothing serious on the accessibility statement', async () => {
    vi.mocked(api.legal.page).mockResolvedValue({
      page: 'accessibility',
      title: 'Accessibility statement',
      content: STATEMENT,
      version: 'abc',
      updated_at: '2026-11-15T10:00:00Z',
    });
    renderAt('/accessibility', <LegalPage page="accessibility" />);
    // The section of the app's language.
    expect(
      await screen.findByText(language === 'de' ? /teilweise barrierefrei/ : /partially accessible/)
    ).toBeInTheDocument();
    expect(await axeViolations()).toEqual([]);
  });

  it('finds nothing serious on the pricing page', async () => {
    renderAt('/pricing', <PricingPage />);
    await screen.findAllByText('Starter');
    expect(await axeViolations()).toEqual([]);
  });
});

describe('axe', () => {
  it('finds nothing serious on the forgotten password page', async () => {
    vi.mocked(api.passwordReset.available).mockResolvedValue({ available: true });
    renderAt('/forgot-password', <ForgotPasswordPage />);
    await screen.findAllByRole('textbox');
    expect(await axeViolations()).toEqual([]);
  });

  it('finds nothing serious on the page for a missing address', async () => {
    renderAt('/nowhere', <NotFoundPage />);
    expect(await axeViolations()).toEqual([]);
  });

  it('finds nothing serious in a line chart and a chart with a table view', async () => {
    render(
      <>
        <LineChart
          title="Loss"
          x={[1, 2, 3]}
          width={400}
          series={[
            {
              id: 'train',
              label: 'Training',
              color: 'var(--chart-series-1)',
              values: [1, 0.6, 0.4],
            },
            {
              id: 'val',
              label: 'Validation',
              color: 'var(--chart-series-2)',
              values: [1.1, 0.8, null],
            },
          ]}
        />
        <ChartFrame title="Classes" table={() => ({ head: ['Class', 'Rows'], rows: [['a', 3]] })}>
          <svg role="img" aria-label="Classes: a 3" />
        </ChartFrame>
      </>
    );
    expect(screen.getByRole('slider', { name: /^Loss/ })).toHaveAttribute(
      'aria-valuetext',
      'Epoch 3: Training 0.4, Validation no value'
    );
    expect(await axeViolations()).toEqual([]);
  });

  it('links an error of the server to its field and puts the focus there', async () => {
    vi.mocked(api.register).mockRejectedValue(
      new ApiError(
        400,
        { username: ['A user with that username already exists.'] },
        'Username: A user with that username already exists.'
      )
    );
    renderAt('/register', <RegisterPage />);
    await userEvent.type(screen.getByLabelText(/Username/), 'ada');
    await userEvent.type(screen.getByLabelText(/Email/), 'ada@example.com');
    await userEvent.type(screen.getByLabelText(/^Password/), 'correct-horse');
    await userEvent.click(await screen.findByRole('checkbox'));
    await userEvent.click(screen.getByRole('button', { name: 'Create account' }));
    const field = screen.getByLabelText(/Username/);
    await vi.waitFor(() => expect(field).toHaveFocus());
    expect(field).toHaveAttribute('aria-invalid', 'true');
    expect(field).toHaveAccessibleDescription('A user with that username already exists.');
    expect(await axeViolations()).toEqual([]);
  });

  it('starts every page with a skip link to the main content', async () => {
    renderAt('/nowhere', <NotFoundPage />);
    const skip = screen.getByRole('link', { name: 'Skip to content' });
    skip.click();
    expect(document.getElementById('main')).toHaveFocus();
    expect(screen.getByRole('navigation', { name: 'Main' })).toBeInTheDocument();
  });
});
