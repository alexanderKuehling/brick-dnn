/**
 * The training panel at a limit of the account's plan: the message names the limit and the
 * plan that lifts it, with one button; the usage of the month is visible before it runs out.
 */
import { axeViolations, render, screen, userEvent, waitFor, within } from '@test-utils';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ModalsProvider } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import {
  api,
  ApiError,
  type Compatibility,
  type CostEstimate,
  type LimitInfo,
  type PlanOffer,
  type Usage,
} from '@/api';
import { FREE, STARTER } from '@/billing/testData';
import { NO_SCOPE } from './access';
import { ANALYSIS, detail, DOC, STATUS } from './fixtures';
import { DEFAULT_SOURCE, matchingBuiltin } from './settings';
import TrainingDrawer, { type TrainingDrawerProps } from './TrainingDrawer';
import type { TrainingController } from './useTraining';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      generate: vi.fn(),
      account: { usage: vi.fn(), setOverage: vi.fn() },
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

function controller(overrides: Partial<TrainingController> = {}): TrainingController {
  return {
    status: STATUS,
    statusError: null,
    reloadStatus: vi.fn(),
    runs: [],
    runsError: null,
    refreshRuns: vi.fn(),
    activeRun: null,
    scope: NO_SCOPE,
    opened: true,
    open: vi.fn(),
    close: vi.fn(),
    view: 'setup',
    showSetup: vi.fn(),
    showHistory: vi.fn(),
    openRun: vi.fn(),
    selectedRunId: null,
    detail: null,
    detailError: null,
    retryDetail: vi.fn(),
    applyRun: vi.fn(),
    forgetRun: vi.fn(),
    data: {
      source: DEFAULT_SOURCE,
      setSource: vi.fn(),
      selection: 0,
      datasets: [],
      datasetsError: null,
      reloadDatasets: vi.fn(),
      uploading: null,
      uploadProgress: null,
      uploadError: null,
      dismissUploadError: vi.fn(),
      upload: vi.fn(),
      cancelUpload: vi.fn(),
      remove: vi.fn(),
    },
    ...overrides,
  };
}

function renderDrawer(props: Partial<TrainingDrawerProps> = {}) {
  const all: TrainingDrawerProps = {
    training: controller(),
    loggedIn: true,
    doc: DOC,
    networkName: 'Tiny MLP',
    analysis: ANALYSIS,
    onLogin: vi.fn(),
    onShowProblems: vi.fn(),
    onLoadNetwork: vi.fn(),
    ...props,
  };
  render(
    <ModalsProvider>
      <TrainingDrawer {...all} />
    </ModalsProvider>
  );
  return all;
}

const startButton = () => screen.getByRole('button', { name: 'Start training' });

/** The message of the refused start (the panel has other notes with the role alert). */
async function limitAlert() {
  const button = await screen.findByText(/It is back on|trains one network at a time/);
  return within(button.closest('[role="alert"]') as HTMLElement);
}

const READY: Compatibility = {
  ok: true,
  errors: [],
  warnings: [],
  plan: {
    task: 'demo',
    summary: 'Demo data: random inputs and a hidden rule to learn (classification).',
    inputs: [],
    targets: [],
    image: null,
    num_classes: 3,
    class_names: null,
    notes: [],
  },
};

const ESTIMATE: CostEstimate = {
  cpu_minutes: 9.9,
  low_cpu_minutes: 3.3,
  high_cpu_minutes: 29.6,
  cost: null,
  currency: null,
  basis: '48,000 training samples, 10 epochs',
  note: 'A rough estimate.',
  details: {},
};

const OFFER: PlanOffer = {
  ...STARTER,
  storage_gb: 3,
  url: 'https://brickdnn.com/pricing?plan=starter',
};

/** What the server says about a free account whose month is used up. */
const USED_UP: LimitInfo = {
  limit: {
    kind: 'cpu_time',
    used: 30,
    limit: 30,
    unit: 'cpu_minutes',
    resets_at: '2026-12-01T00:00:00Z',
  },
  plan: { key: 'free', name: 'Free' },
  upgrade: OFFER,
  overage: null,
};
const SERVER_TEXT =
  'The monthly compute allowance of this account (Free plan) is used up: 30.0 CPU minutes of ' +
  '30.0 CPU minutes this month. The Starter plan (5.00 EUR per month) includes 10 CPU hours ' +
  'per month: https://brickdnn.com/pricing?plan=starter';

function usage(usedMinutes: number, selling = true): Usage {
  return {
    period: { start: '2026-11-01', resets_at: '2026-12-01T00:00:00Z' },
    plan: FREE,
    unlimited: false,
    plans: [],
    upgrade_url: selling ? 'https://brickdnn.com/pricing' : null,
    next_plans: { compute: selling ? OFFER : null, storage: selling ? OFFER : null },
    compute: {
      unit: 'cpu_minutes',
      used: usedMinutes,
      limit: 30,
      remaining: 30 - usedMinutes,
      available: 30 - usedMinutes,
      allowance: 30,
      own_budget: null,
      run_limit: 15,
      reserved_by_active_runs: 0,
      by_kind: { training: usedMinutes, prediction: 0 },
      by_source: [],
    },
    storage: {
      used_bytes: 1024 * 1024,
      limit_bytes: 512 * 1024 * 1024,
      remaining_bytes: 511 * 1024 * 1024,
      by_kind: { data_sources: 0, datasets: 0, training_runs: 1024 * 1024 },
    },
    prices: {
      currency: null,
      per_cpu_hour: null,
      per_gb_month: null,
      compute_cost: null,
      compute_limit_cost: null,
      storage_cost_per_month: null,
    },
    caller: null,
  };
}

function allowed(extra: object = {}) {
  return {
    ...READY,
    estimate: ESTIMATE,
    budget: {
      allowed: true,
      reason: null,
      warning: null,
      run_limit_cpu_minutes: 15,
      limit_reason: 'run_limit' as const,
      remaining_cpu_minutes: 30,
      resets_at: '2026-12-01T00:00:00Z',
      ...extra,
    },
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  vi.mocked(api.training.datasets.list).mockResolvedValue([]);
  vi.mocked(api.training.check).mockResolvedValue(READY);
  vi.mocked(api.training.estimate).mockResolvedValue(allowed());
  vi.mocked(api.account.usage).mockResolvedValue(usage(12));
  vi.spyOn(notifications, 'show').mockImplementation(() => '');
});

describe('a start that a limit refuses', () => {
  it('names the limit and the next plan, with one button to it', async () => {
    vi.mocked(api.training.runs.create).mockRejectedValue(
      new ApiError(403, { detail: SERVER_TEXT, code: 'budget_exceeded', ...USED_UP }, SERVER_TEXT)
    );
    renderDrawer();
    await waitFor(() => expect(startButton()).toBeEnabled(), { timeout: 3000 });
    await userEvent.click(startButton());
    const alert = await limitAlert();
    expect(alert.getByText('This month’s compute is used up')).toBeInTheDocument();
    expect(
      alert.getByText(
        'This month’s compute of the Free plan is used up: 30 CPU min of 30 CPU min. It is back on 1 December 2026.'
      )
    ).toBeInTheDocument();
    expect(
      alert.getByText(
        'The Starter plan includes 10 CPU hours per month, 3 GB of storage, and up to 1 CPU hour per training for €5 per month.'
      )
    ).toBeInTheDocument();
    const links = alert.getAllByRole('link');
    expect(links).toHaveLength(1);
    expect(links[0]).toHaveAccessibleName(/Upgrade to Starter/);
    expect(links[0]).toHaveAttribute('href', '/pricing?plan=starter');
    // The panel keeps its network: the plans open in a new tab.
    expect(links[0]).toHaveAttribute('target', '_blank');
    // No address to copy out of a sentence.
    expect(alert.queryByText(/https:\/\//)).not.toBeInTheDocument();
    expect(await axeViolations()).toEqual([]);
  });

  it('shows no plan, price or button on a server that sells none', async () => {
    const text = 'The monthly compute allowance of this account is used up.';
    vi.mocked(api.training.runs.create).mockRejectedValue(
      new ApiError(403, { detail: text, code: 'budget_exceeded', ...USED_UP, upgrade: null }, text)
    );
    vi.mocked(api.account.usage).mockResolvedValue(usage(12, false));
    renderDrawer();
    await waitFor(() => expect(startButton()).toBeEnabled(), { timeout: 3000 });
    await userEvent.click(startButton());
    const alert = await limitAlert();
    expect(alert.getByText(/It is back on 1 December 2026/)).toBeInTheDocument();
    expect(alert.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.queryByText(/€|Upgrade|See plans/)).not.toBeInTheDocument();
  });

  it('says which plan trains more at a time when one training is running', async () => {
    const text = 'Your plan (Free) allows 1 training at a time.';
    vi.mocked(api.training.runs.create).mockRejectedValue(
      new ApiError(
        409,
        {
          detail: text,
          code: 'concurrent_trainings_limit',
          limit: { kind: 'concurrent_trainings', used: 1, limit: 1, unit: 'trainings' },
          plan: { key: 'free', name: 'Free' },
          upgrade: { ...OFFER, key: 'pro', name: 'Pro', price: 45, concurrent_trainings: 2 },
        },
        text
      )
    );
    renderDrawer();
    await waitFor(() => expect(startButton()).toBeEnabled(), { timeout: 3000 });
    await userEvent.click(startButton());
    const alert = await limitAlert();
    expect(alert.getByText('A training is already running')).toBeInTheDocument();
    expect(alert.getByText(/The Free plan trains one network at a time/)).toBeInTheDocument();
    expect(alert.getByRole('link', { name: /Upgrade to Pro/ })).toHaveAttribute(
      'href',
      '/pricing?plan=pro'
    );
  });

  it('keeps the plain message when the server is busy with somebody else', async () => {
    const text = 'Another training is still running. Stop it or wait until it finishes.';
    vi.mocked(api.training.runs.create).mockRejectedValue(
      new ApiError(409, { detail: text, code: 'server_busy' }, text)
    );
    renderDrawer();
    await waitFor(() => expect(startButton()).toBeEnabled(), { timeout: 3000 });
    await userEvent.click(startButton());
    expect(await screen.findByText(text)).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /Upgrade/ })).not.toBeInTheDocument();
  });
});

describe('before the start', () => {
  it('refuses a training that is too long for the plan, with the plan it fits into', async () => {
    vi.mocked(api.training.estimate).mockResolvedValue(
      allowed({
        allowed: false,
        reason:
          'The estimated CPU time of this training is 9.9 CPU minutes, far more than it can use.',
        code: 'estimate_too_high',
        run_limit_cpu_minutes: 2,
        limit: { kind: 'run_time', limit: 2, unit: 'cpu_minutes' },
        plan: { key: 'free', name: 'Free' },
        upgrade: OFFER,
      })
    );
    renderDrawer();
    expect(
      await screen.findByText('Too long for one training', {}, { timeout: 3000 })
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'With the Free plan one training may use up to 2 CPU min; this one needs about 9.9 CPU min.'
      )
    ).toBeInTheDocument();
    expect(screen.getByText(/The Starter plan includes/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Upgrade to Starter/ })).toBeInTheDocument();
    expect(
      screen.getByText('Train fewer epochs, on fewer samples or a smaller network.')
    ).toBeInTheDocument();
    expect(startButton()).toBeDisabled();
  });

  it('warns up front when the estimate is above what is left, with the same plan', async () => {
    vi.mocked(api.training.estimate).mockResolvedValue(
      allowed({
        warning: 'It will probably be stopped before it finishes.',
        warning_code: 'estimate_above_limit',
        run_limit_cpu_minutes: 6,
        limit_reason: 'budget',
        remaining_cpu_minutes: 6,
        ...USED_UP,
        limit: { ...USED_UP.limit, used: 24 },
      })
    );
    renderDrawer();
    expect(
      await screen.findByText(
        '6 CPU min are left of this month’s compute; this training needs about 9.9 CPU min. It will probably stop before it finishes; what it learned by then is kept.',
        {},
        { timeout: 3000 }
      )
    ).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Upgrade to Starter/ })).toBeInTheDocument();
    // It may still start: the estimate is rough.
    await waitFor(() => expect(startButton()).toBeEnabled());
  });

  it('says how many trainings like this one fit into what is left', async () => {
    renderDrawer();
    const box = await screen.findByRole('region', { name: 'Estimated compute' }, { timeout: 3000 });
    expect(
      within(box).getByText(/About 3 trainings like this one fit into what is left\./)
    ).toBeInTheDocument();
  });
});

describe('the usage of the month in the panel', () => {
  it('is a quiet line while there is room', async () => {
    renderDrawer();
    expect(
      await screen.findByText(/Usage this month: 12 of 30 CPU min · .* of 512 MB storage/)
    ).toBeInTheDocument();
    expect(screen.getByTestId('usage-indicator')).toHaveAttribute('data-level', 'ok');
    expect(screen.queryByRole('link', { name: /See plans/ })).not.toBeInTheDocument();
  });

  it('warns at 80 % and links to the plans', async () => {
    vi.mocked(api.account.usage).mockResolvedValue(usage(24));
    renderDrawer();
    expect(
      await screen.findByText(/80 % of this month’s compute is used\. Starter includes/)
    ).toBeInTheDocument();
    expect(screen.getByTestId('usage-indicator')).toHaveAttribute('data-level', 'nearly');
    expect(screen.getByRole('link', { name: /See plans/ })).toHaveAttribute(
      'href',
      '/pricing?plan=starter'
    );
  });

  it('is asked again when the tab gets the focus back (after an upgrade elsewhere)', async () => {
    renderDrawer();
    await screen.findByText(/Usage this month: 12 of 30/);
    const estimates = vi.mocked(api.training.estimate).mock.calls.length;
    vi.mocked(api.account.usage).mockResolvedValue({
      ...usage(12),
      compute: { ...usage(12).compute, limit: 600, allowance: 600 },
    });
    window.dispatchEvent(new Event('focus'));
    expect(await screen.findByText(/Usage this month: 0.2 of 10 CPU h/)).toBeInTheDocument();
    await waitFor(
      () => expect(vi.mocked(api.training.estimate).mock.calls.length).toBeGreaterThan(estimates),
      { timeout: 3000 }
    );
  });

  it('is not shown without limits (a server of your own)', async () => {
    vi.mocked(api.account.usage).mockResolvedValue({ ...usage(12), unlimited: true });
    renderDrawer();
    await waitFor(() => expect(startButton()).toBeEnabled(), { timeout: 3000 });
    expect(screen.queryByTestId('usage-indicator')).not.toBeInTheDocument();
  });
});

describe('a run that a limit stopped', () => {
  it('names the plan with longer trainings', () => {
    const run = {
      ...detail({
        status: 'stopped',
        message: 'Stopped: this run used 15 CPU minutes, the most one training may use.',
        usage: {
          cpu_minutes: 15,
          limit_cpu_minutes: 15,
          estimated_cpu_minutes: 20,
          stop_reason: 'run_limit',
        },
      }),
      limit_hint: {
        limit: { kind: 'run_time', used: 15, limit: 15, unit: 'cpu_minutes' as const },
        plan: { key: 'free', name: 'Free' },
        upgrade: OFFER,
      },
    };
    renderDrawer({ training: controller({ view: 'run', selectedRunId: run.id, detail: run }) });
    expect(screen.getByText('Stopped at the limit per training')).toBeInTheDocument();
    expect(
      screen.getByText(/The Starter plan includes .* up to 1 CPU hour per training/)
    ).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Upgrade to Starter/ })).toHaveAttribute(
      'href',
      '/pricing?plan=starter'
    );
  });

  it('says nothing more without a hint (other viewers, servers without plans)', () => {
    const run = detail({
      status: 'stopped',
      message: 'Stopped: this run used 15 CPU minutes, the most one training may use.',
      usage: {
        cpu_minutes: 15,
        limit_cpu_minutes: 15,
        estimated_cpu_minutes: 20,
        stop_reason: 'run_limit',
      },
    });
    renderDrawer({ training: controller({ view: 'run', selectedRunId: run.id, detail: run }) });
    expect(screen.getByText('Stopped at the limit per training')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /Upgrade/ })).not.toBeInTheDocument();
  });
});

describe('the built-in dataset a network was made for', () => {
  const dataset = (id: string, name: string, shape: number[], extra: object = {}) => ({
    ...STATUS.builtin_datasets[0],
    id,
    name,
    sample_shape: shape,
    available: true,
    ...extra,
  });
  const DATASETS = [
    dataset('mnist', 'MNIST', [1, 28, 28]),
    dataset('fashion_mnist', 'Fashion-MNIST', [1, 28, 28]),
    dataset('cifar10', 'CIFAR-10', [3, 32, 32]),
    dataset('sentiment', 'Sentiment', [32], { input_kind: 'tokens' }),
  ];
  const network = (shape: number[] | string, dtype = 'float') => ({
    version: 1 as const,
    nodes: [{ id: 'in', type: 'Input', params: { shape, dtype } }],
    edges: [],
  });

  it('is found by the shape of the Input, flat or as an image', () => {
    expect(matchingBuiltin(network([784]), 'MLP classifier (MNIST)', DATASETS)).toBe('mnist');
    expect(matchingBuiltin(network([3, 32, 32]), 'My CNN', DATASETS)).toBe('cifar10');
    expect(matchingBuiltin(network([32], 'long'), 'LSTM', DATASETS)).toBe('sentiment');
  });

  it('is the one the network is named after when several fit', () => {
    expect(matchingBuiltin(network([1, 28, 28]), 'CNN (Fashion-MNIST)', DATASETS)).toBe(
      'fashion_mnist'
    );
    expect(matchingBuiltin(network([1, 28, 28]), 'Untitled', DATASETS)).toBe('mnist');
  });

  it('is none when nothing fits: demo data stays', () => {
    expect(matchingBuiltin(network([4]), 'Iris', DATASETS)).toBeNull();
    expect(matchingBuiltin(network([32]), 'Dense', DATASETS)).toBeNull();
    expect(matchingBuiltin(network('?'), 'Odd', DATASETS)).toBeNull();
    expect(
      matchingBuiltin(network([784]), 'MNIST', [
        dataset('mnist', 'MNIST', [1, 28, 28], { available: false }),
      ])
    ).toBeNull();
  });
});
