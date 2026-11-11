import { render, screen, userEvent, waitFor } from '@test-utils';
import { vi } from 'vitest';
import { notifications } from '@mantine/notifications';
import { api, ApiError, type ApprovalRequest } from '@/api';
import { ApprovalsPanel } from './ApprovalsPanel';
import { estimateText, expiresIn, ruleSummary } from './format';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: { approvals: { approve: vi.fn(), deny: vi.fn() } },
  };
});

const NOW = new Date('2026-11-10T12:00:00Z');

function approval(id: number, changes: Partial<ApprovalRequest> = {}): ApprovalRequest {
  return {
    id,
    status: 'pending',
    category: 'training',
    tool: 'start_training',
    action: 'start_training',
    agent: {
      kind: 'token',
      id: 5,
      key: 'token:5',
      name: 'Claude on my laptop',
      prefix: null,
      active: true,
    },
    summary: 'Start training “GP 1” on dataset “houses.csv” (#3)',
    reason: 'The estimated CPU time (42 CPU min) is above the 15 CPU min allowed.',
    estimate: {
      cpu_minutes: 42.1,
      low_cpu_minutes: 14,
      high_cpu_minutes: 126.3,
      cost: 0.7,
      currency: 'EUR',
    },
    limit_cpu_minutes: 15,
    network: '5 layers (Input, Linear ×2, ReLU, Softmax), 4 connections',
    dataset: 'dataset “houses.csv” (#3)',
    exposes: null,
    warning: null,
    targets: [{ kind: 'dataset', id: 3, name: 'houses.csv' }],
    arguments: {},
    created_at: '2026-11-10T11:55:00Z',
    expires_at: '2026-11-11T11:55:00Z',
    decided_at: null,
    finished_at: null,
    message: '',
    error: '',
    done: null,
    ...changes,
  };
}

const WEIGHTS = approval(8, {
  category: 'publishing',
  tool: 'get_weights_download',
  action: 'get_weights_download',
  summary: 'Get a download link for the weights of training “GP 1”',
  reason: 'Links that work without logging in need approval.',
  estimate: null,
  limit_cpu_minutes: null,
  network: null,
  dataset: null,
  exposes: 'A link to the trained weights that works without logging in for 15 minutes.',
  targets: [{ kind: 'run', id: 'abc', name: 'GP 1' }],
  agent: { kind: 'app', id: 7, key: 'app:7', name: 'Claude', prefix: null, active: true },
});

function setup(approvals = [approval(12), WEIGHTS]) {
  const onDecided = vi.fn();
  const onOpen = vi.fn();
  render(<ApprovalsPanel approvals={approvals} now={NOW} onDecided={onDecided} onOpen={onOpen} />);
  return { onDecided, onOpen };
}

describe('ApprovalsPanel', () => {
  beforeEach(() => {
    vi.mocked(api.approvals.approve).mockReset();
    vi.mocked(api.approvals.deny).mockReset();
  });

  it('shows nothing without waiting requests', () => {
    setup([]);
    expect(screen.queryByText('Waiting for your approval')).not.toBeInTheDocument();
  });

  it('shows what each request would do, with the estimate and what it exposes', async () => {
    const { onOpen } = setup();
    expect(screen.getByRole('heading', { name: 'Waiting for your approval' })).toBeInTheDocument();
    expect(screen.getByLabelText('2 waiting')).toBeInTheDocument();
    expect(screen.getByText('42.1 CPU min (14–126) · €0.70')).toBeInTheDocument();
    expect(screen.getByText('up to 15 CPU min')).toBeInTheDocument();
    expect(screen.getByText(/5 layers \(Input/)).toBeInTheDocument();
    expect(screen.getByText(/works without logging in for 15 minutes/)).toBeInTheDocument();
    expect(screen.getByText('Claude on my laptop')).toBeInTheDocument();
    expect(screen.getAllByText(/expires in 24 h/)).toHaveLength(2);

    await userEvent.click(screen.getByRole('button', { name: 'Open dataset “houses.csv”' }));
    expect(onOpen).toHaveBeenCalledWith({ kind: 'dataset', id: 3, name: 'houses.csv' });
  });

  it('approves and reports what was done', async () => {
    const show = vi.spyOn(notifications, 'show');
    const done = approval(12, { status: 'executed', done: 'Started training “GP 1”' });
    vi.mocked(api.approvals.approve).mockResolvedValue(done);
    const { onDecided } = setup();
    await userEvent.click(screen.getByRole('button', { name: /^Approve: Start training “GP 1”/ }));
    await waitFor(() => expect(onDecided).toHaveBeenCalledWith(done));
    expect(api.approvals.approve).toHaveBeenCalledWith(12);
    expect(show).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Approved', message: 'Started training “GP 1”' })
    );
  });

  it('says when an approved action failed', async () => {
    const show = vi.spyOn(notifications, 'show');
    const failed = approval(12, { status: 'failed', error: 'The budget is used up.' });
    vi.mocked(api.approvals.approve).mockResolvedValue(failed);
    setup();
    await userEvent.click(screen.getByRole('button', { name: /^Approve: Start/ }));
    await waitFor(() =>
      expect(show).toHaveBeenCalledWith(
        expect.objectContaining({ color: 'red', message: 'The budget is used up.' })
      )
    );
  });

  it('denies with a message for the agent', async () => {
    const denied = approval(8, { status: 'denied', message: 'Not public, please.' });
    vi.mocked(api.approvals.deny).mockResolvedValue(denied);
    const { onDecided } = setup();
    await userEvent.click(screen.getByRole('button', { name: /^Deny: Get a download link/ }));
    await userEvent.type(
      screen.getByLabelText('Message for the agent (optional)'),
      'Not public, please.'
    );
    await userEvent.click(screen.getByRole('button', { name: 'Deny' }));
    await waitFor(() => expect(onDecided).toHaveBeenCalledWith(denied));
    expect(api.approvals.deny).toHaveBeenCalledWith(8, 'Not public, please.');
  });

  it('shows the error when the request was decided elsewhere', async () => {
    const show = vi.spyOn(notifications, 'show');
    vi.mocked(api.approvals.approve).mockRejectedValue(
      new ApiError(409, null, 'This request has expired: it can no longer be approved.')
    );
    const { onDecided } = setup([approval(12)]);
    await userEvent.click(screen.getByRole('button', { name: /^Approve/ }));
    await waitFor(() =>
      expect(show).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'Could not approve', color: 'red' })
      )
    );
    expect(onDecided).not.toHaveBeenCalled();
  });

  it('warns when the token is gone', () => {
    setup([approval(12, { agent: { ...approval(1).agent, active: false } })]);
    expect(screen.getByText(/This token was revoked/)).toBeInTheDocument();
  });
});

describe('approval texts', () => {
  it('summarises rules', () => {
    expect(ruleSummary({ training_cpu_minutes: null, publishing: false, deleting: false })).toBe(
      'Never asks'
    );
    expect(ruleSummary({ training_cpu_minutes: 15, publishing: true, deleting: false })).toBe(
      'Asks before trainings above 15 CPU min and publishing'
    );
    expect(ruleSummary({ training_cpu_minutes: 0, publishing: true, deleting: true })).toBe(
      'Asks before every training, publishing and deleting'
    );
  });

  it('formats estimates and expiry', () => {
    expect(estimateText(null)).toBe('Cannot be estimated');
    expect(
      estimateText({
        cpu_minutes: 2.5,
        low_cpu_minutes: 0.83,
        high_cpu_minutes: 7.5,
        cost: null,
        currency: null,
      })
    ).toBe('2.5 CPU min (0.83–7.5)');
    expect(expiresIn('2026-11-10T12:40:00Z', NOW)).toBe('in 40 min');
    expect(expiresIn('2026-11-10T11:00:00Z', NOW)).toBe('now');
  });
});
