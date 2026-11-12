import { act, render, screen, userEvent, waitFor, within } from '@test-utils';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterAll, afterEach, beforeEach, vi } from 'vitest';
import { ModalsProvider } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import {
  api,
  type AgentAction,
  type AgentActivitySummary,
  type ApprovalRequest,
  type AgentActivityPage as Page,
  type Project,
} from '@/api';
import { AgentActivityPage } from './AgentActivityPage';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      agentActivity: { list: vi.fn(), summary: vi.fn() },
      approvals: { approve: vi.fn(), deny: vi.fn() },
      projects: { get: vi.fn() },
    },
  };
});

// A fixed noon: entries from "2 minutes ago" are on the same day (just after midnight they
// were yesterday, and the page then rightly said "Yesterday"). Only Date is faked; timers
// (polling, waitFor) run normally.
vi.useFakeTimers({ toFake: ['Date'] });
vi.setSystemTime(new Date(2026, 5, 15, 12, 0, 0));
afterAll(() => {
  vi.useRealTimers();
});

const minutesAgo = (minutes: number) => new Date(Date.now() - minutes * 60_000).toISOString();

const LAPTOP = {
  kind: 'token' as const,
  id: 5,
  key: 'token:5' as const,
  name: 'Claude on my laptop',
  prefix: 'bdnn_Ab3dEf9h',
};
const CLAUDE = { kind: 'app' as const, id: 7, key: 'app:7' as const, name: 'Claude', prefix: null };

function action(id: number, changes: Partial<AgentAction> = {}): AgentAction {
  return {
    id,
    created_at: minutesAgo(2),
    source: 'mcp',
    agent: LAPTOP,
    user_agent: 'claude-code/2.0.14',
    tool: 'list_projects',
    method: '',
    writes: false,
    outcome: 'ok',
    error: '',
    status_code: null,
    duration_ms: 42,
    summary: 'Listed the saved networks (3 networks)',
    arguments: {},
    targets: [],
    project: null,
    ...changes,
  };
}

const SAVED = action(3, {
  tool: 'save_project',
  writes: true,
  summary: 'Saved network “House prices” (#12) with 7 layers, now version 4',
  arguments: { id: 12, version: 3, graph: '7 layers (Input, Linear ×3, ReLU ×3), 6 connections' },
  targets: [{ kind: 'project', id: 12, name: 'House prices' }],
  project: 12,
});

const FAILED = action(2, {
  agent: CLAUDE,
  tool: 'start_training',
  writes: true,
  outcome: 'error',
  error: 'Too many trainings were started recently. Try again later.',
  summary: 'Could not start training “GP 1” on dataset “houses.csv” (#3)',
  arguments: { name: 'GP 1', dataset: { kind: 'upload', id: 3 } },
  targets: [{ kind: 'dataset', id: 3, name: 'houses.csv' }],
  user_agent: 'Claude-User',
  duration_ms: 1234,
});

const UPLOAD = action(1, {
  tool: 'upload_dataset',
  writes: true,
  summary: 'Uploaded dataset “houses.csv” (#3): 48 KB, 801 lines, 9 columns',
  arguments: { name: 'houses.csv', csv: '48 KB, 801 lines, 9 columns' },
  targets: [{ kind: 'dataset', id: 3, name: 'houses.csv' }],
  created_at: minutesAgo(5),
});

function page(results: AgentAction[], changes: Partial<Page> = {}): Page {
  return {
    results,
    has_more: false,
    latest_id: results[0]?.id ?? 0,
    retention_days: 90,
    ...changes,
  };
}

const SUMMARY: AgentActivitySummary = {
  agents: [
    { ...LAPTOP, active: true, actions: 2, errors: 0, last_at: minutesAgo(2), last_id: 3 },
    { ...CLAUDE, active: false, actions: 1, errors: 1, last_at: minutesAgo(2), last_id: 2 },
  ],
  tools: [],
  total: 3,
  latest_id: 3,
  latest_at: minutesAgo(2),
  retention_days: 90,
};

function Where() {
  const location = useLocation();
  return (
    <p data-testid="where">
      {location.pathname} {JSON.stringify(location.state)}
    </p>
  );
}

function renderPage(url = '/activity', pollInterval = 30) {
  return render(
    <MemoryRouter initialEntries={[url]}>
      <ModalsProvider>
        <Routes>
          <Route path="/activity" element={<AgentActivityPage pollInterval={pollInterval} />} />
          <Route path="*" element={<Where />} />
        </Routes>
      </ModalsProvider>
    </MemoryRouter>
  );
}

/** Answers the first request with `first`, polls with `polls` one after the other, then
 * with nothing new. */
function answer(first: Page, ...polls: Page[]) {
  const queue = [...polls];
  vi.mocked(api.agentActivity.list).mockImplementation(async (query = {}) => {
    if (query.after === undefined) {
      return first;
    }
    return queue.shift() ?? page([], { latest_id: query.after });
  });
}

function setVisibility(state: 'visible' | 'hidden') {
  Object.defineProperty(document, 'visibilityState', { configurable: true, value: state });
  document.dispatchEvent(new Event('visibilitychange'));
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  vi.spyOn(notifications, 'show').mockImplementation(() => 'id');
  vi.mocked(api.agentActivity.summary).mockResolvedValue(SUMMARY);
  answer(page([SAVED, FAILED, UPLOAD]));
});

afterEach(() => setVisibility('visible'));

test('lists what the agents did, newest first, by day', async () => {
  renderPage();
  const saved = await screen.findByRole('listitem', { name: SAVED.summary });
  expect(within(saved).getByText('Claude on my laptop')).toBeInTheDocument();
  expect(within(saved).getByText('save_project')).toBeInTheDocument();
  expect(within(saved).getByText('2 min ago')).toBeInTheDocument();
  expect(within(saved).getByText('42 ms')).toBeInTheDocument();
  expect(
    within(saved).getByRole('button', { name: 'Open network “House prices”' })
  ).toHaveTextContent('“House prices”');

  const items = screen.getAllByRole('listitem');
  expect(items.map((item) => item.getAttribute('aria-label'))).toEqual([
    SAVED.summary,
    FAILED.summary,
    UPLOAD.summary,
  ]);
  expect(screen.getByRole('heading', { name: 'Today' })).toBeInTheDocument();
  expect(screen.getByText(/Entries are kept for 90 days/)).toBeInTheDocument();
  expect(screen.getByRole('status')).toHaveTextContent('Live');
  expect(api.agentActivity.list).toHaveBeenCalledWith({ limit: 50 }, expect.any(AbortSignal));
});

test('marks failed actions with what went wrong', async () => {
  renderPage();
  const failed = await screen.findByRole('listitem', { name: FAILED.summary });
  expect(failed).toHaveAttribute('data-error');
  expect(within(failed).getByText(FAILED.error)).toBeInTheDocument();
  expect(within(failed).getByText('Claude')).toBeInTheDocument();
  expect(screen.getByRole('listitem', { name: SAVED.summary })).not.toHaveAttribute('data-error');
});

test('shows the summarised arguments and the duration in the details', async () => {
  const user = userEvent.setup();
  renderPage();
  const upload = await screen.findByRole('listitem', { name: UPLOAD.summary });
  await user.click(within(upload).getByRole('button', { name: 'Show details' }));
  expect(within(upload).getByText('csv').parentElement).toHaveTextContent(
    'csv: 48 KB, 801 lines, 9 columns'
  );
  expect(within(upload).getByText(/took 42 ms/)).toBeInTheDocument();
  expect(within(upload).getByText('claude-code/2.0.14')).toBeInTheDocument();
  expect(within(upload).getByText('bdnn_Ab3dEf9h…', { exact: false })).toBeInTheDocument();
  expect(within(upload).getByText(/never tokens or passwords/)).toBeInTheDocument();

  const failed = screen.getByRole('listitem', { name: FAILED.summary });
  await user.click(within(failed).getByText(FAILED.error)); // the whole row opens it
  expect(within(failed).getByText(/took 1.2 s/)).toBeInTheDocument();
  expect(within(failed).getByText('dataset').parentElement).toHaveTextContent(
    'dataset: kind: upload, id: 3'
  );
});

test('filters by token or app and by errors', async () => {
  const user = userEvent.setup();
  renderPage();
  await screen.findByRole('listitem', { name: SAVED.summary });

  await user.click(screen.getByRole('combobox', { name: 'Token or app' }));
  // Revoked tokens and disconnected apps stay in the list, marked.
  await user.click(await screen.findByRole('option', { name: 'Claude (disconnected)' }));
  await waitFor(() =>
    expect(api.agentActivity.list).toHaveBeenCalledWith(
      { agent: 'app:7', limit: 50 },
      expect.any(AbortSignal)
    )
  );

  await user.click(screen.getByRole('switch', { name: 'Errors only' }));
  await waitFor(() =>
    expect(api.agentActivity.list).toHaveBeenCalledWith(
      { agent: 'app:7', outcome: 'error', limit: 50 },
      expect.any(AbortSignal)
    )
  );
});

test('takes the filters from the address', async () => {
  answer(page([]));
  renderPage('/activity?agent=token:5&errors=1&project=12');
  expect(await screen.findByText('Nothing matches these filters yet.')).toBeInTheDocument();
  expect(api.agentActivity.list).toHaveBeenCalledWith(
    { agent: 'token:5', outcome: 'error', project: 12, limit: 50 },
    expect.any(AbortSignal)
  );
  expect(screen.getByRole('button', { name: /Only network #12/ })).toBeInTheDocument();
});

test('new actions appear while the page is open', async () => {
  const next = action(4, {
    tool: 'start_training',
    writes: true,
    summary: 'Started training “GP 1” on dataset “houses.csv” (#3)',
    targets: [
      { kind: 'run', id: 'b1d6', name: 'GP 1' },
      { kind: 'dataset', id: 3, name: 'houses.csv' },
    ],
    created_at: new Date().toISOString(),
  });
  answer(page([SAVED, FAILED, UPLOAD]), page([next]));
  renderPage();
  const started = await screen.findByRole('listitem', { name: next.summary });
  expect(started).toHaveAttribute('data-fresh');
  expect(within(started).getByText('just now')).toBeInTheDocument();
  expect(screen.getAllByRole('listitem')[0]).toBe(started);
  // What was there before does not animate again.
  expect(screen.getByRole('listitem', { name: SAVED.summary })).not.toHaveAttribute('data-fresh');
  expect(api.agentActivity.list).toHaveBeenCalledWith({ after: 3, limit: 100 });
  // Then it asks for what is newer than the new entry.
  await waitFor(() =>
    expect(api.agentActivity.list).toHaveBeenCalledWith({ after: 4, limit: 100 })
  );
});

test('repeated checks of a training are one row that counts up', async () => {
  const check = (id: number, epoch: number) =>
    action(id, {
      tool: 'get_training_run',
      summary: `Checked training “GP 1”: running, epoch ${epoch} of 20`,
      targets: [{ kind: 'run', id: 'b1d6', name: 'GP 1' }],
    });
  answer(page([check(12, 3), check(11, 2), check(10, 1), SAVED]), page([check(13, 4)]));
  renderPage();
  const row = await screen.findByRole('listitem', {
    name: 'Checked training “GP 1”: running, epoch 4 of 20',
  });
  expect(within(row).getByLabelText('4 calls in a row')).toHaveTextContent('×4');
  expect(screen.getAllByRole('listitem')).toHaveLength(2);
  // The row stayed where it was: it does not slide in like a new one.
  expect(row).not.toHaveAttribute('data-fresh');
  await userEvent.setup().click(within(row).getByRole('button', { name: 'Show details' }));
  expect(within(row).getByText('4 calls in a row')).toBeInTheDocument();
  expect(within(row).getByText(/epoch 1 of 20/)).toBeInTheDocument();
});

test('pauses while the page is in the background', async () => {
  renderPage();
  await screen.findByRole('listitem', { name: SAVED.summary });
  act(() => setVisibility('hidden'));
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Paused'));
  const calls = vi.mocked(api.agentActivity.list).mock.calls.length;
  await new Promise((resolve) => setTimeout(resolve, 120));
  expect(vi.mocked(api.agentActivity.list).mock.calls.length).toBe(calls);

  act(() => setVisibility('visible'));
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Live'));
  await waitFor(() =>
    expect(vi.mocked(api.agentActivity.list).mock.calls.length).toBeGreaterThan(calls)
  );
});

test('says when the server cannot be reached', async () => {
  renderPage();
  await screen.findByRole('listitem', { name: SAVED.summary });
  vi.mocked(api.agentActivity.list).mockRejectedValue(new Error('Network error'));
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Reconnecting…'));
  // What was there stays.
  expect(screen.getByRole('listitem', { name: SAVED.summary })).toBeInTheDocument();
});

test('explains how to connect an agent when there is nothing yet', async () => {
  answer(page([]));
  vi.mocked(api.agentActivity.summary).mockResolvedValue({ ...SUMMARY, agents: [] });
  renderPage();
  expect(await screen.findByRole('heading', { name: 'No agent activity yet' })).toBeVisible();
  expect(screen.getByRole('link', { name: 'Connect an assistant' })).toHaveAttribute(
    'href',
    '/tokens'
  );
  expect(screen.getByRole('link', { name: 'How it works (MCP)' })).toHaveAttribute(
    'href',
    expect.stringContaining('docs/mcp.md')
  );
  expect(screen.queryByRole('combobox', { name: 'Token or app' })).not.toBeInTheDocument();
});

test('says when the server keeps no log', async () => {
  answer(page([], { retention_days: 0 }));
  renderPage();
  expect(await screen.findByText('This server keeps no log of what agents do.')).toBeVisible();
  expect(screen.queryByText('No agent activity yet')).not.toBeInTheDocument();
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
});

test('loads older entries', async () => {
  const older = action(1, { summary: 'Looked up the layer types', tool: 'list_layer_types' });
  vi.mocked(api.agentActivity.list).mockImplementation(async (query = {}) => {
    if (query.before === 2) {
      return page([older]);
    }
    if (query.after !== undefined) {
      return page([], { latest_id: 3 });
    }
    return page([SAVED, FAILED], { has_more: true });
  });
  const user = userEvent.setup();
  renderPage();
  await user.click(await screen.findByRole('button', { name: 'Show older activity' }));
  expect(await screen.findByRole('listitem', { name: older.summary })).toBeInTheDocument();
  expect(api.agentActivity.list).toHaveBeenCalledWith({ before: 2, limit: 50 });
  expect(screen.queryByRole('button', { name: 'Show older activity' })).not.toBeInTheDocument();
});

test('links open the network, training or dataset', async () => {
  const project = { id: 12, name: 'House prices', graph: { version: 1, nodes: [], edges: [] } };
  vi.mocked(api.projects.get).mockResolvedValue(project as unknown as Project);
  const user = userEvent.setup();
  renderPage();
  await user.click(await screen.findByRole('button', { name: 'Open network “House prices”' }));
  expect(api.projects.get).toHaveBeenCalledWith(12);
  expect(await screen.findByTestId('where')).toHaveTextContent('/');
  expect(JSON.parse(localStorage.getItem('brickdnn:draft') ?? '{}').projectId).toBe(12);
});

test('a link to a dataset opens it in the training panel', async () => {
  const user = userEvent.setup();
  renderPage();
  const upload = await screen.findByRole('listitem', { name: UPLOAD.summary });
  await user.click(within(upload).getByRole('button', { name: 'Open dataset “houses.csv”' }));
  expect(await screen.findByTestId('where')).toHaveTextContent(
    '/ {"openTraining":true,"dataset":3}'
  );
});

test('a network that is gone says so', async () => {
  vi.mocked(api.projects.get).mockRejectedValue(new Error('Not found.'));
  const user = userEvent.setup();
  renderPage();
  await user.click(await screen.findByRole('button', { name: 'Open network “House prices”' }));
  await waitFor(() =>
    expect(notifications.show).toHaveBeenCalledWith(
      expect.objectContaining({ color: 'red', message: expect.stringContaining('deleted') })
    )
  );
});

const WAITING: ApprovalRequest = {
  id: 12,
  status: 'pending',
  category: 'training',
  tool: 'start_training',
  action: 'start_training',
  agent: { ...LAPTOP, active: true },
  summary: 'Start training “Big” on random demo data',
  reason: 'Every training needs approval (estimated 42 CPU min).',
  estimate: {
    cpu_minutes: 42,
    low_cpu_minutes: 14,
    high_cpu_minutes: 126,
    cost: null,
    currency: null,
  },
  limit_cpu_minutes: 0,
  network: '4 layers (Input, Linear ×2, Activation), 3 connections',
  dataset: 'random demo data',
  exposes: null,
  warning: null,
  targets: [],
  arguments: {},
  created_at: minutesAgo(1),
  expires_at: new Date(Date.now() + 86_400_000).toISOString(),
  decided_at: null,
  finished_at: null,
  message: '',
  error: '',
  done: null,
};

const ASKED = action(4, {
  tool: 'start_training',
  writes: true,
  outcome: 'pending',
  approval: 12,
  summary: 'Asked for approval to start training “Big” on random demo data (request #12)',
});

test('requests waiting for approval are on top and in the log', async () => {
  const user = userEvent.setup();
  answer(
    page([ASKED, SAVED], { approvals: { pending_count: 1, pending: [WAITING] } }),
    page([], { latest_id: 4, approvals: { pending_count: 1, pending: [WAITING] } })
  );
  vi.mocked(api.approvals.approve).mockResolvedValue({
    ...WAITING,
    status: 'executed',
    done: 'Started training “Big” on random demo data, approved by you (request #12)',
  });
  renderPage();
  const panel = await screen.findByRole('region', { name: 'Waiting for your approval' });
  expect(within(panel).getByText('Start training “Big” on random demo data')).toBeInTheDocument();
  expect(within(panel).getByText('42 CPU min (14–126)')).toBeInTheDocument();
  const asked = screen.getByRole('listitem', { name: ASKED.summary });
  expect(asked).toHaveAttribute('data-pending');
  expect(within(asked).getByText('Waiting for approval #12')).toBeInTheDocument();

  await user.click(within(panel).getByRole('button', { name: /^Approve/ }));
  await waitFor(() =>
    expect(screen.queryByRole('region', { name: 'Waiting for your approval' })).toBeNull()
  );
  expect(api.approvals.approve).toHaveBeenCalledWith(12);
  // The next poll still lists it (the server was asked before): it stays gone.
  await act(() => new Promise((resolve) => setTimeout(resolve, 80)));
  expect(screen.queryByRole('region', { name: 'Waiting for your approval' })).toBeNull();
  expect(within(asked).getByText('Asked for approval #12')).toBeInTheDocument();
});
