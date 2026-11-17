import { axeViolations, render, screen, userEvent, within } from '@test-utils';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, vi } from 'vitest';
import { ModalsProvider } from '@mantine/modals';
import { api, type NewApiToken } from '@/api';
import { i18n } from '@/i18n';
import { ApiTokensPage } from './ApiTokensPage';
import { loggingCommands } from './LoggingSetup';

/** The snippet with exactly this text (line breaks included). */
function snippet(text: string) {
  return (_: string, element: Element | null) =>
    element?.tagName === 'PRE' && element.textContent === text;
}

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    ...actual,
    api: {
      apiTokens: { list: vi.fn(), create: vi.fn(), revoke: vi.fn(), setBudget: vi.fn() },
      connections: { list: vi.fn(), revoke: vi.fn(), setBudget: vi.fn() },
      agentActivity: { summary: vi.fn() },
      approvals: { rules: vi.fn(), setRule: vi.fn() },
    },
  };
});

const SECRET = `bdnn_${'g'.repeat(43)}`;
const LOGGER: NewApiToken = {
  id: 4,
  name: 'GPU box',
  prefix: 'bdnn_gggggggg',
  read_only: false,
  scope: 'runs:log',
  created_at: '2026-11-17T09:00:00Z',
  last_used_at: null,
  expires_at: null,
  token: SECRET,
};
const RULE = { training_cpu_minutes: 15, publishing: true, deleting: false };

function renderPage() {
  return render(
    <MemoryRouter>
      <ModalsProvider>
        <ApiTokensPage />
      </ModalsProvider>
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.apiTokens.list).mockResolvedValue([{ ...LOGGER, token: undefined } as never]);
  vi.mocked(api.connections.list).mockResolvedValue([]);
  vi.mocked(api.agentActivity.summary).mockResolvedValue({
    agents: [],
    tools: [],
    total: 0,
    latest_id: 0,
    latest_at: null,
    retention_days: 90,
  });
  vi.mocked(api.approvals.rules).mockResolvedValue({
    default: RULE,
    builtin_default: RULE,
    agents: {},
    gated_tools: { training: [], publishing: [], deleting: [] },
    expiry_hours: 24,
  });
});

it('explains how to log a training from the own machine, with a copyable snippet', async () => {
  const user = userEvent.setup(); // with a stub of the clipboard
  renderPage();
  const section = await screen.findByRole('region', {
    name: 'Log trainings from your own machine',
  });
  const url = window.location.origin;
  expect(within(section).getByText(snippet(loggingCommands(null, url)))).toBeInTheDocument();
  expect(loggingCommands(null, url)).toContain('export BRICKDNN_TOKEN=<your token>');
  expect(
    within(section).getByText(/from brickdnn.tracking import BrickDNNRun/)
  ).toBeInTheDocument();
  await user.click(within(section).getByRole('button', { name: 'Copy the commands' }));
  expect(await navigator.clipboard.readText()).toContain(`export BRICKDNN_URL=${url}`);
  // The token that may only log runs is marked in the list.
  const item = await screen.findByRole('listitem', { name: 'GPU box' });
  expect(within(item).getByText('Logs runs')).toBeInTheDocument();
});

it('creates a token that may only log runs and fills it into the commands', async () => {
  vi.mocked(api.apiTokens.create).mockResolvedValue(LOGGER);
  renderPage();
  await screen.findByRole('listitem', { name: 'GPU box' });
  await userEvent.click(screen.getByRole('button', { name: 'Create token' }));
  const dialog = await screen.findByRole('dialog', { name: 'Create an API token' });
  await userEvent.type(within(dialog).getByRole('textbox', { name: 'Name' }), 'GPU box');
  await userEvent.click(within(dialog).getByRole('radio', { name: 'Only log training runs' }));
  await userEvent.click(within(dialog).getByRole('button', { name: 'Create token' }));
  expect(api.apiTokens.create).toHaveBeenCalledWith({
    name: 'GPU box',
    read_only: false,
    scope: 'runs:log',
    expires_in_days: 90,
  });
  const done = await screen.findByRole('dialog', { name: 'Token created' });
  expect(
    within(done).getByText(snippet(loggingCommands(SECRET, window.location.origin)))
  ).toBeInTheDocument();
  expect(within(done).queryByText(/claude mcp add/)).not.toBeInTheDocument();
});

it('finds nothing serious with axe, in German too', async () => {
  await i18n.changeLanguage('de');
  renderPage();
  await screen.findByRole('region', { name: 'Trainings vom eigenen Rechner protokollieren' });
  expect(screen.getByText('Protokolliert Läufe')).toBeInTheDocument();
  expect(await axeViolations()).toEqual([]);
});
