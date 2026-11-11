import { useEffect, useMemo, useState } from 'react';
import { IconBook, IconKey, IconRobot } from '@tabler/icons-react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import {
  Alert,
  Anchor,
  Button,
  Card,
  Center,
  Container,
  Group,
  Loader,
  Select,
  Stack,
  Switch,
  Text,
  ThemeIcon,
  Title,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import {
  api,
  type ActivityAgent,
  type AgentAction,
  type AgentKey,
  type AgentRef,
  type AgentTarget,
} from '@/api';
import { ApprovalsPanel } from '@/approvals/ApprovalsPanel';
import { projectDraft } from '@/builder/draft';
import { useOpenInBuilder } from '@/builder/useOpenInBuilder';
import { groupByDay } from '@/collab/time';
import { REPOSITORY_URL } from '@/links';
import { ActivityItem, type OpenTarget } from './ActivityItem';
import { groupEntries } from './format';
import {
  POLL_INTERVAL,
  useAgentActivity,
  useNow,
  type ActivityFilters,
  type LiveState,
} from './useAgentActivity';
import classes from './Activity.module.css';

export const MCP_DOCS_URL = `${REPOSITORY_URL}/blob/main/docs/mcp.md`;

const AGENT_KEY = /^(token|app):\d+$/;

/** The filters live in the URL: ?agent=token:5&errors=1&project=12 (links from elsewhere). */
function filtersFrom(params: URLSearchParams): ActivityFilters {
  const agent = params.get('agent');
  const project = Number(params.get('project'));
  return {
    agent: agent && AGENT_KEY.test(agent) ? (agent as AgentKey) : null,
    errorsOnly: params.get('errors') === '1',
    project: Number.isInteger(project) && project > 0 ? project : null,
  };
}

interface AgentOption {
  agent: AgentRef;
  /** false: revoked or disconnected; null: not known yet. */
  active: boolean | null;
}

/** The tokens and apps to filter by: those the summary lists and those on the page. */
function agentOptions(known: ActivityAgent[], entries: AgentAction[] | null): AgentOption[] {
  const options = new Map<string, AgentOption>(
    known.map((agent) => [agent.key, { agent, active: agent.active }])
  );
  for (const entry of entries ?? []) {
    if (!options.has(entry.agent.key)) {
      options.set(entry.agent.key, { agent: entry.agent, active: null });
    }
  }
  return [...options.values()];
}

function optionLabel({ agent, active }: AgentOption): string {
  if (active === false) {
    return `${agent.name} (${agent.kind === 'app' ? 'disconnected' : 'revoked'})`;
  }
  return agent.prefix ? `${agent.name} · ${agent.prefix}…` : agent.name;
}

function LiveIndicator({ state }: { state: LiveState }) {
  const label = { live: 'Live', paused: 'Paused', offline: 'Reconnecting…' }[state];
  const title = {
    live: 'New actions appear as they happen.',
    paused: 'Paused while the page is in the background.',
    offline: 'The server cannot be reached; trying again.',
  }[state];
  return (
    <span className={classes.live} title={title} role="status">
      <span className={classes.liveDot} data-state={state} aria-hidden />
      {label}
    </span>
  );
}

/** What links in the log do: open the network, run, dataset or data preparation. */
function useOpenTarget(): OpenTarget {
  const navigate = useNavigate();
  const openInBuilder = useOpenInBuilder();
  return (target: AgentTarget) => {
    switch (target.kind) {
      case 'project':
        api.projects
          .get(Number(target.id))
          .then((project) => openInBuilder(projectDraft(project)))
          .catch(() =>
            notifications.show({
              color: 'red',
              message: `“${target.name}” was deleted or is no longer shared with you.`,
            })
          );
        break;
      case 'run':
        navigate('/', { state: { openTraining: true, run: String(target.id) } });
        break;
      case 'dataset':
        navigate('/', { state: { openTraining: true, dataset: Number(target.id) } });
        break;
      case 'data_prep':
        navigate(`/data/${target.id}`);
        break;
      case 'data_source':
        navigate('/data');
        break;
    }
  };
}

function EmptyLog() {
  return (
    <Card withBorder p="xl" mt="lg">
      <Stack align="center" gap="sm" ta="center">
        <ThemeIcon variant="light" size={48} radius="xl" aria-hidden>
          <IconRobot size={26} />
        </ThemeIcon>
        <Title order={3} size="h4">
          No agent activity yet
        </Title>
        <Text c="dimmed" maw={520}>
          When an AI assistant such as Claude or one of your scripts uses Brick DNN with an API
          token or as a connected app, every action shows up here as it happens: which networks it
          saved, which trainings it started, what failed.
        </Text>
        <Group justify="center" gap="sm">
          <Button component={Link} to="/tokens" leftSection={<IconKey size={16} />}>
            Connect an assistant
          </Button>
          <Button
            component="a"
            href={MCP_DOCS_URL}
            target="_blank"
            rel="noreferrer"
            variant="default"
            leftSection={<IconBook size={16} />}
          >
            How it works (MCP)
          </Button>
        </Group>
      </Stack>
    </Card>
  );
}

export function AgentActivityPage({ pollInterval = POLL_INTERVAL }: { pollInterval?: number }) {
  const [params, setParams] = useSearchParams();
  const filters = filtersFrom(params);
  const activity = useAgentActivity(filters, pollInterval);
  const [agents, setAgents] = useState<ActivityAgent[]>([]);
  const now = useNow();
  const openTarget = useOpenTarget();

  useEffect(() => {
    const controller = new AbortController();
    api.agentActivity
      .summary(controller.signal)
      .then((summary) => setAgents(summary.agents))
      .catch(() => undefined); // The filter then lists the agents on the page only.
    return () => controller.abort();
  }, []);

  const setFilter = (name: string, value: string | null) =>
    setParams(
      (current) => {
        const next = new URLSearchParams(current);
        if (value) {
          next.set(name, value);
        } else {
          next.delete(name);
        }
        return next;
      },
      { replace: true }
    );

  const { entries } = activity;
  const waiting = useMemo(
    () => new Set((activity.approvals ?? []).map((approval) => approval.id)),
    [activity.approvals]
  );
  const options = agentOptions(agents, entries);
  const days = useMemo(
    () =>
      groupByDay(entries ?? [], (entry) => entry.created_at, now).map((day) => ({
        ...day,
        groups: groupEntries(day.items),
      })),
    [entries, now]
  );
  const filtered = filters.agent !== null || filters.errorsOnly || filters.project !== null;
  const retention = activity.retentionDays;

  return (
    <Container size="md" py="xl">
      <Group gap="sm" align="center">
        <Title order={2}>Agent activity</Title>
        {entries && retention !== 0 && <LiveIndicator state={activity.live} />}
      </Group>
      <Text mt="xs" className={classes.intro}>
        Everything AI assistants and scripts did in your account with your{' '}
        <Anchor component={Link} to="/tokens">
          API tokens and connected apps
        </Anchor>
        , as it happens.
        {retention ? ` Entries are kept for ${retention} days.` : ''}
      </Text>

      <ApprovalsPanel
        approvals={activity.approvals ?? []}
        now={now}
        onDecided={activity.decided}
        onOpen={openTarget}
      />

      {retention === 0 && (
        <Alert color="gray" mt="lg">
          This server keeps no log of what agents do.
        </Alert>
      )}

      {(options.length > 0 || filtered) && (
        <Group className={classes.filters} gap="md" align="flex-end" wrap="wrap">
          <Select
            className={classes.agentSelect}
            label="Token or app"
            placeholder="All tokens and apps"
            data={options.map((option) => ({
              value: option.agent.key,
              label: optionLabel(option),
            }))}
            value={filters.agent}
            onChange={(value) => setFilter('agent', value)}
            clearable
            comboboxProps={{ withinPortal: true }}
          />
          <Switch
            label="Errors only"
            checked={filters.errorsOnly}
            onChange={(event) => setFilter('errors', event.currentTarget.checked ? '1' : null)}
            mb={8}
          />
          {filters.project !== null && (
            <Button variant="subtle" size="compact-sm" onClick={() => setFilter('project', null)}>
              Only network #{filters.project} · show all
            </Button>
          )}
        </Group>
      )}

      {activity.error && (
        <Alert color="red" mt="lg">
          {activity.error}
        </Alert>
      )}
      {!entries && !activity.error && (
        <Center h={160}>
          <Loader />
        </Center>
      )}
      {entries?.length === 0 &&
        retention !== 0 &&
        (filtered ? (
          <Stack align="center" gap="xs" py="xl">
            <Text c="dimmed">Nothing matches these filters yet.</Text>
            <Button
              variant="light"
              onClick={() => setParams(new URLSearchParams(), { replace: true })}
            >
              Show all activity
            </Button>
          </Stack>
        ) : (
          <EmptyLog />
        ))}

      {days.map((day) => (
        <section key={day.key} aria-label={day.label}>
          <h3 className={classes.day}>{day.label}</h3>
          <ol className={classes.list}>
            {day.groups.map((group) => (
              <ActivityItem
                key={group.key}
                group={group}
                fresh={activity.fresh.has(group.key)}
                now={now}
                onOpen={openTarget}
                waiting={
                  group.entries[0].approval != null && waiting.has(group.entries[0].approval)
                }
              />
            ))}
          </ol>
        </section>
      ))}

      {entries && activity.hasMore && (
        <Center mt="lg">
          <Button variant="default" loading={activity.loadingMore} onClick={activity.loadMore}>
            Show older activity
          </Button>
        </Center>
      )}
    </Container>
  );
}
