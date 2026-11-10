import { useCallback, useEffect, useState } from 'react';
import { IconKey, IconPlus, IconTrash } from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import {
  Alert,
  Anchor,
  Badge,
  Button,
  Card,
  Center,
  Container,
  Group,
  Loader,
  Stack,
  Text,
  ThemeIcon,
  Title,
} from '@mantine/core';
import { modals } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import {
  api,
  type ActivityAgent,
  type AgentKey,
  type ApiToken,
  type ApprovalRule,
  type ApprovalRules,
} from '@/api';
import { ApprovalRuleEditor } from '@/approvals/ApprovalRuleEditor';
import { formatDate } from '@/training/format';
import { BudgetEditor, MonthUsage } from '@/usage/BudgetEditor';
import { ActivityLink } from './ActivityLink';
import { ConnectedApps } from './ConnectedApps';
import { CreateTokenModal } from './CreateTokenModal';
import { McpSetup } from './McpSetup';
import classes from './Tokens.module.css';

function isExpired(token: ApiToken, now = Date.now()): boolean {
  return token.expires_at !== null && new Date(token.expires_at).getTime() <= now;
}

function TokenItem({
  token,
  activity,
  rule,
  onRevoke,
  onBudget,
  onRule,
}: {
  token: ApiToken;
  activity: ActivityAgent | undefined;
  /** What it may do without asking; undefined while loading (or on older servers). */
  rule: ApprovalRule | undefined;
  onRevoke: () => void;
  onBudget: (budget: number | null) => Promise<void>;
  onRule: (rule: ApprovalRule) => Promise<void>;
}) {
  const expired = isExpired(token);
  return (
    <li className={classes.item} aria-label={token.name}>
      <Group justify="space-between" align="flex-start" wrap="nowrap" gap="sm">
        <div className={classes.itemTitle}>
          <Group gap={8} wrap="wrap">
            <Text fw={600} className={classes.name}>
              {token.name}
            </Text>
            <Badge variant="light" color={token.read_only ? 'gray' : 'brick'}>
              {token.read_only ? 'Read only' : 'Full access'}
            </Badge>
            {expired && (
              <Badge variant="filled" color="red">
                Expired
              </Badge>
            )}
          </Group>
          <Text ff="monospace" size="sm" c="dimmed" mt={2} className={classes.name}>
            {token.prefix}…
          </Text>
        </div>
        <Button
          variant="light"
          color="red"
          size="xs"
          leftSection={<IconTrash size={14} />}
          className={classes.itemAction}
          onClick={onRevoke}
          aria-label={`Revoke “${token.name}”`}
        >
          Revoke
        </Button>
      </Group>
      <dl className={classes.meta}>
        <div>
          <dt>Created</dt>
          <dd>{formatDate(token.created_at)}</dd>
        </div>
        <div>
          <dt>Last used</dt>
          <dd>{token.last_used_at ? formatDate(token.last_used_at) : 'Never'}</dd>
        </div>
        <div>
          <dt>{expired ? 'Expired' : 'Expires'}</dt>
          <dd data-expired={expired || undefined}>
            {token.expires_at ? formatDate(token.expires_at) : 'Never'}
          </dd>
        </div>
        {token.used_cpu_minutes !== undefined && (
          <div>
            <dt>Compute this month</dt>
            <dd className={classes.budget}>
              <MonthUsage used={token.used_cpu_minutes} budget={token.budget_cpu_minutes} />
              <BudgetEditor
                budget={token.budget_cpu_minutes ?? null}
                label={`“${token.name}”`}
                onSave={onBudget}
                description="CPU minutes per month for everything this token starts. Empty: only your account's limits apply."
              />
            </dd>
          </div>
        )}
        {rule && (
          <div>
            <dt>Approval</dt>
            <dd>
              <ApprovalRuleEditor rule={rule} label={`“${token.name}”`} onSave={onRule} />
            </dd>
          </div>
        )}
        <div>
          <dt>Agent activity</dt>
          <dd>
            <ActivityLink agentKey={`token:${token.id}`} activity={activity} />
          </dd>
        </div>
      </dl>
    </li>
  );
}

/** What new tokens and apps may do without asking (the account's default rule). */
function DefaultApprovals({
  rules,
  onRule,
}: {
  rules: ApprovalRules;
  onRule: (rule: ApprovalRule) => Promise<void>;
}) {
  return (
    <section className={classes.section} aria-labelledby="approvals-title">
      <Title order={3} id="approvals-title" mb="xs">
        Approvals
      </Title>
      <Text size="sm" c="dimmed" mb="sm" maw={720}>
        Agents can ask you before expensive trainings, before publishing something (a download link
        that works without logging in, sharing data in a project) and before deleting. Their request
        then waits on{' '}
        <Anchor component={Link} to="/activity">
          the activity page
        </Anchor>{' '}
        until you approve or deny it (at most {rules.expiry_hours} hours). Set it for each token and
        app below.
      </Text>
      <Group gap="xs" wrap="wrap">
        <Text size="sm" fw={600}>
          New tokens and apps:
        </Text>
        <ApprovalRuleEditor rule={rules.default} label="new tokens and apps" onSave={onRule} />
      </Group>
    </section>
  );
}

export function ApiTokensPage() {
  const [tokens, setTokens] = useState<ApiToken[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  // A new key for each new token, so that the dialog starts with an empty form.
  const [dialogKey, setDialogKey] = useState(0);

  const load = useCallback(() => {
    api.apiTokens
      .list()
      .then(setTokens)
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(load, [load]);

  // What each token and app may do without asking (approvals).
  const [rules, setRules] = useState<ApprovalRules | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    api.approvals
      .rules(controller.signal)
      .then(setRules)
      .catch(() => undefined); // The page works without them (e.g. an older server).
    return () => controller.abort();
  }, []);
  const saveRule = async (
    agent: { kind: 'token' | 'app'; id: number } | null,
    rule: ApprovalRule
  ) => {
    const saved = await api.approvals.setRule(agent, rule);
    setRules((current) => {
      if (!current) {
        return current;
      }
      return agent
        ? { ...current, agents: { ...current.agents, [`${agent.kind}:${agent.id}`]: saved } }
        : { ...current, default: saved };
    });
  };

  // What each token and app did lately (the log is on the page "Agent activity").
  const [activity, setActivity] = useState<Map<AgentKey, ActivityAgent>>(new Map());
  useEffect(() => {
    const controller = new AbortController();
    api.agentActivity
      .summary(controller.signal)
      .then((summary) => setActivity(new Map(summary.agents.map((agent) => [agent.key, agent]))))
      .catch(() => undefined); // The links still work.
    return () => controller.abort();
  }, []);

  const create = () => {
    setDialogKey((key) => key + 1);
    setCreating(true);
  };

  const revoke = (token: ApiToken) =>
    modals.openConfirmModal({
      title: `Revoke “${token.name}”?`,
      children: (
        <Text size="sm">
          Apps, assistants and scripts using this token stop working immediately. This cannot be
          undone.
        </Text>
      ),
      labels: { confirm: 'Revoke', cancel: 'Cancel' },
      confirmProps: { color: 'red' },
      onConfirm: async () => {
        try {
          await api.apiTokens.revoke(token.id);
          setTokens((list) => list?.filter((t) => t.id !== token.id) ?? null);
        } catch (err) {
          notifications.show({
            color: 'red',
            title: 'Could not revoke the token',
            message: (err as Error).message,
          });
        }
      },
    });

  return (
    <Container size="md" py="xl">
      <Title order={2}>API tokens</Title>
      <Text mt="xs" maw={720}>
        AI assistants (through MCP) and your own scripts can use Brick DNN as you: apps like Claude
        by signing in, others with an API token. Treat tokens like passwords: anyone who has one can
        use your account as far as the token allows, until it expires or you revoke it. Everything
        they do is listed under{' '}
        <Anchor component={Link} to="/activity">
          Agent activity
        </Anchor>
        .
      </Text>

      <section className={classes.section} aria-labelledby="mcp-title">
        <Title order={3} id="mcp-title" mb="xs">
          Connect an AI assistant (MCP)
        </Title>
        <Text size="sm" c="dimmed" mb="md" maw={720}>
          With Brick DNN&apos;s MCP server, an assistant can look up layers and examples, check
          networks, generate PyTorch code and read your networks and training runs. With full
          access, it can also save networks and start or stop trainings. Give a token or app a
          monthly budget to limit the compute its trainings and predictions may use.
        </Text>
        <McpSetup />
      </section>

      {rules && <DefaultApprovals rules={rules} onRule={(rule) => saveRule(null, rule)} />}

      <ConnectedApps
        activity={activity}
        rules={rules?.agents}
        onRule={(id, rule) => saveRule({ kind: 'app', id }, rule)}
      />

      <section className={classes.section} aria-labelledby="tokens-title">
        <Group justify="space-between" mb="md">
          <Title order={3} id="tokens-title">
            Your tokens
          </Title>
          {tokens && tokens.length > 0 && (
            <Button leftSection={<IconPlus size={16} />} onClick={create}>
              Create token
            </Button>
          )}
        </Group>

        {error && <Alert color="red">{error}</Alert>}
        {!tokens && !error && (
          <Center h={120}>
            <Loader />
          </Center>
        )}
        {tokens?.length === 0 && (
          <Card withBorder p="xl">
            <Stack align="center" gap="sm" ta="center">
              <ThemeIcon variant="light" size={44} radius="xl" aria-hidden>
                <IconKey size={24} />
              </ThemeIcon>
              <Text c="dimmed">
                You have no API tokens yet. Create one to connect an AI assistant or a script.
              </Text>
              <Button leftSection={<IconPlus size={16} />} onClick={create}>
                Create token
              </Button>
            </Stack>
          </Card>
        )}
        {tokens && tokens.length > 0 && (
          <ul className={classes.list} aria-labelledby="tokens-title">
            {tokens.map((token) => (
              <TokenItem
                key={token.id}
                token={token}
                activity={activity.get(`token:${token.id}`)}
                rule={rules?.agents[`token:${token.id}`]}
                onRule={(rule) => saveRule({ kind: 'token', id: token.id }, rule)}
                onRevoke={() => revoke(token)}
                onBudget={async (budget) => {
                  const updated = await api.apiTokens.setBudget(token.id, budget);
                  setTokens((list) => list?.map((t) => (t.id === token.id ? updated : t)) ?? null);
                }}
              />
            ))}
          </ul>
        )}
      </section>

      <CreateTokenModal
        key={dialogKey}
        opened={creating}
        onClose={() => setCreating(false)}
        onCreated={(token) => {
          setTokens((list) => [token, ...(list ?? [])]);
          // A new token gets the default rule.
          setRules((current) =>
            current
              ? {
                  ...current,
                  agents: { ...current.agents, [`token:${token.id}`]: current.default },
                }
              : current
          );
        }}
      />
    </Container>
  );
}
