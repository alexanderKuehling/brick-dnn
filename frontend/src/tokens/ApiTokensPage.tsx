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
import { i18n, Trans, useTranslation } from '@/i18n';
import { formatDate } from '@/training/format';
import { BudgetEditor, MonthUsage } from '@/usage/BudgetEditor';
import { ActivityLink } from './ActivityLink';
import { ConnectedApps } from './ConnectedApps';
import { CreateTokenModal } from './CreateTokenModal';
import { LoggingSetup } from './LoggingSetup';
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
  useTranslation();
  const expired = isExpired(token);
  return (
    <li className={classes.item} aria-label={token.name}>
      <Group justify="space-between" align="flex-start" wrap="nowrap" gap="sm">
        <div className={classes.itemTitle}>
          <Group gap={8} wrap="wrap">
            <Text fw={600} className={classes.name}>
              {token.name}
            </Text>
            <Badge
              variant="light"
              color={token.scope === 'runs:log' ? 'teal' : token.read_only ? 'gray' : 'brick'}
            >
              {token.scope === 'runs:log'
                ? i18n.t('experiments.logging.scopeBadge')
                : token.read_only
                  ? i18n.t('ui.tokens.ApiTokensPage.readOnly')
                  : i18n.t('ui.tokens.ApiTokensPage.fullAccess')}
            </Badge>
            {expired && (
              <Badge variant="filled" color="red">
                {i18n.t('ui.tokens.ApiTokensPage.expired')}
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
          aria-label={i18n.t('ui.tokens.ApiTokensPage.revoke3', { name: token.name })}
        >
          {i18n.t('ui.tokens.ApiTokensPage.revoke')}
        </Button>
      </Group>
      <dl className={classes.meta}>
        <div>
          <dt>{i18n.t('ui.tokens.ApiTokensPage.created')}</dt>
          <dd>{formatDate(token.created_at)}</dd>
        </div>
        <div>
          <dt>{i18n.t('ui.tokens.ApiTokensPage.lastUsed')}</dt>
          <dd>
            {token.last_used_at
              ? formatDate(token.last_used_at)
              : i18n.t('ui.tokens.ApiTokensPage.never')}
          </dd>
        </div>
        <div>
          <dt>
            {expired
              ? i18n.t('ui.tokens.ApiTokensPage.expired')
              : i18n.t('ui.tokens.ApiTokensPage.expires')}
          </dt>
          <dd data-expired={expired || undefined}>
            {token.expires_at
              ? formatDate(token.expires_at)
              : i18n.t('ui.tokens.ApiTokensPage.never')}
          </dd>
        </div>
        {token.used_cpu_minutes !== undefined && (
          <div>
            <dt>{i18n.t('ui.tokens.ApiTokensPage.computeThisMonth')}</dt>
            <dd className={classes.budget}>
              <MonthUsage used={token.used_cpu_minutes} budget={token.budget_cpu_minutes} />
              <BudgetEditor
                budget={token.budget_cpu_minutes ?? null}
                label={`“${token.name}”`}
                onSave={onBudget}
                description={i18n.t('ui.tokens.ApiTokensPage.cpuMinutesPerMonthFor')}
              />
            </dd>
          </div>
        )}
        {rule && (
          <div>
            <dt>{i18n.t('ui.tokens.ApiTokensPage.approval')}</dt>
            <dd>
              <ApprovalRuleEditor rule={rule} label={`“${token.name}”`} onSave={onRule} />
            </dd>
          </div>
        )}
        <div>
          <dt>{i18n.t('ui.tokens.ApiTokensPage.agentActivity')}</dt>
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
  useTranslation();
  return (
    <section className={classes.section} aria-labelledby="approvals-title">
      <Title order={3} id="approvals-title" mb="xs">
        {i18n.t('ui.tokens.ApiTokensPage.approvals')}
      </Title>
      <Text size="sm" c="dimmed" mb="sm" maw={720}>
        <Trans
          i18nKey="ui.tokens.ApiTokensPage.agentsCanAskYouBefore"
          values={{ expiry_hours: rules.expiry_hours }}
          components={[<Anchor component={Link} to="/activity" />]}
        />
      </Text>
      <Group gap="xs" wrap="wrap">
        <Text size="sm" fw={600}>
          {i18n.t('ui.tokens.ApiTokensPage.newTokensAndApps')}
        </Text>
        <ApprovalRuleEditor
          rule={rules.default}
          label={i18n.t('ui.tokens.ApiTokensPage.newTokensAndApps2')}
          onSave={onRule}
        />
      </Group>
    </section>
  );
}

export function ApiTokensPage() {
  useTranslation();
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
      title: i18n.t('ui.tokens.ApiTokensPage.revoke2', { name: token.name }),
      children: (
        <Text size="sm">{i18n.t('ui.tokens.ApiTokensPage.appsAssistantsAndScriptsUsing')}</Text>
      ),
      labels: {
        confirm: i18n.t('ui.tokens.ApiTokensPage.revoke'),
        cancel: i18n.t('ui.tokens.ApiTokensPage.cancel'),
      },
      confirmProps: { color: 'red' },
      onConfirm: async () => {
        try {
          await api.apiTokens.revoke(token.id);
          setTokens((list) => list?.filter((t) => t.id !== token.id) ?? null);
        } catch (err) {
          notifications.show({
            color: 'red',
            title: i18n.t('ui.tokens.ApiTokensPage.couldNotRevokeTheToken'),
            message: (err as Error).message,
          });
        }
      },
    });

  return (
    <Container size="md" py="xl">
      <Title order={2}>{i18n.t('ui.tokens.ApiTokensPage.apiTokens')}</Title>
      <Text mt="xs" maw={720}>
        <Trans
          i18nKey="ui.tokens.ApiTokensPage.aiAssistantsThroughMcpAnd"
          components={[<Anchor component={Link} to="/activity" />]}
        />
      </Text>

      <section className={classes.section} aria-labelledby="mcp-title">
        <Title order={3} id="mcp-title" mb="xs">
          {i18n.t('ui.tokens.ApiTokensPage.connectAnAiAssistantMcp')}
        </Title>
        <Text size="sm" c="dimmed" mb="md" maw={720}>
          {i18n.t('ui.tokens.ApiTokensPage.withBrickDnnSMcp')}
        </Text>
        <McpSetup />
      </section>

      <section className={classes.section} aria-labelledby="logging-title">
        <Title order={3} id="logging-title" mb="xs">
          {i18n.t('experiments.logging.title')}
        </Title>
        <Text size="sm" c="dimmed" mb="md" maw={720}>
          {i18n.t('experiments.logging.intro')}
        </Text>
        <LoggingSetup />
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
            {i18n.t('ui.tokens.ApiTokensPage.yourTokens')}
          </Title>
          {tokens && tokens.length > 0 && (
            <Button leftSection={<IconPlus size={16} />} onClick={create}>
              {i18n.t('ui.tokens.ApiTokensPage.createToken')}
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
              <Text c="dimmed">{i18n.t('ui.tokens.ApiTokensPage.youHaveNoApiTokens')}</Text>
              <Button leftSection={<IconPlus size={16} />} onClick={create}>
                {i18n.t('ui.tokens.ApiTokensPage.createToken')}
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
