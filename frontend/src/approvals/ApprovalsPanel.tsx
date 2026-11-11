import { useState } from 'react';
import {
  IconAlertTriangle,
  IconCheck,
  IconClockHour4,
  IconEye,
  IconKey,
  IconPlayerPlay,
  IconPlugConnected,
  IconShare,
  IconTrash,
  IconX,
} from '@tabler/icons-react';
import {
  Anchor,
  Badge,
  Button,
  Group,
  Paper,
  Stack,
  Text,
  Textarea,
  ThemeIcon,
  Title,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import type { OpenTarget } from '@/activity/ActivityItem';
import { summaryParts, TARGET_LABELS, unnamedTargets } from '@/activity/format';
import { api, type AgentTarget, type ApprovalCategory, type ApprovalRequest } from '@/api';
import { relativeTime } from '@/collab/time';
import { formatCpu } from '@/usage/format';
import { estimateText, expiresIn } from './format';
import classes from './Approvals.module.css';

const CATEGORY_ICONS: Record<ApprovalCategory, typeof IconShare> = {
  training: IconPlayerPlay,
  publishing: IconShare,
  deleting: IconTrash,
};

const CATEGORY_LABELS: Record<ApprovalCategory, string> = {
  training: 'Training',
  publishing: 'Publishing',
  deleting: 'Deleting',
};

function TargetLink({ target, onOpen }: { target: AgentTarget; onOpen: OpenTarget }) {
  return (
    <Anchor
      component="button"
      type="button"
      className={classes.link}
      onClick={() => onOpen(target)}
      aria-label={`Open ${TARGET_LABELS[target.kind].toLowerCase()} “${target.name}”`}
    >
      “{target.name}”
    </Anchor>
  );
}

function Summary({ approval, onOpen }: { approval: ApprovalRequest; onOpen: OpenTarget }) {
  const extra = unnamedTargets(approval.summary, approval.targets);
  return (
    <Text fw={600} className={classes.summary}>
      {summaryParts(approval.summary, approval.targets).map((part, index) =>
        typeof part === 'string' ? (
          part
        ) : (
          <TargetLink key={index} target={part.target} onOpen={onOpen} />
        )
      )}
      {extra.map((target) => (
        <span key={`${target.kind}:${target.id}`}>
          {' · '}
          {TARGET_LABELS[target.kind]} <TargetLink target={target} onOpen={onOpen} />
        </span>
      ))}
    </Text>
  );
}

function Facts({ approval }: { approval: ApprovalRequest }) {
  const rows: [string, string][] = [];
  if (approval.category === 'training') {
    rows.push(['Estimated CPU time', estimateText(approval.estimate)]);
    if (approval.limit_cpu_minutes !== null) {
      rows.push([
        'Allowed without asking',
        approval.limit_cpu_minutes === 0
          ? 'none'
          : `up to ${formatCpu(approval.limit_cpu_minutes)}`,
      ]);
    }
    if (approval.network) {
      rows.push(['Network', approval.network]);
    }
    if (approval.dataset) {
      rows.push(['Data', approval.dataset]);
    }
  }
  if (rows.length === 0) {
    return null;
  }
  return (
    <dl className={classes.facts}>
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function ApprovalCard({
  approval,
  now,
  onDecided,
  onOpen,
}: {
  approval: ApprovalRequest;
  now: Date;
  onDecided: (approval: ApprovalRequest) => void;
  onOpen: OpenTarget;
}) {
  const [busy, setBusy] = useState<'approve' | 'deny' | null>(null);
  const [denying, setDenying] = useState(false);
  const [message, setMessage] = useState('');
  const CategoryIcon = CATEGORY_ICONS[approval.category];
  const AgentIcon = approval.agent.kind === 'app' ? IconPlugConnected : IconKey;
  const titleId = `approval-${approval.id}`;

  const decide = async (what: 'approve' | 'deny') => {
    setBusy(what);
    try {
      const decided =
        what === 'approve'
          ? await api.approvals.approve(approval.id)
          : await api.approvals.deny(approval.id, message.trim());
      onDecided(decided);
      if (decided.status === 'executed') {
        notifications.show({ color: 'green', title: 'Approved', message: decided.done });
      } else if (decided.status === 'failed') {
        notifications.show({
          color: 'red',
          title: 'Approved, but it could not be done',
          message: decided.error,
        });
      } else if (decided.status === 'denied') {
        notifications.show({ message: 'Denied. The agent sees your answer.' });
      }
    } catch (err) {
      notifications.show({
        color: 'red',
        title: what === 'approve' ? 'Could not approve' : 'Could not deny',
        message: (err as Error).message,
      });
      setBusy(null);
    }
  };

  return (
    <Paper component="li" withBorder p="md" className={classes.card} aria-labelledby={titleId}>
      <Group gap="sm" align="flex-start" wrap="nowrap">
        <ThemeIcon
          variant="light"
          color={approval.category === 'deleting' ? 'red' : 'yellow'}
          radius="xl"
          size={34}
          aria-hidden
        >
          <CategoryIcon size={18} />
        </ThemeIcon>
        <div className={classes.main} id={titleId}>
          <Summary approval={approval} onOpen={onOpen} />
          <Group gap={8} mt={4} wrap="wrap">
            <Badge
              variant="light"
              color={approval.agent.kind === 'app' ? 'teal' : 'gray'}
              radius="sm"
              leftSection={<AgentIcon size={12} />}
              className={classes.agent}
            >
              {approval.agent.name}
            </Badge>
            <Badge variant="outline" color="gray" radius="sm">
              {CATEGORY_LABELS[approval.category]}
            </Badge>
            <Text span size="xs" c="dimmed">
              asked {relativeTime(approval.created_at, now)} · expires{' '}
              {expiresIn(approval.expires_at, now)}
            </Text>
          </Group>
        </div>
      </Group>

      <Stack gap={6} mt="sm" className={classes.body}>
        {approval.reason && (
          <Text size="sm" c="dimmed">
            {approval.reason}
          </Text>
        )}
        <Facts approval={approval} />
        {approval.exposes && (
          <Group gap={6} wrap="nowrap" align="flex-start" className={classes.exposes}>
            <IconEye size={16} aria-hidden className={classes.inlineIcon} />
            <Text size="sm">{approval.exposes}</Text>
          </Group>
        )}
        {approval.warning && (
          <Group gap={6} wrap="nowrap" align="flex-start">
            <IconAlertTriangle
              size={16}
              aria-hidden
              className={classes.inlineIcon}
              color="var(--mantine-color-orange-6)"
            />
            <Text size="sm">{approval.warning}</Text>
          </Group>
        )}
        {!approval.agent.active && (
          <Text size="sm" c="red">
            This {approval.agent.kind === 'app' ? 'app was disconnected' : 'token was revoked'}:
            approving it does nothing.
          </Text>
        )}
      </Stack>

      {denying && (
        <Textarea
          mt="sm"
          label="Message for the agent (optional)"
          placeholder="e.g. Use 5 epochs and the smaller dataset."
          autosize
          minRows={2}
          maxLength={500}
          value={message}
          onChange={(event) => setMessage(event.currentTarget.value)}
          data-autofocus
        />
      )}
      <Group gap="xs" mt="sm" justify="flex-end" wrap="wrap">
        {denying ? (
          <>
            <Button
              variant="default"
              size="xs"
              onClick={() => setDenying(false)}
              disabled={busy !== null}
            >
              Cancel
            </Button>
            <Button
              color="red"
              size="xs"
              leftSection={<IconX size={14} />}
              loading={busy === 'deny'}
              disabled={busy === 'approve'}
              onClick={() => void decide('deny')}
            >
              Deny
            </Button>
          </>
        ) : (
          <>
            <Button
              variant="default"
              size="xs"
              leftSection={<IconX size={14} />}
              onClick={() => setDenying(true)}
              disabled={busy !== null}
              aria-label={`Deny: ${approval.summary}`}
            >
              Deny…
            </Button>
            <Button
              color="green"
              size="xs"
              leftSection={<IconCheck size={14} />}
              loading={busy === 'approve'}
              onClick={() => void decide('approve')}
              aria-label={`Approve: ${approval.summary}`}
            >
              Approve
            </Button>
          </>
        )}
      </Group>
    </Paper>
  );
}

/**
 * "Waiting for your approval": what agents asked to do, with Approve and Deny. `onDecided`
 * gets each request once it is decided (it then leaves the list).
 */
export function ApprovalsPanel({
  approvals,
  now = new Date(),
  onDecided,
  onOpen,
}: {
  approvals: ApprovalRequest[];
  now?: Date;
  onDecided: (approval: ApprovalRequest) => void;
  onOpen: OpenTarget;
}) {
  if (approvals.length === 0) {
    return null;
  }
  return (
    <section className={classes.panel} aria-labelledby="approvals-title">
      <Group gap="xs" mb="xs">
        <IconClockHour4 size={20} aria-hidden color="var(--mantine-color-yellow-7)" />
        <Title order={3} size="h4" id="approvals-title">
          Waiting for your approval
        </Title>
        <Badge color="yellow" variant="filled" circle aria-label={`${approvals.length} waiting`}>
          {approvals.length}
        </Badge>
      </Group>
      <Text size="sm" c="dimmed" mb="sm">
        Approving does it right away, as the agent and within its budgets. Denying tells the agent
        your reason.
      </Text>
      <ul className={classes.list}>
        {approvals.map((approval) => (
          <ApprovalCard
            key={approval.id}
            approval={approval}
            now={now}
            onDecided={onDecided}
            onOpen={onOpen}
          />
        ))}
      </ul>
    </section>
  );
}
