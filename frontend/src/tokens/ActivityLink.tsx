import { Link } from 'react-router-dom';
import { Anchor, Text } from '@mantine/core';
import type { ActivityAgent, AgentKey } from '@/api';
import { relativeTime } from '@/collab/time';
import { currentLocale } from '@/i18n';

/** "12 actions · 5 min ago" linking to the token's or app's entries of the activity log. */
export function ActivityLink({
  agentKey,
  activity,
}: {
  agentKey: AgentKey;
  activity: ActivityAgent | undefined;
}) {
  if (!activity) {
    return (
      <Text span size="sm">
        None yet
      </Text>
    );
  }
  const actions = `${activity.actions.toLocaleString(currentLocale())} ${
    activity.actions === 1 ? 'action' : 'actions'
  }`;
  return (
    <Anchor component={Link} to={`/activity?agent=${agentKey}`} size="sm">
      {actions} · {relativeTime(activity.last_at)}
      {activity.errors > 0 && (
        <Text span c="red" size="sm">
          {' '}
          · {activity.errors.toLocaleString(currentLocale())} failed
        </Text>
      )}
    </Anchor>
  );
}
