import { Link } from 'react-router-dom';
import { Anchor, Text } from '@mantine/core';
import type { ActivityAgent, AgentKey } from '@/api';
import { relativeTime } from '@/collab/time';
import { currentLocale, i18n, useTranslation } from '@/i18n';

/** "12 actions · 5 min ago" linking to the token's or app's entries of the activity log. */
export function ActivityLink({
  agentKey,
  activity,
}: {
  agentKey: AgentKey;
  activity: ActivityAgent | undefined;
}) {
  useTranslation();
  if (!activity) {
    return (
      <Text span size="sm">
        {i18n.t('ui.tokens.ActivityLink.noneYet')}
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
          {i18n.t('ui.tokens.ActivityLink.failed', {
            errors: activity.errors.toLocaleString(currentLocale()),
          })}
        </Text>
      )}
    </Anchor>
  );
}
