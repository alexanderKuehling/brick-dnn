import { Badge, Loader, Tooltip } from '@mantine/core';
import { i18n } from '@/i18n';
import type { LiveIndicator } from './useLiveBuilder';
import classes from './Live.module.css';

const BADGES: Record<LiveIndicator, { color: string; label: string; more?: string; tip: string }> =
  {
    live: {
      color: 'green',
      get label() {
        return i18n.t('ui.live.LiveBadge.live');
      },
      get tip() {
        return i18n.t('ui.live.LiveBadge.yourChangesAreSavedAs');
      },
    },
    saving: {
      color: 'blue',
      get label() {
        return i18n.t('ui.live.LiveBadge.saving');
      },
      get tip() {
        return i18n.t('ui.live.LiveBadge.sendingYourChangesToThe');
      },
    },
    offline: {
      color: 'yellow',
      get label() {
        return i18n.t('ui.live.LiveBadge.offline');
      },
      get more() {
        return i18n.t('ui.live.LiveBadge.changesWillSync');
      },
      get tip() {
        return i18n.t('ui.live.LiveBadge.noConnectionToTheServer');
      },
    },
  };

/** Shown instead of the save button while the network is edited live. */
export function LiveBadge({ indicator }: { indicator: LiveIndicator }) {
  const badge = BADGES[indicator];
  return (
    <Tooltip label={badge.tip} multiline w={260}>
      <Badge
        role="status"
        color={badge.color}
        variant="light"
        tt="none"
        size="lg"
        leftSection={
          indicator === 'saving' ? (
            <Loader size={10} color="blue" />
          ) : (
            <span className={classes.dot} />
          )
        }
        style={{ flexShrink: 0 }}
        aria-label={`${badge.label}${badge.more ?? ''}`}
      >
        {/* Phones show the dot only (green, a spinner, yellow): the toolbar is full there. */}
        <span className={classes.label}>{badge.label}</span>
        {badge.more && <span className={classes.more}>{badge.more}</span>}
      </Badge>
    </Tooltip>
  );
}
