import { Badge, Loader, Tooltip } from '@mantine/core';
import type { LiveIndicator } from './useLiveBuilder';
import classes from './Live.module.css';

const BADGES: Record<LiveIndicator, { color: string; label: string; more?: string; tip: string }> =
  {
    live: {
      color: 'green',
      label: 'Live',
      tip: 'Your changes are saved as you make them. Everyone with access sees them right away.',
    },
    saving: { color: 'blue', label: 'Saving…', tip: 'Sending your changes to the server…' },
    offline: {
      color: 'yellow',
      label: 'Offline',
      more: ' – changes will sync',
      tip: 'No connection to the server. Your changes are kept and sent when it is back.',
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
        {/* Phones show the dot only (the offline state keeps its word). */}
        <span className={indicator === 'offline' ? undefined : classes.label}>{badge.label}</span>
        {badge.more && <span className={classes.more}>{badge.more}</span>}
      </Badge>
    </Tooltip>
  );
}
