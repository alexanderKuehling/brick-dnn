import { Avatar, Tooltip, UnstyledButton } from '@mantine/core';
import { userColor } from '@/live/presence';
import classes from './Sharing.module.css';

interface PeopleAvatarsProps {
  /** Usernames, the owner first. */
  names: string[];
  /**
   * Live editing: the usernames of the people who have the network open right now. They come
   * first, in their colour (the one of their selection on the canvas); the others are dimmed.
   */
  here?: string[] | null;
  /** Avatars shown at most (including the "+n" one). */
  max?: number;
  onClick?: () => void;
}

/** Initials of the people who have access to a shared network; opens the share dialog. */
export function PeopleAvatars({ names, here = null, max = 3, onClick }: PeopleAvatarsProps) {
  const present = new Set(here ?? []);
  const owner = names[0];
  const ordered = here
    ? [...names.filter((name) => present.has(name)), ...names.filter((name) => !present.has(name))]
    : names;
  const shown = ordered.length > max ? ordered.slice(0, max - 1) : ordered;
  const rest = ordered.slice(shown.length);
  let label =
    names.length > 1 ? `Shared: ${owner} (owner), ${names.slice(1).join(', ')}` : 'Shared';
  if (here && present.size > 0) {
    label += `. Here now: ${names.filter((name) => present.has(name)).join(', ')}`;
  }
  const describe = (name: string) =>
    `${name}${name === owner ? ' (owner)' : ''}${present.has(name) ? ' · here now' : ''}`;

  return (
    <UnstyledButton
      onClick={onClick}
      aria-label={`${label}. Manage sharing`}
      style={{ flex: 'none' }}
    >
      <Avatar.Group spacing={4}>
        {shown.map((name) => {
          const isHere = present.has(name);
          return (
            <Tooltip key={name} label={describe(name)}>
              <Avatar
                name={name}
                color={here ? (isHere ? userColor(name) : 'gray') : 'initials'}
                variant={here && isHere ? 'filled' : 'light'}
                size={28}
                className={here ? classes.liveAvatar : undefined}
                data-here={(here && isHere) || undefined}
              />
            </Tooltip>
          );
        })}
        {rest.length > 0 && (
          <Tooltip label={rest.map(describe).join(', ')} multiline maw={280}>
            <Avatar size={28}>+{rest.length}</Avatar>
          </Tooltip>
        )}
      </Avatar.Group>
    </UnstyledButton>
  );
}
