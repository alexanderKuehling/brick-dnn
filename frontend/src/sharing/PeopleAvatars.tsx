import { Avatar, Tooltip, UnstyledButton } from '@mantine/core';

interface PeopleAvatarsProps {
  /** Usernames, the owner first. */
  names: string[];
  /** Avatars shown at most (including the "+n" one). */
  max?: number;
  onClick?: () => void;
}

/** Initials of the people who have access to a shared network; opens the share dialog. */
export function PeopleAvatars({ names, max = 3, onClick }: PeopleAvatarsProps) {
  const shown = names.length > max ? names.slice(0, max - 1) : names;
  const rest = names.length - shown.length;
  const label =
    names.length > 1 ? `Shared: ${names[0]} (owner), ${names.slice(1).join(', ')}` : 'Shared';
  return (
    <Tooltip label={label} multiline maw={280}>
      <UnstyledButton
        onClick={onClick}
        aria-label={`${label}. Manage sharing`}
        style={{ flex: 'none' }}
      >
        <Avatar.Group spacing={4}>
          {shown.map((name) => (
            <Avatar key={name} name={name} color="initials" size={28} />
          ))}
          {rest > 0 && <Avatar size={28}>+{rest}</Avatar>}
        </Avatar.Group>
      </UnstyledButton>
    </Tooltip>
  );
}
