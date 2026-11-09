import { IconCoffee } from '@tabler/icons-react';
import { Button, Text } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { SUPPORT_URL } from '@/links';

/** Set once the hint was shown in this browser. */
export const SUPPORT_HINT_KEY = 'brickdnn:support-hint';

function alreadyShown(): boolean {
  try {
    return localStorage.getItem(SUPPORT_HINT_KEY) !== null;
  } catch {
    return true; // Storage unavailable: better never than on every training.
  }
}

/**
 * After the user's first successful training (in this browser), ask once for a coffee.
 * Returns whether the hint was shown.
 */
export function showSupportHintOnce(): boolean {
  if (alreadyShown()) {
    return false;
  }
  try {
    localStorage.setItem(SUPPORT_HINT_KEY, new Date().toISOString());
  } catch {
    return false;
  }
  notifications.show({
    id: 'support-hint',
    color: 'yellow',
    icon: <IconCoffee size={18} />,
    title: 'Your first training is done!',
    autoClose: 30000,
    message: (
      <>
        <Text size="sm">
          Brick DNN is free and open source. If it helps you, you can support its development with a
          coffee.
        </Text>
        <Button
          component="a"
          href={SUPPORT_URL}
          target="_blank"
          rel="noreferrer"
          size="compact-xs"
          variant="light"
          color="yellow"
          mt={6}
          leftSection={<IconCoffee size={14} />}
          onClick={() => notifications.hide('support-hint')}
        >
          Buy me a coffee
        </Button>
      </>
    ),
  });
  return true;
}
