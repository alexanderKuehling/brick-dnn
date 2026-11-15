import { IconCoffee } from '@tabler/icons-react';
import { Button, Text } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { i18n } from '@/i18n';
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
    title: i18n.t('ui.training.supportHint.yourFirstTrainingIsDone'),
    autoClose: 30000,
    message: (
      <>
        <Text size="sm">{i18n.t('ui.training.supportHint.brickDnnIsFreeAnd')}</Text>
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
          {i18n.t('ui.training.supportHint.buyMeACoffee')}
        </Button>
      </>
    ),
  });
  return true;
}
