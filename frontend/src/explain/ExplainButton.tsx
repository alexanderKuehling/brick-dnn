import { lazy, Suspense, useState } from 'react';
import { IconBulb } from '@tabler/icons-react';
import { Button, Group, Text, type ButtonProps } from '@mantine/core';
import type { RunDetail } from '@/api';
import type { ExplainStart } from './ExplainModal';

const ExplainModal = lazy(() =>
  import('./ExplainModal').then((module) => ({ default: module.ExplainModal }))
);

/**
 * The Explain dialog, loaded the first time it opens. `start` null: closed; otherwise open
 * on that tab (and sample).
 */
export function ExplainDialog({
  run,
  start,
  onClose,
}: {
  run: RunDetail;
  start: ExplainStart | null;
  onClose: () => void;
}) {
  const [loaded, setLoaded] = useState(false);
  if (start && !loaded) {
    setLoaded(true);
  }
  if (!loaded) {
    return null;
  }
  return (
    <Suspense fallback={null}>
      <ExplainModal run={run} opened={start !== null} onClose={onClose} {...(start ?? {})} />
    </Suspense>
  );
}

/**
 * "Explain the model": opens the explanations of a finished run with weights (saliency,
 * Grad-CAM, attributions, feature importance, decision boundary, activations, filters).
 */
export function ExplainButton({
  run,
  size = 'sm',
  start = {},
}: {
  run: RunDetail;
  size?: ButtonProps['size'];
  /** The tab to open on (default: the first that works). */
  start?: ExplainStart;
}) {
  const [opened, setOpened] = useState(false);
  return (
    <>
      <Button
        variant="light"
        color="grape"
        size={size}
        leftSection={<IconBulb size={16} />}
        onClick={() => setOpened(true)}
      >
        Explain the model
      </Button>
      <ExplainDialog run={run} start={opened ? start : null} onClose={() => setOpened(false)} />
    </>
  );
}

/** The way into the explanations, under the headline of a finished run. */
export function ExplainHeadline({ run }: { run: RunDetail }) {
  return (
    <Group gap="xs" mt="xs" wrap="wrap">
      <ExplainButton run={run} size="xs" />
      <Text size="xs" c="dimmed">
        Why it decides as it does: what matters, where it looks, what happens inside.
      </Text>
    </Group>
  );
}
