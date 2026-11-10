import { lazy, Suspense, useState } from 'react';
import { IconBulb } from '@tabler/icons-react';
import { Button } from '@mantine/core';
import type { RunDetail } from '@/api';

const ExplainModal = lazy(() =>
  import('./ExplainModal').then((module) => ({ default: module.ExplainModal }))
);

/**
 * "Explain": opens the explanations of a finished run with weights (saliency, Grad-CAM,
 * attributions, feature importance, decision boundary, activations, filters).
 */
export function ExplainButton({ run }: { run: RunDetail }) {
  const [opened, setOpened] = useState(false);
  const [loaded, setLoaded] = useState(false);
  return (
    <>
      <Button
        variant="light"
        color="grape"
        leftSection={<IconBulb size={16} />}
        onClick={() => {
          setLoaded(true);
          setOpened(true);
        }}
      >
        Explain
      </Button>
      {loaded && (
        <Suspense fallback={null}>
          <ExplainModal run={run} opened={opened} onClose={() => setOpened(false)} />
        </Suspense>
      )}
    </>
  );
}
