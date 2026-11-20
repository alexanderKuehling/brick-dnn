import { lazy, Suspense, useState } from 'react';
import { IconCloudUpload } from '@tabler/icons-react';
import { Button } from '@mantine/core';
import type { HubCapabilities, RunSummary } from '@/api';
import { i18n, useTranslation } from '@/i18n';

const PublishDialog = lazy(() =>
  import('./PublishDialog').then((module) => ({ default: module.PublishDialog }))
);

/**
 * "Publish to Hugging Face" for a finished run with weights. The dialog is loaded the first
 * time it opens and mounted anew for every opening, so the access token typed into it is gone
 * as soon as it closes.
 */
export function PublishButton({
  run,
  hub,
}: {
  run: Pick<RunSummary, 'id' | 'name'>;
  hub: HubCapabilities;
}) {
  useTranslation();
  const [opening, setOpening] = useState(0);
  const [opened, setOpened] = useState(false);
  return (
    <>
      <Button
        variant="default"
        leftSection={<IconCloudUpload size={16} />}
        onClick={() => {
          setOpening((count) => count + 1);
          setOpened(true);
        }}
      >
        {i18n.t('huggingface.publish.button')}
      </Button>
      {opening > 0 && (
        <Suspense fallback={null}>
          <PublishDialog
            key={opening}
            opened={opened}
            onClose={() => setOpened(false)}
            run={run}
            hub={hub}
          />
        </Suspense>
      )}
    </>
  );
}
