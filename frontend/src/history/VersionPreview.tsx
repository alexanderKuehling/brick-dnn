import { useEffect, useState } from 'react';
import { IconAlertTriangle, IconRestore } from '@tabler/icons-react';
import {
  Alert,
  Badge,
  Box,
  Button,
  Center,
  Group,
  Loader,
  Modal,
  Paper,
  Stack,
  Text,
  useMantineTheme,
} from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import { api, type ProjectVersion, type ProjectVersionDetail } from '@/api';
import { NetworkViewer } from '@/builder/NetworkViewer';
import { WithoutCollab } from '@/collab/context';
import { fullDateTime, relativeTime } from '@/collab/time';
import { layerCount } from './labels';
import classes from './History.module.css';

interface VersionPreviewProps {
  projectId: number;
  entry: ProjectVersion;
  /** What the history calls it, e.g. "Baseline" or "Edited by bob". */
  title: string;
  /** It holds the current state (nothing to restore). */
  current: boolean;
  /** Owner and editors. */
  canRestore: boolean;
  /** The canvas has changes that are not saved: restoring replaces them. */
  unsaved?: boolean;
  /** Show the confirmation right away ("Restore…" in the list). */
  confirmFirst?: boolean;
  /** Makes it the current state; the preview closes when it resolves. */
  onRestore: () => Promise<void>;
  onClose: () => void;
}

/** An older version of the network, read only, with "Restore this version" for editors. */
export function VersionPreview({
  projectId,
  entry,
  title,
  current,
  canRestore,
  unsaved = false,
  confirmFirst = false,
  onRestore,
  onClose,
}: VersionPreviewProps) {
  const theme = useMantineTheme();
  const phone = useMediaQuery(`(max-width: ${theme.breakpoints.sm})`);
  const [detail, setDetail] = useState<ProjectVersionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [confirming, setConfirming] = useState(confirmFirst && canRestore && !current);
  const [restoring, setRestoring] = useState(false);
  const [restoreError, setRestoreError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setError(null);
    api.projects.versions
      .get(projectId, entry.id, controller.signal)
      .then(setDetail)
      .catch((err: Error) => !controller.signal.aborted && setError(err.message));
    return () => controller.abort();
  }, [projectId, entry.id, attempt]);

  const restore = async () => {
    setRestoring(true);
    setRestoreError(null);
    try {
      await onRestore();
    } catch (err) {
      setRestoreError((err as Error).message);
      setRestoring(false);
    }
  };

  const meta = [
    entry.created_by,
    relativeTime(entry.updated_at),
    layerCount(entry.num_layers),
    entry.name !== title ? `“${entry.name}”` : null,
  ].filter(Boolean);

  return (
    <Modal
      opened
      onClose={onClose}
      fullScreen={phone}
      size="calc(100vw - 4rem)"
      padding={0}
      title={
        <Stack gap={0}>
          <Group gap="xs">
            <Text fw={700}>{title}</Text>
            {current && (
              <Badge size="sm" variant="light" tt="none">
                Current version
              </Badge>
            )}
          </Group>
          <Text size="xs" c="dimmed" title={fullDateTime(entry.updated_at)}>
            {meta.join(' · ')}
          </Text>
        </Stack>
      }
      classNames={{ header: classes.previewHeader, body: classes.previewBody }}
    >
      <div className={classes.previewBar}>
        {current ? (
          <Text size="sm" c="dimmed">
            This is the current state of the network.
          </Text>
        ) : canRestore ? (
          !confirming && (
            <>
              <Text size="sm" c="dimmed" className={classes.previewHint}>
                A read-only preview. Restoring makes it the current state.
              </Text>
              <Button
                leftSection={<IconRestore size={16} />}
                onClick={() => setConfirming(true)}
                disabled={!detail}
              >
                Restore this version
              </Button>
            </>
          )
        ) : (
          <Text size="sm" c="dimmed">
            A read-only preview. The owner and editors can restore earlier versions.
          </Text>
        )}
        {confirming && (
          <Paper withBorder p="sm" radius="md" className={classes.confirm}>
            <Text fw={600} size="sm">
              Restore “{title}”?
            </Text>
            <Text size="sm" mt={4}>
              It becomes the current state of the network, for everyone who works on it. Nothing is
              lost: the current state stays in the history, so you can undo this by restoring it.
            </Text>
            {unsaved && (
              <Text size="sm" mt={4} c="orange">
                <IconAlertTriangle size={14} style={{ verticalAlign: 'text-bottom' }} /> Your
                unsaved changes on the canvas are replaced.
              </Text>
            )}
            {restoreError && (
              <Text size="sm" c="red" mt={4}>
                {restoreError}
              </Text>
            )}
            <Group justify="flex-end" gap="xs" mt="sm">
              <Button variant="default" onClick={() => setConfirming(false)} disabled={restoring}>
                Cancel
              </Button>
              <Button
                leftSection={<IconRestore size={16} />}
                loading={restoring}
                disabled={!detail}
                onClick={() => void restore()}
              >
                Restore
              </Button>
            </Group>
          </Paper>
        )}
      </div>
      <Box className={classes.previewCanvas}>
        {detail ? (
          <WithoutCollab>
            <NetworkViewer doc={detail.graph} />
          </WithoutCollab>
        ) : error ? (
          <Center style={{ flex: 1 }} p="md">
            <Alert color="red" variant="light" title="Could not load this version" maw={420}>
              {error}{' '}
              <Button variant="subtle" size="compact-xs" onClick={() => setAttempt((n) => n + 1)}>
                Try again
              </Button>
            </Alert>
          </Center>
        ) : (
          <Center style={{ flex: 1 }}>
            <Loader />
          </Center>
        )}
      </Box>
    </Modal>
  );
}
