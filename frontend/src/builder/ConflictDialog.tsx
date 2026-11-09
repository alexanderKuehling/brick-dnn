import { Button, Flex, List, Modal, Text } from '@mantine/core';
import type { Project } from '@/api';
import { formatDate } from '@/training/format';

export type ConflictChoice = 'theirs' | 'mine' | 'copy';

interface ConflictDialogProps {
  /** The current state on the server (from the 409 answer); null hides the dialog. */
  project: Project | null;
  onChoose: (choice: ConflictChoice) => void;
  onClose: () => void;
}

/** A save found that someone else changed the network in the meantime. */
export function ConflictDialog({ project, onChoose, onClose }: ConflictDialogProps) {
  const who = project?.updated_by ?? 'Someone';
  return (
    <Modal opened={project !== null} onClose={onClose} title="This network was changed" size="lg">
      <Text>{who} changed this network since you opened it.</Text>
      {project && (
        <Text size="sm" c="dimmed" mt={4}>
          Their version: “{project.name}”, {project.num_layers} layers, saved{' '}
          {formatDate(project.updated_at)}.
        </Text>
      )}
      <List size="sm" mt="md" spacing={6}>
        <List.Item>
          <b>Load their version</b> replaces the canvas. Your changes are lost.
        </List.Item>
        <List.Item>
          <b>Keep mine</b> saves your version over theirs.
        </List.Item>
        <List.Item>
          <b>Save mine as a copy</b> keeps both: yours becomes a new network.
        </List.Item>
      </List>
      <Flex
        direction={{ base: 'column', xs: 'row' }}
        justify="flex-end"
        gap="xs"
        mt="xl"
        wrap="wrap"
      >
        <Button variant="default" onClick={() => onChoose('theirs')}>
          Load their version
        </Button>
        <Button variant="default" onClick={() => onChoose('copy')}>
          Save mine as a copy
        </Button>
        <Button onClick={() => onChoose('mine')}>Keep mine</Button>
      </Flex>
    </Modal>
  );
}
