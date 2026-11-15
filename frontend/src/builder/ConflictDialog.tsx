import { Button, Flex, List, Modal, Text } from '@mantine/core';
import type { Project } from '@/api';
import { Trans, useTranslation } from '@/i18n';
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
  const { t } = useTranslation();
  const who = project?.updated_by ?? t('builder.page.someone');
  return (
    <Modal
      opened={project !== null}
      onClose={onClose}
      title={t('builder.conflict.title')}
      size="lg"
    >
      <Text>{t('builder.conflict.changed', { user: who })}</Text>
      {project && (
        <Text size="sm" c="dimmed" mt={4}>
          {t('builder.conflict.theirs', {
            name: project.name,
            count: project.num_layers,
            date: formatDate(project.updated_at),
          })}
        </Text>
      )}
      <List size="sm" mt="md" spacing={6}>
        <List.Item>
          <Trans i18nKey="builder.conflict.loadTheirsHint" components={{ b: <b /> }} />
        </List.Item>
        <List.Item>
          <Trans i18nKey="builder.conflict.keepMineHint" components={{ b: <b /> }} />
        </List.Item>
        <List.Item>
          <Trans i18nKey="builder.conflict.copyHint" components={{ b: <b /> }} />
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
          {t('builder.conflict.loadTheirs')}
        </Button>
        <Button variant="default" onClick={() => onChoose('copy')}>
          {t('builder.conflict.copy')}
        </Button>
        <Button onClick={() => onChoose('mine')}>{t('builder.conflict.keepMine')}</Button>
      </Flex>
    </Modal>
  );
}
