import { useEffect, useState } from 'react';
import { Button, Checkbox, Group, Modal, Stack, TagsInput, Textarea } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { api, type RunAnnotations, type RunRow } from '@/api';
import { useTranslation } from '@/i18n';

/** Tags, a note and the star of a run (its owner and the editors of its network). */
export function AnnotateDialog({
  row,
  knownTags,
  onClose,
  onSaved,
  maxTags = 20,
  maxNote = 2000,
}: {
  row: RunRow | null;
  /** Tags of other runs, offered while typing. */
  knownTags: string[];
  onClose: () => void;
  onSaved: (annotations: RunAnnotations) => void;
  maxTags?: number;
  maxNote?: number;
}) {
  const { t } = useTranslation();
  const [tags, setTags] = useState<string[]>([]);
  const [note, setNote] = useState('');
  const [starred, setStarred] = useState(false);
  const [saving, setSaving] = useState(false);
  useEffect(() => {
    if (row) {
      setTags(row.tags ?? []);
      setNote(row.note ?? '');
      setStarred(row.starred);
    }
  }, [row]);

  const save = async () => {
    if (!row) {
      return;
    }
    setSaving(true);
    try {
      onSaved(await api.training.runs.annotate(row.id, { tags, note, starred }));
      onClose();
    } catch (err) {
      notifications.show({
        color: 'red',
        title: t('experiments.annotate.failed'),
        message: (err as Error).message,
      });
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal
      opened={row !== null}
      onClose={onClose}
      title={row ? t('experiments.annotate.title', { name: row.name }) : ''}
    >
      <Stack gap="sm">
        <TagsInput
          label={t('experiments.annotate.tags')}
          description={t('experiments.annotate.tagsHelp')}
          data={knownTags}
          value={tags}
          onChange={setTags}
          maxTags={maxTags}
          clearable
          splitChars={[',']}
        />
        <Textarea
          label={t('experiments.annotate.note')}
          value={note}
          onChange={(event) => setNote(event.currentTarget.value)}
          maxLength={maxNote}
          autosize
          minRows={3}
          maxRows={10}
          description={t('experiments.annotate.noteHelp', { count: note.length, max: maxNote })}
        />
        <Checkbox
          label={t('experiments.annotate.star')}
          checked={starred}
          onChange={(event) => setStarred(event.currentTarget.checked)}
        />
        <Group justify="flex-end">
          <Button variant="default" onClick={onClose}>
            {t('experiments.annotate.cancel')}
          </Button>
          <Button onClick={save} loading={saving}>
            {t('experiments.annotate.save')}
          </Button>
        </Group>
      </Stack>
    </Modal>
  );
}
