import { useState } from 'react';
import { Button, Group, Modal, SegmentedControl, Stack, Text, TextInput } from '@mantine/core';
import type { ColumnInfo, ColumnKind, NewStep } from '@/api/types';
import { i18n, Trans, useTranslation } from '@/i18n';
import { castStep, renameError, renameStep } from './columnActions';
import { KIND_LABELS, KINDS } from './kinds';

export type ColumnDialogState = { mode: 'rename' | 'cast'; column: ColumnInfo } | null;

interface ColumnDialogProps {
  state: ColumnDialogState;
  /** All column names (a new name must not be taken). */
  names: string[];
  onClose: () => void;
  onAddStep: (step: NewStep) => void;
}

/** Small dialogs for the column actions that need an input: rename and change type. */
export function ColumnDialog({ state, names, onClose, onAddStep }: ColumnDialogProps) {
  useTranslation();
  const column = state?.column;
  return (
    <Modal
      opened={state !== null}
      onClose={onClose}
      title={
        column
          ? state?.mode === 'rename'
            ? i18n.t('ui.data.ColumnDialog.rename', { name: column.name })
            : i18n.t('ui.data.ColumnDialog.changeTheTypeOf', { name: column.name })
          : ''
      }
      size="sm"
    >
      {column && state?.mode === 'rename' && (
        <RenameForm
          key={column.name}
          column={column}
          names={names}
          onCancel={onClose}
          onSubmit={(step) => {
            onAddStep(step);
            onClose();
          }}
        />
      )}
      {column && state?.mode === 'cast' && (
        <CastForm
          key={column.name}
          column={column}
          onCancel={onClose}
          onSubmit={(step) => {
            onAddStep(step);
            onClose();
          }}
        />
      )}
    </Modal>
  );
}

interface FormProps {
  column: ColumnInfo;
  onCancel: () => void;
  onSubmit: (step: NewStep) => void;
}

function RenameForm({ column, names, onCancel, onSubmit }: FormProps & { names: string[] }) {
  useTranslation();
  const [name, setName] = useState(column.name);
  const [touched, setTouched] = useState(false);
  const error = renameError(column.name, name, names);
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        setTouched(true);
        if (!error) {
          onSubmit(renameStep(column.name, name));
        }
      }}
    >
      <Stack gap="sm">
        <TextInput
          label={i18n.t('ui.data.ColumnDialog.newName')}
          value={name}
          data-autofocus
          error={touched || name !== column.name ? error : null}
          onChange={(event) => setName(event.currentTarget.value)}
          onFocus={(event) => event.currentTarget.select()}
        />
        <Text size="xs" c="dimmed">
          {i18n.t('ui.data.ColumnDialog.addsAStepThatRenames')}
        </Text>
        <Group justify="flex-end" gap="xs">
          <Button variant="default" onClick={onCancel}>
            {i18n.t('ui.data.ColumnDialog.cancel')}
          </Button>
          <Button type="submit" disabled={Boolean(error)}>
            {i18n.t('ui.data.ColumnDialog.rename2')}
          </Button>
        </Group>
      </Stack>
    </form>
  );
}

function CastForm({ column, onCancel, onSubmit }: FormProps) {
  useTranslation();
  const choices = KINDS.filter((kind) => kind !== column.kind);
  const [kind, setKind] = useState<ColumnKind>(choices[0]);
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit(castStep(column.name, kind));
      }}
    >
      <Stack gap="sm">
        <Text size="sm">
          <Trans
            i18nKey="ui.data.ColumnDialog.now"
            values={{ value: KIND_LABELS[column.kind] }}
            components={[<b />]}
          />
        </Text>
        <SegmentedControl
          aria-label={i18n.t('ui.data.ColumnDialog.newType')}
          orientation="vertical"
          fullWidth
          data={choices.map((value) => ({ value, label: KIND_LABELS[value] }))}
          value={kind}
          onChange={(value) => setKind(value as ColumnKind)}
        />
        <Text size="xs" c="dimmed">
          {i18n.t('ui.data.ColumnDialog.addsAStepThatConverts')}
        </Text>
        <Group justify="flex-end" gap="xs">
          <Button variant="default" onClick={onCancel}>
            {i18n.t('ui.data.ColumnDialog.cancel')}
          </Button>
          <Button type="submit">{i18n.t('ui.data.ColumnDialog.changeType')}</Button>
        </Group>
      </Stack>
    </form>
  );
}
