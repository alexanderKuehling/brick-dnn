import { useRef } from 'react';
import { IconFileSpreadsheet, IconPhoto, IconTrash, IconUpload } from '@tabler/icons-react';
import {
  ActionIcon,
  Alert,
  Button,
  FileButton,
  Group,
  Loader,
  Radio,
  SegmentedControl,
  Stack,
  Text,
  ThemeIcon,
  Tooltip,
} from '@mantine/core';
import type { Dataset, TrainingStatus } from '@/api';
import { datasetDetails } from './format';
import type { SourceChoice, SourceTab } from './settings';
import classes from './Training.module.css';

interface DataSourcePickerProps {
  status: TrainingStatus;
  choice: SourceChoice;
  onChange: (choice: SourceChoice) => void;
  datasets: Dataset[] | null;
  datasetsError: string | null;
  onReloadDatasets: () => void;
  /** Name of the file being uploaded. */
  uploading: string | null;
  uploadError: string | null;
  onDismissUploadError: () => void;
  onUpload: (file: File) => void;
  onDelete: (dataset: Dataset) => void;
}

const TABS: { value: SourceTab; label: string }[] = [
  { value: 'random', label: 'Demo data' },
  { value: 'builtin', label: 'Built-in' },
  { value: 'upload', label: 'Your data' },
];

export function DataSourcePicker(props: DataSourcePickerProps) {
  const { status, choice, onChange } = props;

  return (
    <Stack gap="xs">
      <SegmentedControl
        fullWidth
        aria-label="Data source"
        data={TABS}
        value={choice.tab}
        onChange={(tab) => onChange({ ...choice, tab: tab as SourceTab })}
      />

      {choice.tab === 'random' && (
        <Text size="sm" c="dimmed">
          Random inputs with a hidden rule to learn. Works with any network and shows whether it can
          learn at all. Nothing is downloaded.
        </Text>
      )}

      {choice.tab === 'builtin' && (
        <Radio.Group
          label="Dataset"
          description={
            status.torchvision
              ? 'Downloaded to the server on first use.'
              : 'Built-in datasets need torchvision on the server.'
          }
          value={choice.builtin}
          onChange={(id) => onChange({ ...choice, builtin: id })}
        >
          <Stack gap="xs" mt="xs">
            {status.builtin_datasets.map((dataset) => (
              <Radio.Card
                key={dataset.id}
                value={dataset.id}
                disabled={!dataset.available}
                className={classes.card}
                withBorder
              >
                <Group wrap="nowrap" align="flex-start" gap="sm">
                  <Radio.Indicator mt={2} disabled={!dataset.available} />
                  <div>
                    <Text size="sm" fw={600}>
                      {dataset.name}
                    </Text>
                    <Text size="xs" c="dimmed">
                      {dataset.description}
                    </Text>
                    <Text size="xs" c="dimmed">
                      {(dataset.train_size + dataset.val_size).toLocaleString('en-US')} images ·{' '}
                      {dataset.num_classes} classes · {dataset.download_mb} MB download
                    </Text>
                  </div>
                </Group>
              </Radio.Card>
            ))}
          </Stack>
        </Radio.Group>
      )}

      {choice.tab === 'upload' && <Uploads {...props} />}
    </Stack>
  );
}

function Uploads({
  status,
  choice,
  onChange,
  datasets,
  datasetsError,
  onReloadDatasets,
  uploading,
  uploadError,
  onDismissUploadError,
  onUpload,
  onDelete,
}: DataSourcePickerProps) {
  // Reset the file input after each pick, so the same file can be chosen again (e.g. after
  // fixing it, or after a failed upload).
  const resetFile = useRef<() => void>(null);
  return (
    <>
      <Group justify="space-between" align="flex-start" wrap="nowrap" gap="sm">
        <Text size="xs" c="dimmed">
          <b>CSV</b>: numeric feature columns, the target in the last column. <b>ZIP</b>: one folder
          per class with images inside. Up to {status.max_upload_mb} MB.
        </Text>
        <FileButton
          resetRef={resetFile}
          onChange={(file) => {
            resetFile.current?.();
            if (file) {
              onUpload(file);
            }
          }}
          accept=".csv,.zip,text/csv,application/zip"
        >
          {(buttonProps) => (
            <Button
              {...buttonProps}
              variant="light"
              size="xs"
              leftSection={<IconUpload size={14} />}
              loading={uploading !== null}
              style={{ flexShrink: 0 }}
            >
              Upload
            </Button>
          )}
        </FileButton>
      </Group>

      {uploading && (
        <Group gap="xs" wrap="nowrap">
          <Loader size="xs" />
          <Text size="sm">Uploading and checking “{uploading}” …</Text>
        </Group>
      )}
      {uploadError && (
        <Alert color="red" title="Upload failed" withCloseButton onClose={onDismissUploadError}>
          {uploadError}
        </Alert>
      )}
      {datasetsError && (
        <Alert color="red">
          {datasetsError}{' '}
          <Button size="compact-xs" variant="light" color="red" onClick={onReloadDatasets}>
            Try again
          </Button>
        </Alert>
      )}
      {datasets === null && !datasetsError && <Loader size="sm" />}
      {datasets?.length === 0 && (
        <Text size="sm" c="dimmed">
          You have not uploaded any data yet.
        </Text>
      )}

      {datasets && datasets.length > 0 && (
        <Radio.Group
          label="Your datasets"
          value={choice.upload !== null ? String(choice.upload) : null}
          onChange={(id) => onChange({ ...choice, upload: Number(id) })}
        >
          <Stack gap="xs" mt="xs">
            {datasets.map((dataset) => (
              <Group key={dataset.id} wrap="nowrap" gap="xs">
                <Radio.Card
                  value={String(dataset.id)}
                  className={classes.card}
                  withBorder
                  style={{ flex: 1, minWidth: 0 }}
                >
                  <Group wrap="nowrap" gap="sm">
                    <Radio.Indicator />
                    <ThemeIcon
                      variant="light"
                      color={dataset.kind === 'csv' ? 'teal' : 'violet'}
                      aria-hidden
                    >
                      {dataset.kind === 'csv' ? (
                        <IconFileSpreadsheet size={16} />
                      ) : (
                        <IconPhoto size={16} />
                      )}
                    </ThemeIcon>
                    <div style={{ minWidth: 0 }}>
                      <Text size="sm" fw={600} truncate>
                        {dataset.name}
                      </Text>
                      <Text size="xs" c="dimmed">
                        {dataset.kind === 'csv' ? 'CSV' : 'Images'} · {datasetDetails(dataset)}
                      </Text>
                    </div>
                  </Group>
                </Radio.Card>
                <Tooltip label="Delete">
                  <ActionIcon
                    variant="subtle"
                    color="gray"
                    onClick={() => onDelete(dataset)}
                    aria-label={`Delete ${dataset.name}`}
                  >
                    <IconTrash size={16} />
                  </ActionIcon>
                </Tooltip>
              </Group>
            ))}
          </Stack>
        </Radio.Group>
      )}
    </>
  );
}
