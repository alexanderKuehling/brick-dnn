import { useId, useRef } from 'react';
import {
  IconFileSpreadsheet,
  IconPhoto,
  IconTableShortcut,
  IconTrash,
  IconUpload,
  IconUsers,
} from '@tabler/icons-react';
import { Link, useInRouterContext } from 'react-router-dom';
import {
  ActionIcon,
  Alert,
  Anchor,
  Badge,
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
import { UploadProgress } from '@/components/UploadProgress';
import type { UploadBytes } from '@/components/useUploadProgress';
import { isOwnDataset } from './access';
import { datasetDetails, formatSplit } from './format';
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
  /** How much of it is sent; null for the first moment of an upload. */
  uploadProgress: UploadBytes | null;
  uploadError: string | null;
  onDismissUploadError: () => void;
  onUpload: (file: File) => void;
  onCancelUpload: () => void;
  /** Only offered for the user's own datasets. */
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
                      {dataset.train_size.toLocaleString('en-US')} training and{' '}
                      {dataset.test_size.toLocaleString('en-US')} test images ·{' '}
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
  uploadProgress,
  uploadError,
  onDismissUploadError,
  onUpload,
  onCancelUpload,
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

      <DataTabHint />

      {uploading && uploadProgress && (
        <UploadProgress bytes={uploadProgress} fileName={uploading} onCancel={onCancelUpload} />
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
          aria-label="Datasets"
          value={choice.upload !== null ? String(choice.upload) : null}
          onChange={(id) => onChange({ ...choice, upload: Number(id) })}
        >
          <DatasetGroups datasets={datasets} onDelete={onDelete} />
        </Radio.Group>
      )}
    </>
  );
}

/** The user's own datasets, then the ones others shared in the user's networks. */
function DatasetGroups({
  datasets,
  onDelete,
}: {
  datasets: Dataset[];
  onDelete: (dataset: Dataset) => void;
}) {
  const own = datasets.filter(isOwnDataset);
  const shared = datasets.filter((dataset) => !isOwnDataset(dataset));
  const ownHeading = useId();
  const sharedHeading = useId();
  return (
    <Stack gap="xs">
      <Text size="sm" fw={500} id={ownHeading}>
        Your datasets
      </Text>
      {own.length === 0 ? (
        <Text size="sm" c="dimmed">
          You have not uploaded any data yet.
        </Text>
      ) : (
        <Stack gap="xs" role="group" aria-labelledby={ownHeading}>
          {own.map((dataset) => (
            <DatasetCard key={dataset.id} dataset={dataset} onDelete={onDelete} />
          ))}
        </Stack>
      )}
      {shared.length > 0 && (
        <>
          <Group gap={6} wrap="nowrap" mt="xs">
            <IconUsers size={16} color="var(--mantine-color-dimmed)" aria-hidden />
            <Text size="sm" fw={500} id={sharedHeading}>
              Shared with you
            </Text>
          </Group>
          <Stack gap="xs" role="group" aria-labelledby={sharedHeading}>
            {shared.map((dataset) => (
              <DatasetCard key={dataset.id} dataset={dataset} />
            ))}
          </Stack>
        </>
      )}
    </Stack>
  );
}

/** "by alice · shared in Churn" for a dataset of someone else. */
function sharedByLine(dataset: Dataset): string {
  const networks = dataset.shared_in.map((project) => `“${project.name}”`).join(', ');
  return [`by ${dataset.owner}`, networks && `shared in ${networks}`].filter(Boolean).join(' · ');
}

function DatasetCard({
  dataset,
  onDelete,
}: {
  dataset: Dataset;
  /** Without it, the dataset cannot be deleted here (it belongs to someone else). */
  onDelete?: (dataset: Dataset) => void;
}) {
  const own = isOwnDataset(dataset);
  return (
    <Group wrap="nowrap" gap="xs">
      <Radio.Card
        value={String(dataset.id)}
        className={classes.card}
        withBorder
        style={{ flex: 1, minWidth: 0 }}
        aria-label={own ? dataset.name : `${dataset.name}, ${sharedByLine(dataset)}`}
      >
        <Group wrap="nowrap" gap="sm">
          <Radio.Indicator />
          <ThemeIcon variant="light" color={dataset.kind === 'csv' ? 'teal' : 'violet'} aria-hidden>
            {dataset.kind === 'csv' ? <IconFileSpreadsheet size={16} /> : <IconPhoto size={16} />}
          </ThemeIcon>
          <div style={{ minWidth: 0 }}>
            <Group gap={6} wrap="nowrap">
              <Text size="sm" fw={600} truncate>
                {dataset.name}
              </Text>
              {isPrepared(dataset) && (
                <Tooltip
                  label={
                    dataset.prepared_from
                      ? `Made in the Data tab from “${dataset.prepared_from.name}”`
                      : 'Made in the Data tab'
                  }
                >
                  <Badge size="xs" variant="light" color="teal" style={{ flexShrink: 0 }}>
                    Prepared
                  </Badge>
                </Tooltip>
              )}
            </Group>
            {!own && (
              <Text size="xs" c="dimmed" style={{ overflowWrap: 'anywhere' }}>
                {sharedByLine(dataset)}
              </Text>
            )}
            <Text size="xs" c="dimmed">
              {dataset.kind === 'csv' ? 'CSV' : 'Images'} · {datasetDetails(dataset)}
            </Text>
            {dataset.info.predefined_split && (
              <Text size="xs" c="dimmed">
                Split {formatSplit(dataset.info.predefined_split)} (training / validation / test)
              </Text>
            )}
          </div>
        </Group>
      </Radio.Card>
      {onDelete && own ? (
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
      ) : (
        <span className={classes.actionSpace} aria-hidden />
      )}
    </Group>
  );
}

/** Datasets made in the Data tab (they bring their own split). */
function isPrepared(dataset: Dataset): boolean {
  return Boolean(dataset.prepared_from || dataset.info.predefined_split);
}

/** Points to the Data tab, which turns messy CSV files into training data. */
function DataTabHint() {
  // The panel is also shown without a router (e.g. in tests); then a plain link is used.
  const inRouter = useInRouterContext();
  const text = 'Data tab';
  return (
    <Group gap={6} wrap="nowrap" align="flex-start">
      <IconTableShortcut
        size={16}
        color="var(--mantine-color-dimmed)"
        style={{ flexShrink: 0, marginTop: 1 }}
        aria-hidden
      />
      <Text size="xs" c="dimmed">
        Messy CSV? Clean, encode and analyse it in the{' '}
        {inRouter ? (
          <Anchor component={Link} to="/data" size="xs">
            {text}
          </Anchor>
        ) : (
          <Anchor href="/data" size="xs">
            {text}
          </Anchor>
        )}
        .
      </Text>
    </Group>
  );
}
