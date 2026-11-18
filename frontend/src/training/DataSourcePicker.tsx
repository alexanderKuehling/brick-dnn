import { useId, useRef } from 'react';
import {
  IconChevronDown,
  IconChevronRight,
  IconFileSpreadsheet,
  IconLayersIntersect,
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
  Code,
  Collapse,
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
import { useDisclosure } from '@mantine/hooks';
import type { BuiltinDataset, Dataset, TrainingStatus } from '@/api';
import { UploadNotice } from '@/components/UploadNotice';
import { UploadProgress } from '@/components/UploadProgress';
import type { UploadBytes } from '@/components/useUploadProgress';
import { currentLocale, i18n, Trans, useTranslation } from '@/i18n';
import { isOwnDataset } from './access';
import { classShares, datasetDetails, formatSplit, segmentationNotes } from './format';
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

const TABS: SourceTab[] = ['random', 'builtin', 'upload'];

export function DataSourcePicker(props: DataSourcePickerProps) {
  const { t } = useTranslation();
  const { status, choice, onChange } = props;

  return (
    <Stack gap="xs">
      <SegmentedControl
        fullWidth
        aria-label={t('training.source.label')}
        data={TABS.map((value) => ({ value, label: t(`training.source.tabs.${value}`) }))}
        value={choice.tab}
        onChange={(tab) => onChange({ ...choice, tab: tab as SourceTab })}
      />

      {choice.tab === 'random' && (
        <Text size="sm" c="dimmed">
          {t('training.source.random')}
        </Text>
      )}

      {choice.tab === 'builtin' && (
        <Radio.Group
          label={t('training.source.dataset')}
          description={
            status.torchvision
              ? status.builtin_datasets.some((dataset) => dataset.download_mb === 0)
                ? t('training.source.downloadedOrGenerated')
                : t('training.source.downloaded')
              : t('training.source.needsTorchvision')
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
                      {builtinDetails(dataset)}
                    </Text>
                    {dataset.id === choice.builtin && <TextExamples dataset={dataset} />}
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

/** The first test texts of a built-in text dataset with their class (nothing for images). */
function TextExamples({ dataset }: { dataset: BuiltinDataset }) {
  const { t } = useTranslation();
  if (dataset.input_kind !== 'tokens' || !dataset.examples?.length) {
    return null;
  }
  return (
    <Stack gap={2} mt={6} aria-label={t('training.source.examples')}>
      <Text size="xs" fw={600}>
        {t('training.source.examples')}
      </Text>
      {dataset.examples.map((example) => (
        <Text key={example.text} size="xs" className={classes.textExample}>
          <span className={classes.textExampleText}>{example.text}</span>
          {' → '}
          <b>{example.class}</b>
        </Text>
      ))}
    </Stack>
  );
}

/** "60,000 training and 10,000 test images · 10 classes · 12 MB download". */
function builtinDetails(dataset: BuiltinDataset): string {
  const numbers = {
    train: dataset.train_size.toLocaleString(currentLocale()),
    test: dataset.test_size.toLocaleString(currentLocale()),
  };
  if (dataset.input_kind === 'tokens') {
    // Generated on the server: texts as token ids, one class per text.
    return i18n.t('training.source.textDetails', {
      sizes: i18n.t('training.source.textSizes', numbers),
      count: dataset.num_classes,
      classes: dataset.class_names.join(', '),
      words: (dataset.vocab_size ?? 0).toLocaleString(currentLocale()),
      length: dataset.sequence_length ?? dataset.sample_shape[0],
    });
  }
  const sizes = i18n.t('training.source.sizes', numbers);
  if (dataset.task === 'segmentation') {
    // Generated on the server: a class for every pixel.
    const classes = dataset.class_names.join(', ');
    return i18n.t('training.source.segmentationDetails', {
      sizes,
      count: dataset.num_classes,
      classes,
    });
  }
  return i18n.t('training.source.details', {
    sizes,
    count: dataset.num_classes,
    mb: dataset.download_mb.toLocaleString(currentLocale()),
  });
}

/** How to pack images and masks for a segmentation dataset (shown on demand). */
function SegmentationHelp({ status }: { status: TrainingStatus }) {
  useTranslation();
  const [opened, { toggle }] = useDisclosure(false);
  const limits = status.segmentation;
  if (!limits) {
    return null; // A server without segmentation.
  }
  return (
    <div>
      <Button
        variant="subtle"
        color="gray"
        size="compact-xs"
        leftSection={opened ? <IconChevronDown size={14} /> : <IconChevronRight size={14} />}
        onClick={toggle}
        aria-expanded={opened}
      >
        {i18n.t('ui.training.DataSourcePicker.segmentationHowToPackThe')}
      </Button>
      <Collapse expanded={opened}>
        <Stack gap={6} mt={6}>
          <Text size="xs" c="dimmed">
            {i18n.t('ui.training.DataSourcePicker.imagesWithOneMaskEach')}
          </Text>
          <Code block fz="xs">
            {i18n.t('ui.training.DataSourcePicker.images001JpgImages002')}
          </Code>
          <Text size="xs" c="dimmed">
            <Trans
              i18nKey="ui.training.DataSourcePicker.masksArePngFilesThat"
              values={{ max_classes: limits.max_classes }}
              components={[<b />]}
            />
          </Text>
          <Text size="xs" c="dimmed">
            <Trans
              i18nKey="ui.training.DataSourcePicker.withTrainImagesTrainMasks"
              values={{
                max_input_side: limits.max_input_side,
                max_input_side2: limits.max_input_side,
              }}
              components={[<Code fz="xs" />, <Code fz="xs" />, <Code fz="xs" />, <Code fz="xs" />]}
            />
          </Text>
        </Stack>
      </Collapse>
    </div>
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
  useTranslation();
  // Reset the file input after each pick, so the same file can be chosen again (e.g. after
  // fixing it, or after a failed upload).
  const resetFile = useRef<() => void>(null);
  return (
    <>
      <Group justify="space-between" align="flex-start" wrap="nowrap" gap="sm">
        <Text size="xs" c="dimmed">
          <b>{i18n.t('ui.training.DataSourcePicker.csv')}</b>
          {i18n.t('ui.training.DataSourcePicker.numericFeatureColumnsTheTarget')}{' '}
          <b>{i18n.t('ui.training.DataSourcePicker.zip')}</b>
          {i18n.t('ui.training.DataSourcePicker.oneFolderPerClassWith')}
          {status.segmentation ? (
            <>
              <Trans
                i18nKey="ui.training.DataSourcePicker.orForSegmentationTheFolders"
                components={[<Code fz="xs" />, <Code fz="xs" />]}
              />
            </>
          ) : null}
          {i18n.t('ui.training.DataSourcePicker.upToMb', { max_upload_mb: status.max_upload_mb })}
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
              {i18n.t('ui.training.DataSourcePicker.upload')}
            </Button>
          )}
        </FileButton>
      </Group>

      <SegmentationHelp status={status} />
      <UploadNotice />
      <DataTabHint />

      {uploading && uploadProgress && (
        <UploadProgress bytes={uploadProgress} fileName={uploading} onCancel={onCancelUpload} />
      )}
      {uploadError && (
        <Alert
          color="red"
          title={i18n.t('ui.training.DataSourcePicker.uploadFailed')}
          withCloseButton
          onClose={onDismissUploadError}
        >
          {uploadError}
        </Alert>
      )}
      {datasetsError && (
        <Alert color="red">
          {datasetsError}{' '}
          <Button size="compact-xs" variant="light" color="red" onClick={onReloadDatasets}>
            {i18n.t('ui.training.DataSourcePicker.tryAgain')}
          </Button>
        </Alert>
      )}
      {datasets === null && !datasetsError && <Loader size="sm" />}
      {datasets?.length === 0 && (
        <Text size="sm" c="dimmed">
          {i18n.t('ui.training.DataSourcePicker.youHaveNotUploadedAny')}
        </Text>
      )}

      {datasets && datasets.length > 0 && (
        <Radio.Group
          aria-label={i18n.t('ui.training.DataSourcePicker.datasets')}
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
  useTranslation();
  const own = datasets.filter(isOwnDataset);
  const shared = datasets.filter((dataset) => !isOwnDataset(dataset));
  const ownHeading = useId();
  const sharedHeading = useId();
  return (
    <Stack gap="xs">
      <Text size="sm" fw={500} id={ownHeading}>
        {i18n.t('ui.training.DataSourcePicker.yourDatasets')}
      </Text>
      {own.length === 0 ? (
        <Text size="sm" c="dimmed">
          {i18n.t('ui.training.DataSourcePicker.youHaveNotUploadedAny')}
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
              {i18n.t('ui.training.DataSourcePicker.sharedWithYou')}
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
  useTranslation();
  const own = isOwnDataset(dataset);
  const segmentation = dataset.kind === 'segmentation';
  const shares = segmentation ? classShares(dataset) : null;
  const notes = segmentation ? segmentationNotes(dataset) : null;
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
          <ThemeIcon variant="light" color={KIND_COLORS[dataset.kind]} aria-hidden>
            {dataset.kind === 'csv' ? (
              <IconFileSpreadsheet size={16} />
            ) : dataset.kind === 'segmentation' ? (
              <IconLayersIntersect size={16} />
            ) : (
              <IconPhoto size={16} />
            )}
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
                      ? i18n.t('ui.training.DataSourcePicker.madeInTheDataTab', {
                          name: dataset.prepared_from.name,
                        })
                      : i18n.t('ui.training.DataSourcePicker.madeInTheDataTab2')
                  }
                >
                  <Badge size="xs" variant="light" color="teal" style={{ flexShrink: 0 }}>
                    {i18n.t('ui.training.DataSourcePicker.prepared')}
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
              {KIND_LABELS[dataset.kind]} · {datasetDetails(dataset)}
            </Text>
            {shares && (
              <Text size="xs" c="dimmed" style={{ overflowWrap: 'anywhere' }}>
                {i18n.t('ui.training.DataSourcePicker.pixels', { shares })}
              </Text>
            )}
            {dataset.info.predefined_split && (
              <Text size="xs" c="dimmed">
                {i18n.t('ui.training.DataSourcePicker.splitTrainingValidationTest', {
                  predefined_split: formatSplit(dataset.info.predefined_split),
                })}
              </Text>
            )}
            {notes && (
              <Text size="xs" c="orange" style={{ overflowWrap: 'anywhere' }}>
                {notes}
              </Text>
            )}
          </div>
        </Group>
      </Radio.Card>
      {onDelete && own ? (
        <Tooltip label={i18n.t('ui.training.DataSourcePicker.delete')}>
          <ActionIcon
            variant="subtle"
            color="gray"
            onClick={() => onDelete(dataset)}
            aria-label={i18n.t('ui.training.DataSourcePicker.delete2', { name: dataset.name })}
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

const KIND_LABELS: Record<Dataset['kind'], string> = {
  csv: 'CSV',
  get images() {
    return i18n.t('ui.training.DataSourcePicker.images');
  },
  get segmentation() {
    return i18n.t('ui.training.DataSourcePicker.segmentation');
  },
};
const KIND_COLORS: Record<Dataset['kind'], string> = {
  csv: 'teal',
  images: 'violet',
  segmentation: 'grape',
};

/** Datasets made in the Data tab (they bring their own split). */
function isPrepared(dataset: Dataset): boolean {
  // Segmentation datasets can bring a split too: their train/, val/ and test/ folders.
  return dataset.kind === 'csv' && Boolean(dataset.prepared_from || dataset.info.predefined_split);
}

/** Points to the Data tab, which turns messy CSV files into training data. */
function DataTabHint() {
  // The panel is also shown without a router (e.g. in tests); then a plain link is used.
  const { t } = useTranslation();
  const inRouter = useInRouterContext();
  const text = t('training.source.dataTab');
  return (
    <Group gap={6} wrap="nowrap" align="flex-start">
      <IconTableShortcut
        size={16}
        color="var(--mantine-color-dimmed)"
        style={{ flexShrink: 0, marginTop: 1 }}
        aria-hidden
      />
      <Text size="xs" c="dimmed">
        {t('training.source.messyCsv')}{' '}
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
