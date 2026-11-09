import { useEffect, useRef, useState } from 'react';
import {
  IconAlertTriangle,
  IconChevronDown,
  IconChevronRight,
  IconCircleCheck,
  IconFileSpreadsheet,
  IconUpload,
} from '@tabler/icons-react';
import {
  Alert,
  Button,
  Checkbox,
  Collapse,
  FileButton,
  Group,
  List,
  Modal,
  Select,
  SimpleGrid,
  Stack,
  TagsInput,
  Text,
  TextInput,
  ThemeIcon,
} from '@mantine/core';
import { api, type DataSource, type ImportOptions, type ImportReport } from '@/api';
import { UploadNotice } from '@/components/UploadNotice';
import { UploadProgress } from '@/components/UploadProgress';
import { useUploadProgress } from '@/components/useUploadProgress';
import { formatBytes } from '@/training/format';
import classes from './DataPage.module.css';

export const ACCEPTED_FILES = '.csv,.tsv,.txt,text/csv,text/plain,text/tab-separated-values';

const DEFAULT_OPTIONS: ImportOptions = {
  delimiter: 'auto',
  decimal: 'auto',
  header: 'auto',
  encoding: 'auto',
  missing_values: null,
};

const DELIMITERS: { value: ImportOptions['delimiter']; label: string }[] = [
  { value: 'auto', label: 'Detect' },
  { value: ',', label: 'Comma ( , )' },
  { value: ';', label: 'Semicolon ( ; )' },
  { value: '\t', label: 'Tab' },
  { value: '|', label: 'Pipe ( | )' },
];
const DECIMALS: { value: ImportOptions['decimal']; label: string }[] = [
  { value: 'auto', label: 'Detect' },
  { value: '.', label: 'Point (1.5)' },
  { value: ',', label: 'Comma (1,5)' },
];
const HEADERS: { value: ImportOptions['header']; label: string }[] = [
  { value: 'auto', label: 'Detect' },
  { value: 'yes', label: 'First row has the column names' },
  { value: 'no', label: 'No header row' },
];
const ENCODINGS: { value: ImportOptions['encoding']; label: string }[] = [
  { value: 'auto', label: 'Detect' },
  { value: 'utf-8', label: 'UTF-8' },
  { value: 'cp1252', label: 'Windows-1252' },
  { value: 'latin-1', label: 'Latin-1 (ISO 8859-1)' },
];

const ENCODING_NAMES: Record<string, string> = {
  'utf-8': 'UTF-8',
  'utf-8-sig': 'UTF-8',
  cp1252: 'Windows-1252',
  'latin-1': 'Latin-1',
};

const DELIMITER_NAMES: Record<string, string> = {
  ',': 'comma',
  ';': 'semicolon',
  '\t': 'tab',
  '|': 'pipe',
};

/** "1,200 rows × 9 columns" */
export function tableSize(rows: number, columns: number): string {
  const plural = (n: number, word: string) =>
    `${n.toLocaleString('en-US')} ${word}${n === 1 ? '' : 's'}`;
  return `${plural(rows, 'row')} × ${plural(columns, 'column')}`;
}

/** What the reader detected: "semicolon-separated · decimal comma · UTF-8 · header row". */
export function describeReport(report: ImportReport): string {
  const delimiter = DELIMITER_NAMES[report.delimiter] ?? `“${report.delimiter}”`;
  return [
    `${delimiter}-separated`,
    report.decimal === ',' ? 'decimal comma' : 'decimal point',
    ENCODING_NAMES[report.encoding.toLowerCase()] ?? report.encoding,
    report.header ? 'header row' : 'no header row',
  ].join(' · ');
}

/** Why `file` cannot be imported, or null. */
export function fileProblem(file: File, maxUploadMb: number): string | null {
  if (!/\.(csv|tsv|txt)$/i.test(file.name)) {
    return `“${file.name}” is not a CSV file. Choose a .csv, .tsv or .txt file.`;
  }
  if (maxUploadMb > 0 && file.size > maxUploadMb * 1024 * 1024) {
    return `The file is ${formatBytes(file.size)}; the server accepts up to ${maxUploadMb} MB.`;
  }
  return null;
}

const stem = (filename: string) => filename.replace(/\.[^.]+$/, '');

interface ImportDialogProps {
  opened: boolean;
  onClose: () => void;
  /** A file dropped on the page; the dialog starts with it. */
  initialFile: File | null;
  maxUploadMb: number;
  /** The open preparation, if any: importing a table to join it there is an option. */
  currentPrepName: string | null;
  /**
   * Called after a successful import. `openNew`: start a new preparation with the table.
   * `signal` is aborted if the dialog is closed meanwhile (then nothing should open). Errors
   * are reported by the callee.
   */
  onImported: (source: DataSource, openNew: boolean, signal: AbortSignal) => Promise<void> | void;
}

/**
 * Import a CSV file: choose or drop it, name it, optionally override what the reader
 * detects. Shows the reader's warnings afterwards. Mount it with a new `key` per import.
 */
export function ImportDialog({
  opened,
  onClose,
  initialFile,
  maxUploadMb,
  currentPrepName,
  onImported,
}: ImportDialogProps) {
  const [file, setFile] = useState<File | null>(initialFile);
  const [name, setName] = useState(initialFile ? stem(initialFile.name) : '');
  const [nameEdited, setNameEdited] = useState(false);
  const [options, setOptions] = useState<ImportOptions>(DEFAULT_OPTIONS);
  const [advanced, setAdvanced] = useState(false);
  const [openNew, setOpenNew] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(
    initialFile ? fileProblem(initialFile, maxUploadMb) : null
  );
  const [imported, setImported] = useState<DataSource | null>(null);
  const [dragging, setDragging] = useState(false);
  const resetFile = useRef<() => void>(null);
  /** The running upload; aborted when the dialog is cancelled or the page is left. */
  const upload = useRef<AbortController | null>(null);
  const progress = useUploadProgress();

  useEffect(() => () => upload.current?.abort(), []);

  /** Close the dialog; a running import is cancelled and its result ignored. */
  const close = () => {
    upload.current?.abort();
    onClose();
  };

  const choose = (next: File | null) => {
    resetFile.current?.();
    if (!next) {
      return;
    }
    setFile(next);
    setError(fileProblem(next, maxUploadMb));
    if (!nameEdited) {
      setName(stem(next.name));
    }
  };

  const set =
    <K extends keyof ImportOptions>(key: K) =>
    (value: ImportOptions[K]) =>
      setOptions((current) => ({ ...current, [key]: value }));

  const submit = async () => {
    if (!file || busy) {
      return;
    }
    const problem = fileProblem(file, maxUploadMb);
    if (problem) {
      setError(problem);
      return;
    }
    const controller = new AbortController();
    upload.current = controller;
    setBusy(true);
    setError(null);
    progress.start(file.size);
    let source: DataSource;
    try {
      const changed = Object.fromEntries(
        Object.entries(options).filter(
          ([key, value]) => value !== DEFAULT_OPTIONS[key as keyof ImportOptions]
        )
      ) as Partial<ImportOptions>;
      source = await api.data.sources.upload(
        file,
        { ...changed, ...(name.trim() ? { name: name.trim() } : {}) },
        { signal: controller.signal, onUploadProgress: progress.report }
      );
    } catch (err) {
      if (!controller.signal.aborted) {
        setError((err as Error).message);
        setBusy(false);
        progress.stop();
      }
      return;
    }
    if (controller.signal.aborted) {
      // Cancelled or left while the answer was on its way.
      return;
    }
    await onImported(source, openNew, controller.signal);
    if (controller.signal.aborted) {
      return;
    }
    upload.current = null;
    setBusy(false);
    progress.stop();
    if (source.report.warnings.length > 0) {
      setImported(source);
    } else {
      onClose();
    }
  };

  return (
    <Modal
      opened={opened}
      onClose={close}
      title={imported ? 'Imported' : 'Import a CSV file'}
      size="lg"
      closeOnClickOutside={!busy}
    >
      {imported ? (
        <Stack gap="md">
          <Alert
            color="green"
            icon={<IconCircleCheck size={18} />}
            title={`“${imported.name}”: ${tableSize(imported.n_rows, imported.n_columns)}`}
          >
            <Text size="sm">Read as {describeReport(imported.report)}.</Text>
          </Alert>
          <Alert color="yellow" icon={<IconAlertTriangle size={18} />} title="Please check">
            <List size="sm" spacing={4}>
              {imported.report.warnings.map((warning) => (
                <List.Item key={warning}>{warning}</List.Item>
              ))}
            </List>
          </Alert>
          <Text size="xs" c="dimmed">
            If the file was read wrongly, import it again with other options under “Import options”,
            or fix the column types with a “Change type” step.
          </Text>
          <Group justify="flex-end">
            <Button onClick={onClose} data-autofocus>
              Continue
            </Button>
          </Group>
        </Stack>
      ) : (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
        >
          <Stack gap="md">
            <div
              className={classes.dropzone}
              data-active={dragging || undefined}
              onDragOver={(event) => {
                if (event.dataTransfer.types.includes('Files')) {
                  event.preventDefault();
                  event.stopPropagation();
                  setDragging(true);
                }
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                event.stopPropagation();
                setDragging(false);
                choose(event.dataTransfer.files[0] ?? null);
              }}
            >
              <ThemeIcon variant="light" size={40} radius="xl" aria-hidden>
                <IconFileSpreadsheet size={22} />
              </ThemeIcon>
              {file ? (
                <div>
                  <Text size="sm" fw={600} style={{ overflowWrap: 'anywhere' }}>
                    {file.name}
                  </Text>
                  <Text size="xs" c="dimmed">
                    {formatBytes(file.size)}
                  </Text>
                </div>
              ) : (
                <Text size="sm" c="dimmed">
                  Drop a CSV file here, or choose one. Up to {maxUploadMb} MB.
                </Text>
              )}
              <FileButton resetRef={resetFile} onChange={choose} accept={ACCEPTED_FILES}>
                {(buttonProps) => (
                  <Button
                    {...buttonProps}
                    variant={file ? 'default' : 'light'}
                    size="xs"
                    disabled={busy}
                    leftSection={<IconUpload size={14} />}
                  >
                    {file ? 'Choose another file' : 'Choose file'}
                  </Button>
                )}
              </FileButton>
            </div>
            <UploadNotice />

            <TextInput
              label="Name"
              placeholder={file ? stem(file.name) : 'e.g. customers'}
              value={name}
              onChange={(event) => {
                setName(event.currentTarget.value);
                setNameEdited(true);
              }}
              maxLength={120}
              disabled={busy}
            />
            <Checkbox
              label="Start a new preparation with this table"
              description={
                currentPrepName
                  ? `Turn this off to only add the table, e.g. to join it into “${currentPrepName}”.`
                  : undefined
              }
              checked={openNew}
              onChange={(event) => setOpenNew(event.currentTarget.checked)}
              disabled={busy}
            />

            <div>
              <Button
                variant="subtle"
                color="gray"
                size="compact-sm"
                leftSection={
                  advanced ? <IconChevronDown size={14} /> : <IconChevronRight size={14} />
                }
                onClick={() => setAdvanced((value) => !value)}
                aria-expanded={advanced}
              >
                Import options
              </Button>
              <Collapse expanded={advanced}>
                <Stack gap="sm" pt="xs">
                  <Text size="xs" c="dimmed">
                    Brick DNN detects these from the file. Set them only if the detection gets it
                    wrong.
                  </Text>
                  <SimpleGrid cols={{ base: 1, xs: 2 }} spacing="sm" verticalSpacing="sm">
                    <Select
                      label="Separator"
                      data={DELIMITERS}
                      value={options.delimiter}
                      onChange={(value) =>
                        value && set('delimiter')(value as ImportOptions['delimiter'])
                      }
                      allowDeselect={false}
                      disabled={busy}
                    />
                    <Select
                      label="Decimal mark"
                      data={DECIMALS}
                      value={options.decimal}
                      onChange={(value) =>
                        value && set('decimal')(value as ImportOptions['decimal'])
                      }
                      allowDeselect={false}
                      disabled={busy}
                    />
                    <Select
                      label="Header row"
                      data={HEADERS}
                      value={options.header}
                      onChange={(value) => value && set('header')(value as ImportOptions['header'])}
                      allowDeselect={false}
                      disabled={busy}
                    />
                    <Select
                      label="Text encoding"
                      data={ENCODINGS}
                      value={options.encoding}
                      onChange={(value) =>
                        value && set('encoding')(value as ImportOptions['encoding'])
                      }
                      allowDeselect={false}
                      disabled={busy}
                    />
                  </SimpleGrid>
                  <TagsInput
                    label="Values that mean “missing”"
                    description="Empty: the usual ones (NA, N/A, null, -, ? …). Press Enter after each value."
                    placeholder={options.missing_values?.length ? undefined : 'e.g. -999, unknown'}
                    value={options.missing_values ?? []}
                    onChange={(values) => set('missing_values')(values.length > 0 ? values : null)}
                    acceptValueOnBlur
                    disabled={busy}
                  />
                </Stack>
              </Collapse>
            </div>

            {error && (
              <Alert color="red" icon={<IconAlertTriangle size={18} />} title="Cannot import">
                {error}
              </Alert>
            )}
            {busy && file && progress.bytes && (
              <UploadProgress
                bytes={progress.bytes}
                fileName={file.name}
                processingText="Reading and checking the CSV …"
              />
            )}

            <Group justify="flex-end" gap="xs">
              <Button variant="default" onClick={close}>
                Cancel
              </Button>
              <Button
                type="submit"
                loading={busy}
                disabled={!file || fileProblem(file, maxUploadMb) !== null}
              >
                Import
              </Button>
            </Group>
          </Stack>
        </form>
      )}
    </Modal>
  );
}
