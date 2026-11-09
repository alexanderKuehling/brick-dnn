import { useEffect, useRef, useState } from 'react';
import {
  IconAlertTriangle,
  IconChevronDown,
  IconChevronRight,
  IconCircleCheck,
  IconFileSpreadsheet,
  IconLink,
  IconUpload,
} from '@tabler/icons-react';
import {
  Alert,
  Anchor,
  Button,
  Checkbox,
  Collapse,
  FileButton,
  Group,
  List,
  Modal,
  Progress,
  Radio,
  ScrollArea,
  Select,
  SimpleGrid,
  Stack,
  Tabs,
  TagsInput,
  Text,
  TextInput,
  ThemeIcon,
} from '@mantine/core';
import {
  api,
  ApiError,
  type DataSource,
  type ImportFormat,
  type ImportOptions,
  type ImportReport,
} from '@/api';
import { UploadProgress } from '@/components/UploadProgress';
import { useUploadProgress } from '@/components/useUploadProgress';
import { formatBytes } from '@/training/format';
import classes from './DataPage.module.css';

/** What the server imports if its status does not say (see DataStatus.file_types). */
export const DEFAULT_FILE_TYPES = [
  '.csv',
  '.tsv',
  '.txt',
  '.xlsx',
  '.xls',
  '.ods',
  '.json',
  '.jsonl',
  '.ndjson',
  '.parquet',
  '.feather',
  '.arrow',
  '.gz',
  '.bz2',
  '.xz',
  '.zip',
];

const MIME_TYPES = [
  'text/csv',
  'text/plain',
  'text/tab-separated-values',
  'application/json',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  'application/vnd.ms-excel',
  'application/vnd.oasis.opendocument.spreadsheet',
];

/** The `accept` attribute of the file picker. */
export function acceptedFiles(fileTypes: string[]): string {
  return [...fileTypes, ...MIME_TYPES].join(',');
}

/** An example for the link field: a small public CSV file. */
export const EXAMPLE_LINK =
  'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv';

const DEFAULT_OPTIONS: ImportOptions = {
  delimiter: 'auto',
  decimal: 'auto',
  header: 'auto',
  encoding: 'auto',
  missing_values: null,
  sheet: null,
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

const FORMAT_NAMES: Record<ImportFormat, string> = {
  csv: 'CSV',
  xlsx: 'Excel workbook',
  xls: 'Excel 97-2003 workbook',
  ods: 'OpenDocument spreadsheet',
  json: 'JSON',
  jsonl: 'JSON Lines',
  parquet: 'Parquet',
  feather: 'Feather / Arrow',
};

const PACKED: Record<string, string> = {
  gzip: 'gzip-packed',
  bz2: 'bzip2-packed',
  xz: 'xz-packed',
  zip: 'from a ZIP file',
};

const WORKBOOKS: ImportFormat[] = ['xlsx', 'xls', 'ods'];

/** "1,200 rows × 9 columns" */
export function tableSize(rows: number, columns: number): string {
  const plural = (n: number, word: string) =>
    `${n.toLocaleString('en-US')} ${word}${n === 1 ? '' : 's'}`;
  return `${plural(rows, 'row')} × ${plural(columns, 'column')}`;
}

/**
 * What the reader detected: "semicolon-separated · decimal comma · UTF-8 · header row",
 * "Excel workbook, sheet “Q1” · header row", "Parquet".
 */
export function describeReport(report: ImportReport): string {
  const format = report.format ?? 'csv';
  const parts: string[] = [];
  if (format === 'csv') {
    const delimiter = report.delimiter ?? ',';
    parts.push(
      `${DELIMITER_NAMES[delimiter] ?? `“${delimiter}”`}-separated`,
      report.decimal === ',' ? 'decimal comma' : 'decimal point',
      ENCODING_NAMES[(report.encoding ?? '').toLowerCase()] ?? report.encoding ?? 'UTF-8'
    );
  } else if (report.sheet) {
    parts.push(`${FORMAT_NAMES[format]}, sheet “${report.sheet}”`);
  } else if (report.json_path?.length) {
    parts.push(`${FORMAT_NAMES[format]}, the list under “${report.json_path.join('.')}”`);
  } else {
    parts.push(FORMAT_NAMES[format] ?? format);
  }
  if (report.compression) {
    parts.push(
      report.member ? `“${report.member}” from a ZIP file` : (PACKED[report.compression] ?? '')
    );
  }
  if (format === 'csv' || WORKBOOKS.includes(format)) {
    parts.push(report.header ? 'header row' : 'no header row');
  }
  return parts.filter(Boolean).join(' · ');
}

/** The ending of a file name, e.g. ".csv"; "" if it has none. */
function suffix(name: string): string {
  return /\.[^./\\]+$/.exec(name.toLowerCase())?.[0] ?? '';
}

/** Why `file` cannot be imported, or null. */
export function fileProblem(
  file: File,
  maxUploadMb: number,
  fileTypes: string[] = DEFAULT_FILE_TYPES
): string | null {
  if (!fileTypes.includes(suffix(file.name))) {
    return (
      `“${file.name}” cannot be imported. Choose a CSV, Excel, OpenDocument, JSON or Parquet ` +
      `file (${fileTypes.join(', ')}).`
    );
  }
  if (maxUploadMb > 0 && file.size > maxUploadMb * 1024 * 1024) {
    return `The file is ${formatBytes(file.size)}; the server accepts up to ${maxUploadMb} MB.`;
  }
  return null;
}

/** Why `url` is not a link the server can download, or null. */
export function linkProblem(url: string): string | null {
  return /^https?:\/\/[^\s/?#]+\S*$/i.test(url.trim())
    ? null
    : 'Enter a link that starts with https:// or http://.';
}

/** The name of a table from its file name: "sales" for "sales.csv.gz". */
export function fileStem(filename: string): string {
  return filename.replace(/\.(gz|bz2|xz)$/i, '').replace(/\.[^.]+$/, '') || filename;
}

/** Which import options apply: text files have all, workbooks a header row, others neither. */
function optionsFor(fileName: string | null): 'text' | 'sheet' | 'other' {
  if (fileName === null) {
    return 'text';
  }
  const inner = fileName.toLowerCase().replace(/\.(gz|bz2|xz)$/, '');
  if (/\.(xlsx|xlsm|xls|ods)$/.test(inner)) {
    return 'sheet';
  }
  if (/\.(json|geojson|jsonl|ndjson|parquet|pq|feather|arrow|arrows|ipc)$/.test(inner)) {
    return 'other';
  }
  return 'text';
}

/** The sheets of a workbook in an error answer (the requested sheet was not usable). */
function errorSheets(error: unknown): string[] | null {
  if (!(error instanceof ApiError)) {
    return null;
  }
  const sheets = (error.data as { sheets?: unknown } | null)?.sheets;
  return Array.isArray(sheets) && sheets.every((s) => typeof s === 'string') ? sheets : null;
}

type Mode = 'file' | 'link';

interface ImportDialogProps {
  opened: boolean;
  onClose: () => void;
  /** A file dropped on the page; the dialog starts with it. */
  initialFile: File | null;
  maxUploadMb: number;
  /** File name endings the server imports (from its status). */
  fileTypes?: string[];
  /** Whether the server can import files from links. */
  urlImport?: boolean;
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
 * Import a data file (CSV, Excel, JSON, Parquet …) from the computer or from a link: name it,
 * optionally override what the reader detects, choose the sheet of a workbook with several.
 * Shows the reader's warnings afterwards. Mount it with a new `key` per import.
 */
export function ImportDialog({
  opened,
  onClose,
  initialFile,
  maxUploadMb,
  fileTypes = DEFAULT_FILE_TYPES,
  urlImport = false,
  currentPrepName,
  onImported,
}: ImportDialogProps) {
  const [mode, setMode] = useState<Mode>('file');
  const [file, setFile] = useState<File | null>(initialFile);
  const [link, setLink] = useState('');
  const [name, setName] = useState(initialFile ? fileStem(initialFile.name) : '');
  const [nameEdited, setNameEdited] = useState(false);
  const [options, setOptions] = useState<ImportOptions>(DEFAULT_OPTIONS);
  const [advanced, setAdvanced] = useState(false);
  const [openNew, setOpenNew] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(
    initialFile ? fileProblem(initialFile, maxUploadMb, fileTypes) : null
  );
  /** The sheets of a workbook whose (chosen) sheet could not be imported. */
  const [sheetChoices, setSheetChoices] = useState<string[] | null>(null);
  /** Imported from a workbook with several sheets; the user confirms or picks another. */
  const [pending, setPending] = useState<DataSource | null>(null);
  const [sheet, setSheet] = useState<string | null>(null);
  const [imported, setImported] = useState<DataSource | null>(null);
  const [dragging, setDragging] = useState(false);
  const resetFile = useRef<() => void>(null);
  /** The running import; aborted when the dialog is cancelled or the page is left. */
  const upload = useRef<AbortController | null>(null);
  /** `pending` for clean-ups: a table imported only to choose a sheet is removed again. */
  const pendingSource = useRef<DataSource | null>(null);
  const progress = useUploadProgress();

  const discardPending = () => {
    const source = pendingSource.current;
    pendingSource.current = null;
    if (source) {
      api.data.sources.remove(source.id).catch(() => undefined);
    }
  };

  useEffect(
    () => () => {
      upload.current?.abort();
      const source = pendingSource.current;
      if (source) {
        api.data.sources.remove(source.id).catch(() => undefined);
      }
    },
    []
  );

  /** Close the dialog; a running import is cancelled and its result ignored. */
  const close = () => {
    upload.current?.abort();
    discardPending();
    onClose();
  };

  const choose = (next: File | null) => {
    resetFile.current?.();
    if (!next) {
      return;
    }
    setFile(next);
    setSheetChoices(null);
    setOptions((current) => ({ ...current, sheet: null }));
    setError(fileProblem(next, maxUploadMb, fileTypes));
    if (!nameEdited) {
      setName(fileStem(next.name));
    }
  };

  const set =
    <K extends keyof ImportOptions>(key: K) =>
    (value: ImportOptions[K]) =>
      setOptions((current) => ({ ...current, [key]: value }));

  const problem =
    mode === 'file'
      ? file
        ? fileProblem(file, maxUploadMb, fileTypes)
        : 'Choose a file.'
      : linkProblem(link);
  const source = mode === 'file' ? (file?.name ?? '') : link.trim();

  /** Send the file or link; null if it failed (the error is shown) or was cancelled. */
  const importSource = async (
    sheetName: string | null
  ): Promise<{ source: DataSource; controller: AbortController } | null> => {
    const controller = new AbortController();
    upload.current = controller;
    setBusy(true);
    setError(null);
    const changed = Object.fromEntries(
      Object.entries(options).filter(
        ([key, value]) => value !== DEFAULT_OPTIONS[key as keyof ImportOptions]
      )
    ) as Partial<ImportOptions>;
    const fields = {
      ...changed,
      ...(sheetName ? { sheet: sheetName } : {}),
      ...(name.trim() ? { name: name.trim() } : {}),
    };
    try {
      let result: DataSource;
      if (mode === 'file' && file) {
        progress.start(file.size);
        result = await api.data.sources.upload(file, fields, {
          signal: controller.signal,
          onUploadProgress: progress.report,
        });
      } else {
        result = await api.data.sources.importUrl(link.trim(), fields, controller.signal);
      }
      return controller.signal.aborted ? null : { source: result, controller };
    } catch (err) {
      if (!controller.signal.aborted) {
        setError((err as Error).message);
        const sheets = errorSheets(err);
        if (sheets && !pendingSource.current) {
          setSheetChoices(sheets);
          setOptions((current) => ({
            ...current,
            sheet: sheetName && sheets.includes(sheetName) ? sheetName : (sheets[0] ?? null),
          }));
        }
        setBusy(false);
        progress.stop();
      }
      return null;
    }
  };

  /** Hand the imported table over, then show the warnings or close. */
  const finish = async (result: DataSource, controller: AbortController) => {
    upload.current = controller;
    setBusy(true);
    await onImported(result, openNew, controller.signal);
    if (controller.signal.aborted) {
      return;
    }
    upload.current = null;
    setBusy(false);
    progress.stop();
    if (result.report.warnings.length > 0) {
      setImported(result);
    } else {
      onClose();
    }
  };

  const submit = async () => {
    if (busy || problem) {
      return;
    }
    const result = await importSource(options.sheet ?? null);
    if (!result) {
      return;
    }
    const sheets = result.source.report.sheets ?? [];
    if (!options.sheet && sheets.length > 1) {
      // Several sheets: let the user confirm the one that was read, or pick another.
      pendingSource.current = result.source;
      setPending(result.source);
      setSheet(result.source.report.sheet ?? sheets[0] ?? null);
      upload.current = null;
      setBusy(false);
      progress.stop();
      return;
    }
    await finish(result.source, result.controller);
  };

  const confirmSheet = async () => {
    if (!pending || busy || !sheet) {
      return;
    }
    if (sheet === pending.report.sheet) {
      pendingSource.current = null;
      await finish(pending, new AbortController());
      return;
    }
    const result = await importSource(sheet);
    if (!result) {
      return;
    }
    discardPending();
    setPending(null);
    await finish(result.source, result.controller);
  };

  const applies = optionsFor(mode === 'file' ? (file?.name ?? null) : null);
  const label = mode === 'file' ? (file?.name ?? '') : link.trim();

  return (
    <Modal
      opened={opened}
      onClose={close}
      title={imported ? 'Imported' : pending ? 'Choose the sheet' : 'Import data'}
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
      ) : pending ? (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void confirmSheet();
          }}
        >
          <Stack gap="md">
            <Text size="sm" style={{ overflowWrap: 'anywhere' }}>
              “{pending.filename || label}” has {pending.report.sheets?.length} sheets. Which one do
              you want to import?
            </Text>
            <Radio.Group value={sheet} onChange={setSheet} label="Sheet">
              <ScrollArea.Autosize mah={260} type="auto" offsetScrollbars>
                <Stack gap={8} mt={6}>
                  {(pending.report.sheets ?? []).map((sheetName) => (
                    <Radio
                      key={sheetName}
                      value={sheetName}
                      label={sheetName}
                      disabled={busy}
                      description={
                        sheetName === pending.report.sheet
                          ? `Read: ${tableSize(pending.n_rows, pending.n_columns)}`
                          : undefined
                      }
                      styles={{ label: { overflowWrap: 'anywhere' } }}
                    />
                  ))}
                </Stack>
              </ScrollArea.Autosize>
            </Radio.Group>
            {error && (
              <Alert color="red" icon={<IconAlertTriangle size={18} />} title="Cannot import">
                {error}
              </Alert>
            )}
            <ImportProgress mode={mode} label={label} busy={busy} bytes={progress.bytes} />
            <Group justify="flex-end" gap="xs">
              <Button variant="default" onClick={close}>
                Cancel
              </Button>
              <Button type="submit" loading={busy} disabled={!sheet}>
                {sheet === pending.report.sheet ? 'Continue' : 'Import this sheet'}
              </Button>
            </Group>
          </Stack>
        </form>
      ) : (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
        >
          <Stack gap="md">
            <Tabs
              value={mode}
              onChange={(value) => {
                if (value && !busy) {
                  setMode(value as Mode);
                  setError(
                    value === 'file' && file ? fileProblem(file, maxUploadMb, fileTypes) : null
                  );
                  setSheetChoices(null);
                  setOptions((current) => ({ ...current, sheet: null }));
                }
              }}
            >
              {urlImport && (
                <Tabs.List grow mb="md">
                  <Tabs.Tab value="file" leftSection={<IconUpload size={14} />} disabled={busy}>
                    File
                  </Tabs.Tab>
                  <Tabs.Tab value="link" leftSection={<IconLink size={14} />} disabled={busy}>
                    From a link
                  </Tabs.Tab>
                </Tabs.List>
              )}
              <Tabs.Panel value="file">
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
                      Drop a file here, or choose one: CSV, Excel, OpenDocument, JSON, Parquet,
                      Feather, also packed as .zip or .gz. Up to {maxUploadMb} MB.
                    </Text>
                  )}
                  <FileButton
                    resetRef={resetFile}
                    onChange={choose}
                    accept={acceptedFiles(fileTypes)}
                  >
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
              </Tabs.Panel>
              {urlImport && (
                <Tabs.Panel value="link">
                  <Stack gap="xs">
                    <TextInput
                      label="Link to the file"
                      type="url"
                      inputMode="url"
                      autoComplete="off"
                      spellCheck={false}
                      placeholder="https://…"
                      value={link}
                      onChange={(event) => {
                        setLink(event.currentTarget.value);
                        setError(null);
                        setSheetChoices(null);
                      }}
                      error={link.trim() && linkProblem(link) ? linkProblem(link) : undefined}
                      disabled={busy}
                      data-autofocus
                    />
                    <Text size="xs" c="dimmed">
                      A public link to a CSV, Excel, JSON or Parquet file, up to {maxUploadMb} MB.
                      The server downloads it. For example:
                    </Text>
                    <List size="xs" c="dimmed" spacing={2} withPadding>
                      <List.Item>
                        <b>Google Sheets:</b> the link from “Share” (shared with “Anyone with the
                        link”); the tab that is open in the link is imported.
                      </List.Item>
                      <List.Item>
                        <b>GitHub:</b> the page of a file (…/blob/main/data.csv) or its “Raw” link.
                      </List.Item>
                      <List.Item>
                        <b>Hugging Face:</b> the page of a file in a dataset
                        (…/datasets/…/blob/main/train.csv).
                      </List.Item>
                    </List>
                    <Text size="xs" c="dimmed">
                      No link at hand?{' '}
                      <Anchor
                        component="button"
                        type="button"
                        size="xs"
                        onClick={() => {
                          setLink(EXAMPLE_LINK);
                          setError(null);
                        }}
                        disabled={busy}
                      >
                        Try the iris flowers (a CSV file on GitHub)
                      </Anchor>
                    </Text>
                  </Stack>
                </Tabs.Panel>
              )}
            </Tabs>

            <TextInput
              label="Name"
              placeholder={
                mode === 'file'
                  ? file
                    ? fileStem(file.name)
                    : 'e.g. customers'
                  : 'Empty: the name of the file'
              }
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

            {sheetChoices && (
              <Select
                label="Sheet"
                description="The sheet of the workbook to import."
                data={sheetChoices}
                value={options.sheet ?? null}
                onChange={(value) => value && set('sheet')(value)}
                allowDeselect={false}
                disabled={busy}
              />
            )}

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
                    {mode === 'link' &&
                      ' Separator, decimal mark and encoding apply to CSV files only.'}
                  </Text>
                  <SimpleGrid cols={{ base: 1, xs: 2 }} spacing="sm" verticalSpacing="sm">
                    {applies === 'text' && (
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
                    )}
                    {applies === 'text' && (
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
                    )}
                    {applies !== 'other' && (
                      <Select
                        label="Header row"
                        data={HEADERS}
                        value={options.header}
                        onChange={(value) =>
                          value && set('header')(value as ImportOptions['header'])
                        }
                        allowDeselect={false}
                        disabled={busy}
                      />
                    )}
                    {applies === 'text' && (
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
                    )}
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
            <ImportProgress mode={mode} label={source} busy={busy} bytes={progress.bytes} />

            <Group justify="flex-end" gap="xs">
              <Button variant="default" onClick={close}>
                Cancel
              </Button>
              <Button type="submit" loading={busy} disabled={problem !== null}>
                Import
              </Button>
            </Group>
          </Stack>
        </form>
      )}
    </Modal>
  );
}

/** The upload progress of a file, or an indeterminate bar while the server downloads a link. */
function ImportProgress({
  mode,
  label,
  busy,
  bytes,
}: {
  mode: Mode;
  label: string;
  busy: boolean;
  bytes: ReturnType<typeof useUploadProgress>['bytes'];
}) {
  if (!busy) {
    return null;
  }
  if (mode === 'file') {
    return bytes ? (
      <UploadProgress
        bytes={bytes}
        fileName={label}
        processingText="Reading and checking the file …"
      />
    ) : null;
  }
  const text = 'Downloading and reading the file …';
  return (
    <Stack gap={6}>
      <Text size="sm" role="status">
        {text}
      </Text>
      <Progress.Root size="md">
        <Progress.Section
          value={100}
          striped
          animated
          withAria={false}
          role="progressbar"
          aria-label="Import from the link"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuetext={text}
        />
      </Progress.Root>
      <Text size="xs" c="dimmed" style={{ overflowWrap: 'anywhere' }}>
        {label}
      </Text>
    </Stack>
  );
}
