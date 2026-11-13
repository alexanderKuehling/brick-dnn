import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  IconAlertTriangle,
  IconChartHistogram,
  IconCircleCheck,
  IconDatabaseOff,
  IconFileImport,
  IconListDetails,
  IconLock,
  IconPlugConnectedX,
  IconPlus,
  IconTable,
  IconTableImport,
} from '@tabler/icons-react';
import { Link, Navigate, useLocation, useNavigate, useParams } from 'react-router-dom';
import {
  Alert,
  Badge,
  Box,
  Button,
  Center,
  Code,
  Drawer,
  Group,
  List,
  Loader,
  Stack,
  Tabs,
  Text,
  TextInput,
  ThemeIcon,
  Title,
  useMantineTheme,
} from '@mantine/core';
import { useDisclosure, useHotkeys, useMediaQuery } from '@mantine/hooks';
import { modals } from '@mantine/modals';
import { notifications } from '@mantine/notifications';
import {
  api,
  ApiError,
  type DataPrep,
  type Dataset,
  type DataSource,
  type DataStatus,
  type NewStep,
  type PrepDownloadFormat,
  type PreviewResult,
} from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { downloadBlob } from '@/builder/download';
import { HEADER_HEIGHT } from '@/components/AppHeader';
import { ShareInNetworkDialog } from '@/sharing/ShareInNetworkDialog';
import { loadSource, saveSource } from '@/training/settings';
import { AnalysisPanel } from './analysis/AnalysisPanel';
import type { RowsSource } from './analysis/rows/RowsView';
import { describeReport, ImportDialog, tableSize } from './ImportDialog';
import { PipelineEditor } from './pipeline/PipelineEditor';
import { accessOf, PrepAccessContext, prepRole } from './prepAccess';
import { PrepSidebar } from './PrepSidebar';
import { PrepToolbar, type SharedPrep, type ToolbarAction } from './PrepToolbar';
import { ReadinessPanel } from './readiness/ReadinessPanel';
import { firstFailedStep, stepSpec } from './steps';
import { DataTable } from './table/DataTable';
import { useDataPrep, type DataPrepController } from './useDataPrep';
import { useDataResources, type DataResources } from './useDataResources';
import classes from './DataPage.module.css';

type View = 'steps' | 'data' | 'analysis' | 'readiness';

const PAGE_HEIGHT = `calc(100dvh - ${HEADER_HEIGHT}px)`;
/** The layout is chosen before the first paint, so it does not jump. */
const MEDIA_OPTIONS = { getInitialValueInEffect: false };
const LAST_PREP_KEY = 'brickdnn:data-prep';

function readLastPrep(): number | null {
  try {
    const value = Number(localStorage.getItem(LAST_PREP_KEY));
    return Number.isInteger(value) && value > 0 ? value : null;
  } catch {
    return null;
  }
}

function writeLastPrep(id: number | null): void {
  try {
    if (id === null) {
      localStorage.removeItem(LAST_PREP_KEY);
    } else {
      localStorage.setItem(LAST_PREP_KEY, String(id));
    }
  } catch {
    // Storage unavailable: the page simply opens the first preparation next time.
  }
}

function parseId(value: string | undefined): number | null {
  if (value === undefined || !/^\d+$/.test(value)) {
    return null;
  }
  return Number(value);
}

/** `base`, or "base 2", "base 3" … if a preparation has that name already. */
function uniqueName(base: string, preps: DataPrep[]): string {
  const taken = new Set(preps.map((prep) => prep.name.toLowerCase()));
  let name = base;
  for (let n = 2; taken.has(name.toLowerCase()); n += 1) {
    name = `${base} ${n}`;
  }
  return name;
}

function slug(text: string): string {
  return (
    text
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '_')
      .replace(/^_+|_+$/g, '')
      .slice(0, 60) || 'prepared_data'
  );
}

const DOWNLOAD_NAMES: Record<PrepDownloadFormat, (base: string) => string> = {
  csv: (base) => `${base}.csv`,
  recipe: (base) => `${base}.recipe.json`,
  python: (base) => `${base}_prepare.py`,
};

/** "14 features, 2 classes" for an exported dataset. */
function datasetSummary(dataset: Dataset): string {
  const { num_features: features, num_classes: classes_, target_kind: kind } = dataset.info;
  const parts: string[] = [];
  if (typeof features === 'number') {
    parts.push(`${features} ${features === 1 ? 'feature' : 'features'}`);
  }
  if (kind === 'number') {
    parts.push('numeric target');
  } else if (typeof classes_ === 'number') {
    parts.push(`${classes_} classes`);
  }
  return parts.join(', ');
}

/**
 * The Data tab: import data files, prepare them step by step with a live preview and analysis,
 * and turn the result into a training dataset.
 */
export function DataPage() {
  const { user, ready } = useAuth();
  const resources = useDataResources(Boolean(user));
  const { status } = resources;

  if (!status) {
    return (
      <Center h={PAGE_HEIGHT} px="md">
        {resources.statusError ? (
          <Message
            icon={<IconPlugConnectedX size={24} />}
            color="red"
            title="Cannot reach the backend"
            actions={<Button onClick={resources.reloadStatus}>Try again</Button>}
          >
            {resources.statusError}
          </Message>
        ) : (
          <Loader />
        )}
      </Center>
    );
  }
  if (!status.available) {
    return <Unavailable status={status} onRetry={resources.reloadStatus} />;
  }
  if (!ready) {
    return (
      <Center h={PAGE_HEIGHT}>
        <Loader />
      </Center>
    );
  }
  if (!user) {
    return <LoggedOut />;
  }
  return <Workspace resources={resources} status={status} />;
}

// --------------------------------------------------------------------------
// States without a workspace
// --------------------------------------------------------------------------

function Message({
  icon,
  color,
  title,
  children,
  actions,
}: {
  icon: ReactNode;
  color?: string;
  title: string;
  children: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <Stack align="center" ta="center" gap="sm" maw={480} mx="auto">
      <ThemeIcon size={48} radius="xl" variant="light" color={color}>
        {icon}
      </ThemeIcon>
      <Title order={3}>{title}</Title>
      <Text size="sm" c="dimmed" component="div">
        {children}
      </Text>
      {actions && (
        <Group justify="center" gap="xs" mt="xs">
          {actions}
        </Group>
      )}
    </Stack>
  );
}

function LoggedOut() {
  const { pathname } = useLocation();
  return (
    <Center h={PAGE_HEIGHT} px="md">
      <Message
        icon={<IconLock size={24} />}
        title="Log in to prepare data"
        actions={
          <>
            <Button component={Link} to="/login" state={{ from: pathname }}>
              Log in
            </Button>
            <Button component={Link} to="/register" state={{ from: pathname }} variant="default">
              Sign up
            </Button>
          </>
        }
      >
        Import CSV, Excel or JSON files, clean and combine them step by step, see what is in them,
        and turn them into training data. Your files are stored on the Brick DNN server, so this
        needs a free account, just like saving and training.
      </Message>
    </Center>
  );
}

function Unavailable({ status, onRetry }: { status: DataStatus; onRetry: () => void }) {
  const missingPackages = status.enabled;
  return (
    <Center h={PAGE_HEIGHT} px="md">
      <Message
        icon={<IconDatabaseOff size={24} />}
        color={missingPackages ? 'yellow' : 'gray'}
        title="Data preparation is not available"
        actions={
          missingPackages ? (
            <Button variant="light" onClick={onRetry}>
              Check again
            </Button>
          ) : undefined
        }
      >
        <Text size="sm">
          {status.reason ??
            (status.enabled
              ? 'pandas is not installed on the server.'
              : 'Data preparation is switched off on this server.')}
        </Text>
        {missingPackages && (
          <Stack gap={6} mt="md" ta="left">
            <Text size="sm">
              It needs pandas in the backend environment. In the <Code>backend</Code> folder, with
              the virtual environment active:
            </Text>
            <Code block>pip install -r requirements.txt</Code>
            <Text size="sm">Then restart the backend and check again.</Text>
          </Stack>
        )}
      </Message>
    </Center>
  );
}

// --------------------------------------------------------------------------
// Workspace
// --------------------------------------------------------------------------

interface WorkspaceProps {
  resources: DataResources;
  status: DataStatus;
}

function Workspace({ resources, status }: WorkspaceProps) {
  const params = useParams();
  const navigate = useNavigate();
  const theme = useMantineTheme();
  const wide = useMediaQuery(`(min-width: ${theme.breakpoints.lg})`, true, MEDIA_OPTIONS);
  const split = useMediaQuery(`(min-width: ${theme.breakpoints.md})`, true, MEDIA_OPTIONS);
  const prepId = parseId(params.prepId);
  const invalidId = params.prepId !== undefined && prepId === null;
  const { catalog, preps, sources, putPrep, dropPrep, putSource, dropSource } = resources;
  const prep = useDataPrep(prepId, { catalog, onSaved: putPrep });

  const [drawerOpened, drawer] = useDisclosure(false);
  const [importing, setImporting] = useState<{ key: number; file: File | null } | null>(null);
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);
  const [busy, setBusy] = useState<ToolbarAction | null>(null);
  const busyRef = useRef(false);
  const [view, setView] = useState<View>('data');
  const [visited, setVisited] = useState<ReadonlySet<View>>(() => new Set<View>(['data']));
  const [column, setColumn] = useState<string | null>(null);
  // Work that finishes after the page was left (an import, a new preparation) does not
  // navigate any more.
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const loaded = prep.prep;
  const [sharingPrep, setSharingPrep] = useState<DataPrep | null>(null);
  // The source of a preparation shared with the user is not in their own list.
  const sourceId = loaded?.source ?? null;
  const listedSource = useMemo(
    () => sources?.find((s) => s.id === sourceId) ?? null,
    [sources, sourceId]
  );
  const foreign = loaded !== null && prepRole(loaded) !== 'owner';
  const [foreignSource, setForeignSource] = useState<DataSource | null>(null);
  useEffect(() => {
    if (!foreign || sourceId === null || listedSource) {
      return undefined;
    }
    const controller = new AbortController();
    api.data.sources
      .get(sourceId, controller.signal)
      .then((result) => !controller.signal.aborted && setForeignSource(result))
      .catch(() => undefined); // The table details (rows, columns) are missing then.
    return () => controller.abort();
  }, [foreign, sourceId, listedSource]);
  const source = listedSource ?? (foreignSource?.id === sourceId ? foreignSource : null);
  const access = useMemo(() => accessOf(loaded), [loaded?.role, loaded?.owner]);
  const shared: SharedPrep | null =
    loaded && prepRole(loaded) !== 'owner'
      ? {
          owner: loaded.owner,
          role: prepRole(loaded) === 'editor' ? 'editor' : 'viewer',
          projects: loaded.shared_in,
        }
      : null;

  // ---------------------------------------------------------------- navigation state
  useEffect(() => {
    if (loaded) {
      writeLastPrep(loaded.id);
    }
  }, [loaded?.id]);

  useEffect(() => {
    if (prep.notFound && prepId !== null) {
      if (readLastPrep() === prepId) {
        writeLastPrep(null);
      }
      dropPrep(prepId);
    }
  }, [prep.notFound, prepId, dropPrep]);

  useEffect(() => setColumn(null), [prepId]);

  // Changes are saved automatically; Ctrl+S saves at once (also while typing in a field)
  // instead of opening the browser's "Save page" dialog.
  useHotkeys([['mod+S', () => prep.flush().catch(() => undefined)]], []);

  const showView = useCallback((next: View) => {
    setView(next);
    setVisited((current) => (current.has(next) ? current : new Set([...current, next])));
  }, []);

  // ---------------------------------------------------------------- import
  const openImport = (file: File | null = null) => {
    drawer.close();
    setImporting({ key: Date.now(), file });
  };

  /** `signal`: do not open the new preparation if it is aborted meanwhile. */
  const createPrep = async (
    from: { id: number; name: string },
    copy?: Pick<DataPrep, 'name' | 'steps' | 'settings'>,
    signal?: AbortSignal
  ) => {
    const name = uniqueName(copy ? `${copy.name} (copy)` : from.name, preps ?? []);
    try {
      const created = await api.data.preps.create({
        name,
        source: from.id,
        ...(copy ? { steps: copy.steps, settings: copy.settings } : {}),
      });
      putPrep(created);
      if (!mounted.current || signal?.aborted) {
        return;
      }
      drawer.close();
      navigate(`/data/${created.id}`);
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Cannot create the preparation',
        message: (error as Error).message,
      });
    }
  };

  const onImported = async (imported: DataSource, openNew: boolean, signal: AbortSignal) => {
    putSource(imported);
    if (imported.report.warnings.length === 0) {
      notifications.show({
        color: 'green',
        title: `Imported “${imported.name}”`,
        message: `${tableSize(imported.n_rows, imported.n_columns)}, read as ${describeReport(imported.report)}.`,
      });
    }
    if (openNew) {
      await createPrep(imported, undefined, signal);
    }
  };

  const hasFiles = (event: React.DragEvent) =>
    Array.from(event.dataTransfer.types).includes('Files');
  const dropHandlers = {
    onDragEnter: (event: React.DragEvent) => {
      if (!hasFiles(event) || importing) {
        return;
      }
      event.preventDefault();
      dragDepth.current += 1;
      setDragging(true);
    },
    onDragLeave: (event: React.DragEvent) => {
      if (!hasFiles(event) || importing) {
        return;
      }
      dragDepth.current -= 1;
      if (dragDepth.current <= 0) {
        dragDepth.current = 0;
        setDragging(false);
      }
    },
    onDragOver: (event: React.DragEvent) => {
      if (hasFiles(event) && !importing) {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'copy';
      }
    },
    onDrop: (event: React.DragEvent) => {
      if (!hasFiles(event) || importing) {
        return;
      }
      event.preventDefault();
      dragDepth.current = 0;
      setDragging(false);
      const file = event.dataTransfer.files[0];
      if (file) {
        openImport(file);
      }
    },
  };

  // ---------------------------------------------------------------- preparations
  const renamePrep = (target: DataPrep) => {
    const current = target.id === prepId;
    let value = current ? prep.name : target.name;
    modals.openConfirmModal({
      title: 'Rename preparation',
      children: (
        <TextInput
          data-autofocus
          label="Name"
          defaultValue={value}
          maxLength={120}
          onChange={(event) => {
            value = event.currentTarget.value;
          }}
        />
      ),
      labels: { confirm: 'Save', cancel: 'Cancel' },
      onConfirm: async () => {
        const name = value.trim();
        if (!name) {
          return;
        }
        if (current) {
          prep.setName(name);
          return;
        }
        try {
          putPrep(await api.data.preps.update(target.id, { name }));
        } catch (error) {
          notifications.show({
            color: 'red',
            title: 'Cannot rename the preparation',
            message: (error as Error).message,
          });
        }
      },
    });
  };

  const duplicatePrep = (target: DataPrep) => {
    const from = { id: target.source, name: target.source_name };
    if (target.id === prepId && prep.settings) {
      void createPrep(from, { name: prep.name, steps: prep.steps, settings: prep.settings });
    } else {
      void createPrep(from, target);
    }
  };

  const deletePrep = (target: DataPrep) => {
    const current = target.id === prepId;
    const name = current ? prep.name : target.name;
    modals.openConfirmModal({
      title: `Delete “${name}”?`,
      children: (
        <Text size="sm">
          Its steps and settings are deleted. The imported tables and the training datasets made
          from it are kept.
        </Text>
      ),
      labels: { confirm: 'Delete', cancel: 'Cancel' },
      confirmProps: { color: 'red' },
      onConfirm: async () => {
        try {
          if (current) {
            // Keep the changes if deleting fails.
            await prep.flush().catch(() => undefined);
          }
          await api.data.preps.remove(target.id);
        } catch (error) {
          if (!(error instanceof ApiError && error.status === 404)) {
            notifications.show({
              color: 'red',
              title: 'Cannot delete the preparation',
              message: (error as Error).message,
            });
            return;
          }
        }
        dropPrep(target.id);
        if (current) {
          prep.discard();
          writeLastPrep(null);
          navigate('/data', { replace: true });
        }
      },
    });
  };

  const sourceRenamed = (renamed: DataSource) => {
    putSource(renamed);
    preps
      ?.filter((p) => p.source === renamed.id)
      .forEach((p) => putPrep({ ...p, source_name: renamed.name }));
  };

  // ---------------------------------------------------------------- export & downloads
  /** The readiness of the full pipeline, if the preview is up to date. */
  const freshReadiness =
    prep.preview && !prep.previewing && !prep.previewError && prep.upto === null
      ? prep.preview.readiness
      : null;

  const run = async (action: ToolbarAction, task: () => Promise<void>) => {
    if (busyRef.current || prepId === null) {
      return;
    }
    busyRef.current = true;
    setBusy(action);
    try {
      await task();
    } finally {
      busyRef.current = false;
      setBusy(null);
    }
  };

  const saveFirst = async (): Promise<boolean> => {
    try {
      await prep.flush();
      return true;
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Cannot save the preparation',
        message: (error as Error).message,
      });
      return false;
    }
  };

  const trainWith = (dataset: Dataset) => {
    // Remembered for a builder that is opened now; one that is open already (the
    // notification outlives this page) gets the dataset with the navigation state.
    saveSource({ ...loadSource(), tab: 'upload', upload: dataset.id });
    navigate('/', { state: { openTraining: true, dataset: dataset.id } });
    const summary = datasetSummary(dataset);
    notifications.show({
      color: 'green',
      title: 'Training data selected',
      message: `“${dataset.name}” is selected in the training panel${summary ? ` (${summary})` : ''}.`,
    });
  };

  const exportDataset = (then: 'notify' | 'train') =>
    run(then === 'train' ? 'train' : 'export', async () => {
      if (freshReadiness && !freshReadiness.ready) {
        const errors = freshReadiness.issues.filter((issue) => issue.severity === 'error').length;
        showView('readiness');
        notifications.show({
          color: 'orange',
          title: 'Not ready for training yet',
          message:
            errors === 1
              ? 'Fix the problem listed under Readiness first.'
              : `Fix the ${errors} problems listed under Readiness first.`,
        });
        return;
      }
      if (!(await saveFirst())) {
        return;
      }
      let dataset: Dataset;
      try {
        dataset = await api.data.preps.export(prepId!);
      } catch (error) {
        // The server checks the readiness again and lists the problems.
        const notReady =
          error instanceof ApiError &&
          error.status === 400 &&
          Array.isArray((error.data as { issues?: unknown } | null)?.issues);
        if (notReady) {
          showView('readiness');
        }
        notifications.show({
          color: 'red',
          title: notReady ? 'Not ready for training' : 'Cannot create the training dataset',
          message: (error as Error).message,
        });
        return;
      }
      prep.addDataset(dataset);
      if (loaded) {
        const listed = preps?.find((p) => p.id === loaded.id) ?? loaded;
        putPrep({
          ...listed,
          datasets: [
            { id: dataset.id, name: dataset.name, created_at: dataset.created_at },
            ...listed.datasets,
          ],
        });
      }
      if (then === 'train') {
        trainWith(dataset);
        return;
      }
      const id = `dataset-${dataset.id}`;
      notifications.show({
        id,
        color: 'green',
        title: 'Training dataset created',
        autoClose: 10000,
        message: (
          <>
            <Text size="sm">
              “{dataset.name}” is listed under “Your data” in the training panel
              {dataset.mine === false &&
                ` (shared with you: it belongs to ${dataset.owner}, like the preparation)`}
              .
            </Text>
            <Button
              size="compact-xs"
              variant="light"
              mt={6}
              onClick={() => {
                notifications.hide(id);
                trainWith(dataset);
              }}
            >
              Train with it
            </Button>
          </>
        ),
      });
    });

  const download = (format: PrepDownloadFormat) =>
    run(format, async () => {
      if (!(await saveFirst())) {
        return;
      }
      try {
        const file = await api.data.preps.download(prepId!, format);
        downloadBlob(file.filename ?? DOWNLOAD_NAMES[format](slug(prep.name)), file.blob);
      } catch (error) {
        notifications.show({
          color: 'red',
          title: 'Download failed',
          message: (error as Error).message,
        });
      }
    });

  // ---------------------------------------------------------------- editing from the panels
  // On small screens the step list is on another tab, so added steps are confirmed (once
  // for all steps added together, e.g. by "Fix all").
  const added = useRef<string[]>([]);
  const addStep = (step: NewStep) => {
    const id = prep.addStep(step);
    if (!id || !catalog || split) {
      return;
    }
    added.current.push(stepSpec(catalog, step.type)?.label ?? step.type);
    if (added.current.length === 1) {
      queueMicrotask(() => {
        const labels = added.current;
        added.current = [];
        notifications.show({
          message:
            labels.length === 1
              ? `Added “${labels[0]}” to the steps.`
              : `Added ${labels.length} steps.`,
          autoClose: 2500,
        });
      });
    }
  };

  const setTarget = (target: string) => {
    if (prep.settings) {
      prep.setSettings({ ...prep.settings, target });
    }
  };

  const showColumn = (name: string) => {
    setColumn(name);
    showView('analysis');
  };

  // ---------------------------------------------------------------- render
  const sidebar = (
    <PrepSidebar
      preps={preps}
      sources={sources}
      error={resources.listsError}
      onRetry={resources.retry}
      currentId={prepId}
      currentName={loaded ? prep.name : null}
      currentSourceId={loaded?.source ?? null}
      onNavigate={drawer.close}
      onImport={() => openImport()}
      onNewPrep={(s) => void createPrep(s)}
      onRenamePrep={renamePrep}
      onDuplicatePrep={duplicatePrep}
      onDeletePrep={deletePrep}
      onSharePrep={setSharingPrep}
      onSourceRenamed={sourceRenamed}
      onSourceDeleted={dropSource}
    />
  );

  let main: ReactNode;
  if (invalidId || prep.notFound) {
    main = <NotFound />;
  } else if (prepId === null) {
    if (resources.listsError) {
      main = (
        <Center className={classes.empty}>
          <Message
            icon={<IconAlertTriangle size={24} />}
            color="red"
            title="Cannot load your data"
            actions={<Button onClick={resources.retry}>Try again</Button>}
          >
            {resources.listsError}
          </Message>
        </Center>
      );
    } else if (preps === null || sources === null) {
      main = (
        <Center className={classes.empty}>
          <Loader />
        </Center>
      );
    } else if (preps.length > 0) {
      const last = readLastPrep();
      const target =
        preps.find((p) => p.id === last) ?? preps.find((p) => prepRole(p) === 'owner') ?? preps[0];
      main = <Navigate replace to={`/data/${target.id}`} />;
    } else {
      main = (
        <Welcome
          sources={sources}
          maxUploadMb={status.max_upload_mb}
          onImport={() => openImport()}
          onNewPrep={(s) => void createPrep(s)}
        />
      );
    }
  } else if (prep.loadError) {
    main = (
      <Center className={classes.empty}>
        <Message
          icon={<IconAlertTriangle size={24} />}
          color="red"
          title="Cannot load the preparation"
          actions={<Button onClick={prep.reload}>Try again</Button>}
        >
          {prep.loadError}
        </Message>
      </Center>
    );
  } else if (!loaded || !prep.settings) {
    main = (
      <Center className={classes.empty}>
        <Loader />
      </Center>
    );
  } else {
    main = (
      <PrepAccessContext.Provider value={access}>
        <PrepToolbar
          shared={shared}
          name={prep.name}
          onNameChange={prep.setName}
          source={source}
          sourceName={source?.name ?? loaded.source_name}
          saveStatus={prep.saveStatus}
          saveError={prep.saveError}
          onRetrySave={prep.retrySave}
          busy={busy}
          onDownload={(format) => void download(format)}
          onExport={() => void exportDataset('notify')}
          onTrain={() => void exportDataset('train')}
          onOpenSidebar={wide ? null : drawer.open}
        />
        <PrepContent
          key={prepId}
          prep={prep}
          resources={resources}
          source={source}
          // Combine steps read the owner's tables: others only see the main one.
          editorSources={foreign ? (source ? [source] : []) : (sources ?? [])}
          split={split}
          view={view}
          visited={visited}
          onViewChange={showView}
          column={column}
          onColumnChange={setColumn}
          onAddStep={addStep}
          onSetTarget={setTarget}
          onShowColumn={showColumn}
          onExport={() => void exportDataset('notify')}
          exporting={busy === 'export'}
        />
      </PrepAccessContext.Provider>
    );
  }

  return (
    <Box className={classes.page} style={{ height: PAGE_HEIGHT }} {...dropHandlers}>
      {wide && (
        <aside className={classes.sidebar} aria-label="Preparations and tables">
          {sidebar}
        </aside>
      )}
      <div className={classes.main}>{main}</div>

      {!wide && (
        <Drawer
          opened={drawerOpened}
          onClose={drawer.close}
          title="Data"
          size={300}
          classNames={{ body: classes.drawerBody }}
        >
          {sidebar}
        </Drawer>
      )}
      {importing && (
        <ImportDialog
          key={importing.key}
          opened
          onClose={() => setImporting(null)}
          initialFile={importing.file}
          maxUploadMb={status.max_upload_mb}
          fileTypes={status.file_types}
          urlImport={status.url_import ?? false}
          currentPrepName={loaded ? prep.name : null}
          onImported={onImported}
        />
      )}
      {sharingPrep && (
        <ShareInNetworkDialog
          kind="prep"
          item={sharingPrep}
          datasets={sharingPrep.datasets}
          onClose={() => setSharingPrep(null)}
          onChange={(sharedIn) => {
            const listed = preps?.find((p) => p.id === sharingPrep.id) ?? sharingPrep;
            putPrep({ ...listed, shared_in: sharedIn });
          }}
        />
      )}
      {dragging && (
        <div className={classes.dropOverlay} aria-hidden>
          <Stack align="center" gap="xs">
            <IconTableImport size={36} color="var(--mantine-primary-color-filled)" />
            <Text fw={600}>Drop the file to import it</Text>
          </Stack>
        </div>
      )}
    </Box>
  );
}

function NotFound() {
  return (
    <Center className={classes.empty}>
      <Message
        icon={<IconAlertTriangle size={24} />}
        color="gray"
        title="Preparation not found"
        actions={
          <Button component={Link} to="/data" replace>
            Show my preparations
          </Button>
        }
      >
        This preparation does not exist (any more).
      </Message>
    </Center>
  );
}

function Welcome({
  sources,
  maxUploadMb,
  onImport,
  onNewPrep,
}: {
  sources: DataSource[];
  maxUploadMb: number;
  onImport: () => void;
  onNewPrep: (source: DataSource) => void;
}) {
  return (
    <div className={classes.empty}>
      <Message
        icon={<IconTableImport size={24} />}
        title="Prepare data for training"
        actions={
          <Button leftSection={<IconFileImport size={16} />} onClick={onImport}>
            Import data
          </Button>
        }
      >
        <Text size="sm" c="dimmed">
          Import a CSV, Excel, JSON or Parquet file (up to {maxUploadMb} MB), drop it here, or
          import it from a link. Separators, decimal commas, dates and missing values are detected.
          Then clean and combine the data step by step, see what is in it, and turn it into a
          training dataset.
        </Text>
        {sources.length > 0 && (
          <Stack gap={6} mt="lg" align="center">
            <Text size="sm">Or start with a table you imported:</Text>
            <Group gap="xs" justify="center">
              {sources.map((s) => (
                <Button
                  key={s.id}
                  size="xs"
                  variant="light"
                  leftSection={<IconPlus size={14} />}
                  onClick={() => onNewPrep(s)}
                >
                  {s.name}
                </Button>
              ))}
            </Group>
          </Stack>
        )}
      </Message>
    </div>
  );
}

// --------------------------------------------------------------------------
// Editor and results
// --------------------------------------------------------------------------

interface PrepContentProps {
  prep: DataPrepController;
  resources: DataResources;
  source: DataSource | null;
  /** The tables combine steps can choose from. */
  editorSources: DataSource[];
  /** Editor and results side by side (otherwise as tabs). */
  split: boolean;
  view: View;
  visited: ReadonlySet<View>;
  onViewChange: (view: View) => void;
  column: string | null;
  onColumnChange: (column: string | null) => void;
  onAddStep: (step: NewStep) => void;
  onSetTarget: (column: string) => void;
  onShowColumn: (column: string) => void;
  onExport: () => void;
  exporting: boolean;
}

function PrepContent(props: PrepContentProps) {
  const { prep, resources, source, split, view, visited, onViewChange } = props;
  const { catalog, catalogError } = resources;
  const result = prep.preview;
  const settings = prep.settings!;
  const sourceColumns = useMemo(
    () => source?.columns.map(({ name, kind }) => ({ name, kind })) ?? [],
    [source]
  );
  const tab: View = split && view === 'steps' ? 'data' : view;
  const errors = result?.readiness.issues.filter((issue) => issue.severity === 'error').length;
  // A failed step blocks the export even while the readiness shown is older than that.
  const failedIndex = firstFailedStep(prep.steps, prep.reports);
  const exportBlocked =
    failedIndex >= 0 && catalog
      ? `Step ${failedIndex + 1} “${stepSpec(catalog, prep.steps[failedIndex].type)?.label ?? prep.steps[failedIndex].type}” failed.`
      : failedIndex >= 0
        ? `Step ${failedIndex + 1} failed.`
        : null;
  // The last result stays visible while a new one is computed, slightly faded.
  const stale = (prep.previewing && result !== null) || undefined;
  // The Rows view analyses what the preview shows: the same steps, settings and step.
  const pipelineNow = useRef({ steps: prep.steps, settings, upto: prep.upto });
  pipelineNow.current = { steps: prep.steps, settings, upto: prep.upto };
  const prepId = prep.prep?.id ?? null;
  const changeSplit = prep.changeSplit;
  const rowsSource = useMemo<RowsSource | undefined>(
    () =>
      prepId === null
        ? undefined
        : {
            revision: result,
            load: (body, signal) =>
              api.data.preps.rows(prepId, { ...pipelineNow.current, ...body }, signal),
            similar: (body, signal) =>
              api.data.preps.similar(prepId, { ...pipelineNow.current, ...body }, signal),
            onChangeSplit: changeSplit,
          },
    [prepId, result, changeSplit]
  );

  const editor = catalog ? (
    <PipelineEditor
      catalog={catalog}
      steps={prep.steps}
      onStepsChange={prep.setSteps}
      settings={settings}
      onSettingsChange={prep.setSettings}
      reports={prep.reports}
      sourceColumns={sourceColumns}
      sources={props.editorSources}
      currentSourceId={prep.prep!.source}
      upto={prep.upto}
      onUptoChange={prep.setUpto}
      previewing={prep.previewing}
      task={result?.task ?? null}
      splitCounts={result?.profile.split_counts ?? result?.readiness.split_counts ?? null}
      splitInfo={result?.split ?? null}
    />
  ) : catalogError ? (
    <Alert color="red" icon={<IconAlertTriangle size={18} />} title="Cannot load the step types">
      <Text size="sm">{catalogError}</Text>
      <Button size="compact-sm" variant="light" color="red" mt="xs" onClick={resources.retry}>
        Try again
      </Button>
    </Alert>
  ) : (
    <Center py="xl">
      <Loader size="sm" />
    </Center>
  );

  const body = (content: (preview: PreviewResult) => ReactNode, padded = true) => {
    if (!result) {
      return prep.previewError ? (
        <Box p="sm">
          <PreviewError message={prep.previewError} onRetry={prep.retryPreview} />
        </Box>
      ) : (
        <Center py={60}>
          <Stack align="center" gap="xs">
            <Loader />
            <Text size="sm" c="dimmed">
              Running the preparation …
            </Text>
          </Stack>
        </Center>
      );
    }
    return padded ? <div className={classes.panelPadded}>{content(result)}</div> : content(result);
  };

  const tabs = (
    <Tabs
      value={tab}
      onChange={(value) => value && onViewChange(value as View)}
      keepMounted={false}
      className={classes.tabs}
      classNames={{ tab: classes.tab }}
    >
      <div className={classes.tabBar}>
        <Tabs.List className={classes.tabList}>
          {!split && (
            <Tabs.Tab
              value="steps"
              leftSection={<IconListDetails size={14} />}
              rightSection={
                <Badge size="xs" variant="light" color="gray" circle>
                  {prep.steps.length}
                </Badge>
              }
            >
              Steps
            </Tabs.Tab>
          )}
          <Tabs.Tab value="data" leftSection={<IconTable size={14} />}>
            Data
          </Tabs.Tab>
          <Tabs.Tab value="analysis" leftSection={<IconChartHistogram size={14} />}>
            Analysis
          </Tabs.Tab>
          <Tabs.Tab
            value="readiness"
            leftSection={<IconCircleCheck size={14} />}
            aria-label={
              errors ? `Readiness, ${errors} ${errors === 1 ? 'problem' : 'problems'}` : undefined
            }
            rightSection={
              errors ? (
                <Badge size="xs" color="red" circle>
                  {errors}
                </Badge>
              ) : result?.readiness.ready && !exportBlocked ? (
                <IconCircleCheck size={14} color="var(--mantine-color-green-6)" aria-hidden />
              ) : undefined
            }
          >
            Readiness
          </Tabs.Tab>
        </Tabs.List>
        <PreviewStatus prep={prep} />
      </div>

      {(split || tab !== 'steps') && <Notices {...props} />}

      {!split && (
        <Tabs.Panel value="steps" keepMounted={visited.has('steps')} className={classes.panel}>
          <div className={classes.editorInner}>{editor}</div>
        </Tabs.Panel>
      )}
      <Tabs.Panel
        value="data"
        keepMounted={visited.has('data')}
        className={classes.panel}
        data-stale={stale}
      >
        {body(
          (preview) => (
            <DataTable
              sample={preview.sample}
              profile={preview.profile}
              target={settings.target}
              onAddStep={props.onAddStep}
              onSetTarget={props.onSetTarget}
              onShowColumn={props.onShowColumn}
            />
          ),
          false
        )}
      </Tabs.Panel>
      <Tabs.Panel
        value="analysis"
        keepMounted={visited.has('analysis')}
        className={classes.panel}
        data-stale={stale}
      >
        {body((preview) => (
          <AnalysisPanel
            profile={preview.profile}
            before={preview.before}
            sample={preview.sample}
            target={settings.target}
            onRequestCompare={prep.requestCompare}
            onAddStep={props.onAddStep}
            column={props.column}
            onColumnChange={props.onColumnChange}
            rows={rowsSource}
          />
        ))}
      </Tabs.Panel>
      <Tabs.Panel
        value="readiness"
        keepMounted={visited.has('readiness')}
        className={classes.panel}
        data-stale={stale}
      >
        {body((preview) => (
          <ReadinessPanel
            readiness={preview.readiness}
            onAddStep={props.onAddStep}
            onChangeSplit={prep.changeSplit}
            onExport={props.onExport}
            exporting={props.exporting}
            fixesDisabled={prep.previewing}
            exportBlocked={exportBlocked}
          />
        ))}
      </Tabs.Panel>
    </Tabs>
  );

  if (!split) {
    return <div className={classes.results}>{tabs}</div>;
  }
  return (
    <div className={classes.split}>
      <section className={classes.editor} aria-label="Steps">
        <div className={classes.editorInner}>{editor}</div>
      </section>
      <section className={classes.results} aria-label="Preview">
        {tabs}
      </section>
    </div>
  );
}

function PreviewStatus({ prep }: { prep: DataPrepController }) {
  const result = prep.preview;
  if (prep.previewing && result) {
    return (
      <Group gap={6} wrap="nowrap" ml="auto" style={{ flexShrink: 0 }}>
        <Loader size={12} />
        <Text size="xs" c="dimmed" visibleFrom="xs">
          Updating …
        </Text>
      </Group>
    );
  }
  if (!result) {
    return null;
  }
  return (
    <Text size="xs" c="dimmed" ml="auto" visibleFrom="sm" style={{ flexShrink: 0 }}>
      {tableSize(result.profile.n_rows, result.profile.n_columns)}
    </Text>
  );
}

function PreviewError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <Alert color="red" icon={<IconAlertTriangle size={18} />} title="The preview failed">
      <Text size="sm">{message}</Text>
      <Button size="compact-sm" variant="light" color="red" mt="xs" onClick={onRetry}>
        Try again
      </Button>
    </Alert>
  );
}

/** What the user should know about the shown preview. */
function Notices({ prep, resources, split, onViewChange }: PrepContentProps) {
  const result = prep.preview;
  if (!result) {
    return null;
  }
  const { catalog } = resources;
  const label = (type: string) => (catalog && stepSpec(catalog, type)?.label) || type;
  const uptoIndex = prep.upto === null ? -1 : prep.steps.findIndex((step) => step.id === prep.upto);
  // Steps after the previewed one did not run; their reports are older.
  const failedIndex = firstFailedStep(
    uptoIndex >= 0 ? prep.steps.slice(0, uptoIndex + 1) : prep.steps,
    prep.reports
  );
  const failed = failedIndex >= 0 ? prep.steps[failedIndex] : null;
  const showSteps = !split && (
    <Button size="compact-xs" variant="light" mt={6} onClick={() => onViewChange('steps')}>
      Show steps
    </Button>
  );

  return (
    <div className={classes.notices}>
      {prep.previewError && (
        <Alert color="red" icon={<IconAlertTriangle size={16} />} p="xs">
          <Group gap="xs" justify="space-between">
            <Text size="sm">The preview could not be updated: {prep.previewError}</Text>
            <Button size="compact-xs" variant="light" color="red" onClick={prep.retryPreview}>
              Try again
            </Button>
          </Group>
        </Alert>
      )}
      {failed && (
        <Alert color="red" icon={<IconAlertTriangle size={16} />} p="xs">
          <Text size="sm">
            Step {failedIndex + 1} “{label(failed.type)}” failed, so the steps after it did not run.
            The preview shows the data before it.
          </Text>
          {showSteps}
        </Alert>
      )}
      {uptoIndex >= 0 && !failed && (
        <Alert color="blue" p="xs">
          <Group gap="xs" justify="space-between">
            <Text size="sm">
              Showing the data after step {uptoIndex + 1} “{label(prep.steps[uptoIndex].type)}”.
            </Text>
            <Button size="compact-xs" variant="light" onClick={() => prep.setUpto(null)}>
              Show the result of all steps
            </Button>
          </Group>
        </Alert>
      )}
      {result.warnings.length > 0 && (
        <Alert color="yellow" icon={<IconAlertTriangle size={16} />} p="xs">
          {result.warnings.length === 1 ? (
            <Text size="sm">{result.warnings[0]}</Text>
          ) : (
            <List size="sm" spacing={2}>
              {result.warnings.map((warning) => (
                <List.Item key={warning}>{warning}</List.Item>
              ))}
            </List>
          )}
        </Alert>
      )}
    </div>
  );
}
