import { useState } from 'react';
import {
  IconAlertTriangle,
  IconArrowsMaximize,
  IconArrowsMinimize,
  IconEye,
  IconHistory,
  IconLock,
  IconPlayerPlay,
  IconPlugConnectedX,
} from '@tabler/icons-react';
import {
  ActionIcon,
  Alert,
  Button,
  Center,
  Code,
  Drawer,
  Group,
  Loader,
  Stack,
  Tabs,
  Text,
  ThemeIcon,
  Title,
  Tooltip,
  useMantineTheme,
} from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import type { Analysis, GraphDocument } from '@/api';
import { useReturnFocusOnClose } from '@/components/useReturnFocus';
import { i18n, Trans, useTranslation } from '@/i18n';
import { canTrain, otherStarter, trainingProject, type RunFilter } from './access';
import { canStartTraining, trainingFamily, unavailableReason, type Family } from './classical';
import { HistoryView } from './HistoryView';
import { RunView } from './RunView';
import { SetupView } from './SetupView';
import type { DatasetFitController } from './useDatasetFit';
import type { TrainingController } from './useTraining';
import layout from './RunLayout.module.css';

/** Set while the panel is shown at the full width of the window (per browser). */
export const WIDE_PANEL_KEY = 'brickdnn:training-panel-wide';

function readWide(): boolean {
  try {
    return localStorage.getItem(WIDE_PANEL_KEY) === '1';
  } catch {
    return false;
  }
}

function writeWide(wide: boolean): void {
  try {
    if (wide) {
      localStorage.setItem(WIDE_PANEL_KEY, '1');
    } else {
      localStorage.removeItem(WIDE_PANEL_KEY);
    }
  } catch {
    // Storage unavailable: the choice holds until the page is reloaded.
  }
}

export interface TrainingDrawerProps {
  training: TrainingController;
  loggedIn: boolean;
  doc: GraphDocument;
  networkName: string;
  analysis: Analysis | null;
  onLogin: () => void;
  onShowProblems: () => void;
  onLoadNetwork: (graph: GraphDocument, name: string) => void;
  /** The builder's check against the chosen dataset and the "Adapt network" action. */
  fit?: DatasetFitController | null;
}

export default function TrainingDrawer(props: TrainingDrawerProps) {
  const { t } = useTranslation();
  const { training } = props;
  const theme = useMantineTheme();
  const phone = useMediaQuery(`(max-width: ${theme.breakpoints.sm})`);
  // Kept while the drawer is closed (its content is not).
  const [filter, setFilter] = useState<RunFilter>('network');
  const trainable = canTrain(training.scope);
  // On phones the panel always fills the screen.
  const [wide, setWide] = useState(readWide);
  // The focus goes back to the Train button when the panel closes.
  useReturnFocusOnClose(training.opened);
  const toggleWide = () => {
    setWide(!wide);
    writeWide(!wide);
  };

  return (
    <Drawer.Root
      opened={training.opened}
      onClose={training.close}
      position="right"
      size={phone || wide ? '100%' : 560}
    >
      <Drawer.Overlay />
      <Drawer.Content>
        <Drawer.Header>
          <Drawer.Title>
            <Group gap="xs">
              {trainable ? <IconPlayerPlay size={18} /> : <IconHistory size={18} />}
              <Text fw={600}>
                {trainable ? t('training.drawer.title') : t('training.drawer.titleRuns')}
              </Text>
            </Group>
          </Drawer.Title>
          <Group gap={4} wrap="nowrap">
            {!phone && (
              <Tooltip
                label={wide ? t('training.drawer.sidePanel') : t('training.drawer.fullWidth')}
                position="bottom"
              >
                <ActionIcon
                  variant="subtle"
                  color="gray"
                  size="lg"
                  aria-label={t('training.drawer.fullWidth')}
                  aria-pressed={wide}
                  onClick={toggleWide}
                >
                  {wide ? <IconArrowsMinimize size={18} /> : <IconArrowsMaximize size={18} />}
                </ActionIcon>
              </Tooltip>
            )}
            {/* As before, the dialog's first focus goes to closing it. */}
            <Drawer.CloseButton data-autofocus aria-label={t('common.close')} />
          </Group>
        </Drawer.Header>
        <Drawer.Body className={layout.panel}>
          <DrawerContent {...props} filter={filter} onFilterChange={setFilter} />
        </Drawer.Body>
      </Drawer.Content>
    </Drawer.Root>
  );
}

function DrawerContent({
  training,
  loggedIn,
  doc,
  networkName,
  analysis,
  onLogin,
  onShowProblems,
  onLoadNetwork,
  fit = null,
  filter,
  onFilterChange,
}: TrainingDrawerProps & { filter: RunFilter; onFilterChange: (filter: RunFilter) => void }) {
  const { t } = useTranslation();
  const { status, statusError, scope } = training;
  // Viewers of the network see its runs, but cannot start trainings in it.
  const trainable = canTrain(scope);
  // Neural networks need PyTorch on the server, classical models scikit-learn.
  const family = trainingFamily(analysis);

  if (!status) {
    return statusError ? (
      <Alert
        color="red"
        icon={<IconPlugConnectedX size={18} />}
        title={t('training.drawer.unavailable')}
      >
        <Text size="sm">{statusError}</Text>
        <Button
          size="compact-sm"
          variant="light"
          color="red"
          mt="xs"
          onClick={training.reloadStatus}
        >
          {t('common.retry')}
        </Button>
      </Alert>
    ) : (
      <Center h={200}>
        <Loader />
      </Center>
    );
  }
  if (!status.enabled) {
    return <Unavailable reason={status.reason} disabled onRetry={training.reloadStatus} />;
  }
  if (!loggedIn) {
    return (
      <Stack gap="md">
        {!canStartTraining(status, family) && (
          <Unavailable
            reason={unavailableReason(status, family)}
            disabled={false}
            family={family}
            sklearnMissing={!status.classical?.available}
            torchMissing={!status.available}
            onRetry={training.reloadStatus}
          />
        )}
        <Stack align="center" ta="center" gap="sm" py="xl" px="md">
          <ThemeIcon size={48} radius="xl" variant="light">
            <IconLock size={24} />
          </ThemeIcon>
          <Title order={4}>{t('training.drawer.loginTitle')}</Title>
          <Text size="sm" c="dimmed" maw={380}>
            {t('training.drawer.loginText')}
          </Text>
          <Button onClick={onLogin}>{t('auth.logIn')}</Button>
        </Stack>
      </Stack>
    );
  }

  const tab = training.view === 'setup' && trainable ? 'setup' : 'runs';
  return (
    <Tabs
      value={tab}
      onChange={(value) => (value === 'setup' ? training.showSetup() : training.showHistory())}
      keepMounted={false}
    >
      <Tabs.List mb="md">
        {trainable && (
          <Tabs.Tab value="setup" leftSection={<IconPlayerPlay size={14} />}>
            {t('training.drawer.newTraining')}
          </Tabs.Tab>
        )}
        <Tabs.Tab
          value="runs"
          leftSection={<IconHistory size={14} />}
          rightSection={training.activeRun ? <Loader size={10} /> : undefined}
        >
          {t('training.drawer.runs')}
        </Tabs.Tab>
      </Tabs.List>

      <Tabs.Panel value="setup">
        {/* Without PyTorch (scikit-learn for classical models) no new training can start,
            but past runs stay usable. */}
        {canStartTraining(status, family) ? (
          <SetupView
            status={status}
            doc={doc}
            networkName={networkName}
            project={trainingProject(scope)}
            analysis={analysis}
            activeRun={training.activeRun}
            activeRunBy={training.activeRun && otherStarter(training.activeRun, scope)}
            data={training.data}
            onStarted={training.applyRun}
            onOpenRun={training.openRun}
            onShowProblems={onShowProblems}
            fit={fit}
            onRefresh={() => {
              training.reloadStatus();
              training.refreshRuns();
            }}
          />
        ) : (
          <Unavailable
            reason={unavailableReason(status, family)}
            disabled={false}
            family={family}
            sklearnMissing={!status.classical?.available}
            torchMissing={!status.available}
            onRetry={training.reloadStatus}
          />
        )}
      </Tabs.Panel>

      <Tabs.Panel value="runs">
        {!trainable && (
          <Group gap={6} wrap="nowrap" align="flex-start" mb="sm">
            <IconEye
              size={16}
              color="var(--mantine-color-dimmed)"
              style={{ flexShrink: 0, marginTop: 2 }}
              aria-hidden
            />
            <Text size="xs" c="dimmed">
              {t('training.drawer.viewerNote')}
            </Text>
          </Group>
        )}
        {training.view === 'run' && training.selectedRunId ? (
          <SelectedRun training={training} doc={doc} onLoadNetwork={onLoadNetwork} />
        ) : (
          <HistoryView
            runs={training.runs}
            error={training.runsError}
            onRetry={training.refreshRuns}
            onOpen={training.openRun}
            onDeleted={training.forgetRun}
            onNewTraining={training.showSetup}
            scope={scope}
            filter={filter}
            onFilterChange={onFilterChange}
            onCombined={trainable ? training.applyRun : undefined}
            combineProject={trainingProject(scope)?.id ?? null}
          />
        )}
      </Tabs.Panel>
    </Tabs>
  );
}

function SelectedRun({
  training,
  doc,
  onLoadNetwork,
}: Pick<TrainingDrawerProps, 'training' | 'doc' | 'onLoadNetwork'>) {
  const { t } = useTranslation();
  const { detail, detailError } = training;
  if (!detail) {
    return detailError ? (
      <Alert
        color="red"
        icon={<IconAlertTriangle size={18} />}
        title={t('training.drawer.cannotLoad')}
      >
        <Text size="sm">{detailError}</Text>
        <Group gap="xs" mt="xs">
          <Button size="compact-sm" variant="light" color="red" onClick={training.retryDetail}>
            {t('common.retry')}
          </Button>
          <Button size="compact-sm" variant="subtle" color="gray" onClick={training.showHistory}>
            {t('training.drawer.allRuns')}
          </Button>
        </Group>
      </Alert>
    ) : (
      <Center h={200}>
        <Loader />
      </Center>
    );
  }
  return (
    <Stack gap="xs">
      {detailError && (
        <Alert color="yellow" icon={<IconPlugConnectedX size={18} />}>
          {t('training.drawer.lostConnection', { error: detailError })}
        </Alert>
      )}
      <RunView
        run={detail}
        doc={doc}
        scope={training.scope}
        onBack={training.showHistory}
        onNewTraining={training.showSetup}
        onUpdate={training.applyRun}
        onDeleted={training.forgetRun}
        onLoadNetwork={onLoadNetwork}
        onOpenRun={training.openRun}
      />
    </Stack>
  );
}

function Unavailable({
  reason,
  disabled,
  family = 'neural',
  sklearnMissing = false,
  torchMissing = true,
  onRetry,
}: {
  reason: string | null;
  disabled: boolean;
  family?: Family;
  /** scikit-learn is not installed (classical models and hybrid networks need it). */
  sklearnMissing?: boolean;
  /** PyTorch is not installed (or cannot be loaded). */
  torchMissing?: boolean;
  onRetry: () => void;
}) {
  const { t } = useTranslation();
  // A hybrid network needs both: PyTorch for its neural layers, scikit-learn for its model.
  const hybridWithoutSklearn = family === 'hybrid' && sklearnMissing && !torchMissing;
  if ((family === 'classical' || hybridWithoutSklearn) && !disabled) {
    return (
      <Stack gap="md">
        <Alert
          color="yellow"
          icon={<IconAlertTriangle size={18} />}
          title={t('training.drawer.unavailable')}
        >
          {reason ?? t('training.drawer.noSklearn')}
        </Alert>
        <Stack gap="xs">
          {hybridWithoutSklearn ? (
            <Text size="sm">
              <Trans i18nKey="training.drawer.hybridNoSklearn" components={{ code: <Code /> }} />
            </Text>
          ) : (
            <Text size="sm">
              <Trans i18nKey="training.drawer.classicalNoSklearn" components={{ code: <Code /> }} />
            </Text>
          )}
          <Code block>{i18n.t('ui.training.TrainingDrawer.pipInstallRRequirementsTxt')}</Code>
          <Text size="sm" c="dimmed">
            {t('training.drawer.dockerSklearn')}
          </Text>
          <Button variant="light" onClick={onRetry} style={{ alignSelf: 'flex-start' }}>
            {t('training.drawer.checkAgain')}
          </Button>
        </Stack>
      </Stack>
    );
  }
  return (
    <Stack gap="md">
      <Alert
        color={disabled ? 'gray' : 'yellow'}
        icon={<IconAlertTriangle size={18} />}
        title={t('training.drawer.unavailable')}
      >
        {reason ?? (disabled ? t('training.drawer.disabled') : t('training.drawer.noTorch'))}
      </Alert>
      {!disabled && (
        <Stack gap="xs">
          <Text size="sm">{t('training.drawer.needsTorch')}</Text>
          <Text size="sm" fw={600}>
            {t('training.drawer.local')}
          </Text>
          <Text size="sm" c="dimmed">
            <Trans i18nKey="training.drawer.inBackend" components={{ code: <Code /> }} />
          </Text>
          <Code block>{i18n.t('ui.training.TrainingDrawer.pipInstallTorchTorchvision')}</Code>
          <Text size="sm" fw={600}>
            {i18n.t('ui.training.TrainingDrawer.docker')}
          </Text>
          <Code block>{i18n.t('ui.training.TrainingDrawer.brickdnnInstallTorchTrueDocker')}</Code>
          {family === 'hybrid' && sklearnMissing && (
            <Text size="sm">
              <Trans i18nKey="training.drawer.hybridAlso" components={{ code: <Code /> }} />
            </Text>
          )}
          <Text size="sm" c="dimmed">
            {t('training.drawer.restart')}
          </Text>
          <Button variant="light" onClick={onRetry} style={{ alignSelf: 'flex-start' }}>
            {t('training.drawer.checkAgain')}
          </Button>
        </Stack>
      )}
    </Stack>
  );
}
