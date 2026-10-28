import {
  IconAlertTriangle,
  IconHistory,
  IconLock,
  IconPlayerPlay,
  IconPlugConnectedX,
} from '@tabler/icons-react';
import {
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
  useMantineTheme,
} from '@mantine/core';
import { useMediaQuery } from '@mantine/hooks';
import type { Analysis, GraphDocument } from '@/api';
import { HistoryView } from './HistoryView';
import { RunView } from './RunView';
import { SetupView } from './SetupView';
import type { TrainingController } from './useTraining';

export interface TrainingDrawerProps {
  training: TrainingController;
  loggedIn: boolean;
  doc: GraphDocument;
  networkName: string;
  analysis: Analysis | null;
  onLogin: () => void;
  onShowProblems: () => void;
  onLoadNetwork: (graph: GraphDocument, name: string) => void;
}

export default function TrainingDrawer(props: TrainingDrawerProps) {
  const { training } = props;
  const theme = useMantineTheme();
  const phone = useMediaQuery(`(max-width: ${theme.breakpoints.sm})`);

  return (
    <Drawer
      opened={training.opened}
      onClose={training.close}
      position="right"
      size={phone ? '100%' : 560}
      title={
        <Group gap="xs">
          <IconPlayerPlay size={18} />
          <Text fw={600}>Train this network</Text>
        </Group>
      }
    >
      <DrawerContent {...props} />
    </Drawer>
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
}: TrainingDrawerProps) {
  const { status, statusError } = training;

  if (!status) {
    return statusError ? (
      <Alert color="red" icon={<IconPlugConnectedX size={18} />} title="Training is not available">
        <Text size="sm">{statusError}</Text>
        <Button
          size="compact-sm"
          variant="light"
          color="red"
          mt="xs"
          onClick={training.reloadStatus}
        >
          Try again
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
        {!status.available && (
          <Unavailable reason={status.reason} disabled={false} onRetry={training.reloadStatus} />
        )}
        <Stack align="center" ta="center" gap="sm" py="xl" px="md">
          <ThemeIcon size={48} radius="xl" variant="light">
            <IconLock size={24} />
          </ThemeIcon>
          <Title order={4}>Log in to train</Title>
          <Text size="sm" c="dimmed" maw={380}>
            Training runs on the Brick DNN server and needs a free account, just like saving. Your
            network stays on the canvas while you log in.
          </Text>
          <Button onClick={onLogin}>Log in</Button>
        </Stack>
      </Stack>
    );
  }

  const tab = training.view === 'setup' ? 'setup' : 'runs';
  return (
    <Tabs
      value={tab}
      onChange={(value) => (value === 'setup' ? training.showSetup() : training.showHistory())}
      keepMounted={false}
    >
      <Tabs.List mb="md">
        <Tabs.Tab value="setup" leftSection={<IconPlayerPlay size={14} />}>
          New training
        </Tabs.Tab>
        <Tabs.Tab
          value="runs"
          leftSection={<IconHistory size={14} />}
          rightSection={training.activeRun ? <Loader size={10} /> : undefined}
        >
          Runs
        </Tabs.Tab>
      </Tabs.List>

      <Tabs.Panel value="setup">
        {/* Without PyTorch no new training can start, but past runs stay usable. */}
        {status.available ? (
          <SetupView
            status={status}
            doc={doc}
            networkName={networkName}
            analysis={analysis}
            activeRun={training.activeRun}
            data={training.data}
            onStarted={training.applyRun}
            onOpenRun={training.openRun}
            onShowProblems={onShowProblems}
            onRefresh={() => {
              training.reloadStatus();
              training.refreshRuns();
            }}
          />
        ) : (
          <Unavailable reason={status.reason} disabled={false} onRetry={training.reloadStatus} />
        )}
      </Tabs.Panel>

      <Tabs.Panel value="runs">
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
  const { detail, detailError } = training;
  if (!detail) {
    return detailError ? (
      <Alert color="red" icon={<IconAlertTriangle size={18} />} title="Cannot load the training">
        <Text size="sm">{detailError}</Text>
        <Group gap="xs" mt="xs">
          <Button size="compact-sm" variant="light" color="red" onClick={training.retryDetail}>
            Try again
          </Button>
          <Button size="compact-sm" variant="subtle" color="gray" onClick={training.showHistory}>
            All runs
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
          Lost the connection to the server ({detailError}). Retrying …
        </Alert>
      )}
      <RunView
        run={detail}
        doc={doc}
        onBack={training.showHistory}
        onNewTraining={training.showSetup}
        onUpdate={training.applyRun}
        onDeleted={training.forgetRun}
        onLoadNetwork={onLoadNetwork}
      />
    </Stack>
  );
}

function Unavailable({
  reason,
  disabled,
  onRetry,
}: {
  reason: string | null;
  disabled: boolean;
  onRetry: () => void;
}) {
  return (
    <Stack gap="md">
      <Alert
        color={disabled ? 'gray' : 'yellow'}
        icon={<IconAlertTriangle size={18} />}
        title="Training is not available"
      >
        {reason ??
          (disabled
            ? 'Training is disabled on this server.'
            : 'PyTorch is not installed on the server.')}
      </Alert>
      {!disabled && (
        <Stack gap="xs">
          <Text size="sm">
            Training runs on the machine of the Brick DNN backend and needs PyTorch there. To
            install it:
          </Text>
          <Text size="sm" fw={600}>
            Local installation
          </Text>
          <Text size="sm" c="dimmed">
            In the <Code>backend</Code> folder, with the virtual environment active:
          </Text>
          <Code block>pip install torch torchvision</Code>
          <Text size="sm" fw={600}>
            Docker
          </Text>
          <Code block>BRICKDNN_INSTALL_TORCH=true docker compose up -d --build</Code>
          <Text size="sm" c="dimmed">
            Then restart the backend and check again.
          </Text>
          <Button variant="light" onClick={onRetry} style={{ alignSelf: 'flex-start' }}>
            Check again
          </Button>
        </Stack>
      )}
    </Stack>
  );
}
