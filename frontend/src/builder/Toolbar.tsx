import {
  IconAdjustmentsHorizontal,
  IconChevronDown,
  IconCircleCheck,
  IconCode,
  IconDeviceFloppy,
  IconDownload,
  IconFileExport,
  IconFileImport,
  IconFilePlus,
  IconLayoutGrid,
  IconLayoutSidebar,
  IconPlayerPlay,
  IconSchool,
} from '@tabler/icons-react';
import {
  ActionIcon,
  Badge,
  Button,
  Group,
  Loader,
  Menu,
  Text,
  TextInput,
  Tooltip,
} from '@mantine/core';
import type { Analysis, Example, RunSummary } from '@/api/types';
import { progressLabel } from '@/training/format';
import { formatCount } from './format';

interface ToolbarProps {
  name: string;
  onNameChange: (name: string) => void;
  analysis: Analysis | null;
  analyzing: boolean;
  analysisError: string | null;
  empty: boolean;
  examples: Example[];
  saving: boolean;
  isSaved: boolean;
  onLoadExample: (example: Example) => void;
  onNew: () => void;
  onImport: () => void;
  onExportJson: () => void;
  onTidy: () => void;
  onSave: () => void;
  onShowCode: () => void;
  onExportPython: () => void;
  onTrain: () => void;
  /** The user's training that is currently running, if any. */
  trainingRun: RunSummary | null;
  onShowSummary: () => void;
  onOpenPalette: () => void;
  onOpenInspector: () => void;
}

function Status({
  analysis,
  analyzing,
  analysisError,
  empty,
  onShowSummary,
}: Pick<ToolbarProps, 'analysis' | 'analyzing' | 'analysisError' | 'empty' | 'onShowSummary'>) {
  if (empty) {
    return (
      <Badge color="gray" variant="light" tt="none" style={{ flexShrink: 0 }}>
        Empty
      </Badge>
    );
  }
  if (analysisError) {
    return (
      <Tooltip label={analysisError} multiline w={260}>
        <Badge color="red" variant="light" tt="none" style={{ flexShrink: 0 }}>
          Server unreachable
        </Badge>
      </Tooltip>
    );
  }
  if (!analysis) {
    return <Loader size="xs" />;
  }
  if (analysis.valid) {
    return (
      <Badge
        color="green"
        variant="light"
        leftSection={<IconCircleCheck size={12} />}
        tt="none"
        style={{ cursor: 'pointer', flexShrink: 0, opacity: analyzing ? 0.6 : 1 }}
        onClick={onShowSummary}
      >
        Valid · {formatCount(analysis.total_params)} params
      </Badge>
    );
  }
  const count = analysis.errors.length;
  return (
    <Badge
      color="red"
      variant="light"
      tt="none"
      style={{ cursor: 'pointer', flexShrink: 0, opacity: analyzing ? 0.6 : 1 }}
      onClick={onShowSummary}
    >
      {count === 1 ? '1 problem' : `${count} problems`}
    </Badge>
  );
}

export function Toolbar(props: ToolbarProps) {
  const { name, onNameChange, examples, saving, isSaved } = props;

  return (
    <Group
      justify="space-between"
      wrap="wrap"
      px="sm"
      py={6}
      gap="xs"
      style={{ borderBottom: '1px solid var(--mantine-color-default-border)' }}
    >
      <Group gap="xs" wrap="nowrap" style={{ minWidth: 0 }}>
        <Tooltip label="Layers">
          <ActionIcon
            hiddenFrom="md"
            variant="default"
            size="lg"
            onClick={props.onOpenPalette}
            aria-label="Show layers"
          >
            <IconLayoutSidebar size={18} />
          </ActionIcon>
        </Tooltip>
        <TextInput
          aria-label="Network name"
          value={name}
          onChange={(event) => onNameChange(event.currentTarget.value)}
          w={{ base: 150, sm: 210 }}
          styles={{ input: { fontWeight: 600 } }}
        />
        <Status {...props} />
      </Group>

      <Group gap="xs" wrap="nowrap" ml="auto">
        <Menu position="bottom-end" withinPortal>
          <Menu.Target>
            <Button variant="default" rightSection={<IconChevronDown size={14} />}>
              File
            </Button>
          </Menu.Target>
          <Menu.Dropdown maw={360}>
            <Menu.Item leftSection={<IconFilePlus size={16} />} onClick={props.onNew}>
              New network
            </Menu.Item>
            <Menu.Item leftSection={<IconFileImport size={16} />} onClick={props.onImport}>
              Open JSON file…
            </Menu.Item>
            <Menu.Item leftSection={<IconFileExport size={16} />} onClick={props.onExportJson}>
              Save as JSON file
            </Menu.Item>
            <Menu.Item leftSection={<IconLayoutGrid size={16} />} onClick={props.onTidy}>
              Tidy up layout
            </Menu.Item>
            <Menu.Item
              hiddenFrom="sm"
              leftSection={<IconDeviceFloppy size={16} />}
              onClick={props.onSave}
            >
              Save to account
            </Menu.Item>
            <Menu.Item
              hiddenFrom="sm"
              leftSection={<IconCode size={16} />}
              onClick={props.onShowCode}
            >
              View code
            </Menu.Item>
            <Menu.Divider />
            <Menu.Label>Examples</Menu.Label>
            {examples.map((example) => (
              <Menu.Item
                key={example.id}
                leftSection={<IconSchool size={16} />}
                onClick={() => props.onLoadExample(example)}
              >
                <Text size="sm">{example.name}</Text>
                <Text size="xs" c="dimmed" visibleFrom="sm">
                  {example.description}
                </Text>
              </Menu.Item>
            ))}
          </Menu.Dropdown>
        </Menu>

        <Tooltip label={isSaved ? 'Saved to your account' : 'Save to your account (Ctrl+S)'}>
          <ActionIcon
            variant="default"
            size="lg"
            loading={saving}
            onClick={props.onSave}
            visibleFrom="sm"
            aria-label="Save to account"
          >
            <IconDeviceFloppy size={18} />
          </ActionIcon>
        </Tooltip>
        <Button
          variant="default"
          leftSection={<IconCode size={16} />}
          onClick={props.onShowCode}
          visibleFrom="sm"
        >
          Code
        </Button>
        <TrainButton run={props.trainingRun} onClick={props.onTrain} />
        <Tooltip label="Download the PyTorch script (Ctrl+E)">
          <Button leftSection={<IconDownload size={16} />} onClick={props.onExportPython}>
            Export .py
          </Button>
        </Tooltip>
        <Tooltip label="Properties">
          <ActionIcon
            hiddenFrom="lg"
            variant="default"
            size="lg"
            onClick={props.onOpenInspector}
            aria-label="Show properties"
          >
            <IconAdjustmentsHorizontal size={18} />
          </ActionIcon>
        </Tooltip>
      </Group>
    </Group>
  );
}

function TrainButton({ run, onClick }: { run: RunSummary | null; onClick: () => void }) {
  const label = run
    ? `Training “${run.name}”: ${progressLabel(run)}`
    : 'Train this network on the server (Ctrl+Enter)';
  return (
    <>
      <Tooltip label={label}>
        <Button
          variant="default"
          leftSection={run ? <Loader size={14} /> : <IconPlayerPlay size={16} />}
          onClick={onClick}
          visibleFrom="sm"
        >
          {run ? progressLabel(run) : 'Train'}
        </Button>
      </Tooltip>
      <Tooltip label={label}>
        <ActionIcon
          variant="default"
          size="lg"
          onClick={onClick}
          hiddenFrom="sm"
          aria-label={run ? `Training: ${progressLabel(run)}` : 'Train'}
        >
          {run ? <Loader size={16} /> : <IconPlayerPlay size={18} />}
        </ActionIcon>
      </Tooltip>
    </>
  );
}
