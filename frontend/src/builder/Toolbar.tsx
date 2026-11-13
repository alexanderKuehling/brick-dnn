import {
  IconAdjustmentsHorizontal,
  IconChevronDown,
  IconCircleCheck,
  IconCode,
  IconCopyPlus,
  IconDeviceFloppy,
  IconDownload,
  IconEye,
  IconFileExport,
  IconFileImport,
  IconFilePlus,
  IconHelp,
  IconLayoutGrid,
  IconLayoutSidebar,
  IconPlayerPlay,
  IconSchool,
  IconUsersPlus,
} from '@tabler/icons-react';
import {
  ActionIcon,
  Badge,
  Box,
  Button,
  Group,
  Loader,
  Menu,
  Text,
  TextInput,
  Tooltip,
} from '@mantine/core';
import type { Analysis, Example, RunSummary } from '@/api/types';
import { LiveBadge } from '@/live/LiveBadge';
import type { LiveIndicator } from '@/live/useLiveBuilder';
import { PeopleAvatars } from '@/sharing/PeopleAvatars';
import sharingClasses from '@/sharing/Sharing.module.css';
import { progressLabel, queueText } from '@/training/format';
import { networkSize } from './format';
import { networkFamily } from './kinds';
import classes from './Toolbar.module.css';

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
  /** View only (shared with the user as a viewer): no saving, editing or training. */
  readOnly?: boolean;
  /** Username of the owner of the open network, if it is saved and known. */
  owner?: string | null;
  /** Everyone with access to a shared network, the owner first; null if it is not shared. */
  people?: string[] | null;
  /** Live editing: who has the network open right now (usernames). */
  here?: string[] | null;
  /** Live editing: the network is saved as it changes, so a status replaces the save button. */
  live?: LiveIndicator | null;
  onLoadExample: (example: Example) => void;
  onNew: () => void;
  onImport: () => void;
  onExportJson: () => void;
  onTidy: () => void;
  onSave: () => void;
  /** Opens the share dialog. */
  onShare?: () => void;
  /** Viewers: copy the network into the own account. */
  onDuplicate?: () => void;
  onShowCode: () => void;
  onExportPython: () => void;
  onTrain: () => void;
  /** The user's training that is currently running, if any. */
  trainingRun: RunSummary | null;
  onShowSummary: () => void;
  onOpenPalette: () => void;
  onOpenInspector: () => void;
  /** Opens the guide to the builder again (Onboarding.tsx). */
  onShowGuide?: () => void;
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
      <Badge color="gray" variant="light" tt="none" style={{ flexShrink: 0 }} data-tour="status">
        Empty
      </Badge>
    );
  }
  if (analysisError) {
    return (
      <Tooltip label={analysisError} multiline w={260}>
        <Badge color="red" variant="light" tt="none" style={{ flexShrink: 0 }} data-tour="status">
          Server unreachable
        </Badge>
      </Tooltip>
    );
  }
  if (!analysis) {
    return <Loader size="xs" />;
  }
  if (analysis.valid) {
    const count = networkSize(analysis);
    return (
      <Badge
        color="green"
        variant="light"
        leftSection={<IconCircleCheck size={12} />}
        tt="none"
        style={{ cursor: 'pointer', flexShrink: 0, opacity: analyzing ? 0.6 : 1 }}
        onClick={onShowSummary}
        data-tour="status"
      >
        {/* Phones keep the space for the network name: the check says it is valid. */}
        <Box component="span" visibleFrom="xs">
          Valid ·{' '}
        </Box>
        {count}
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
      data-tour="status"
    >
      {count === 1 ? '1 problem' : `${count} problems`}
    </Badge>
  );
}

export function Toolbar(props: ToolbarProps) {
  const { name, onNameChange, examples, saving, isSaved, readOnly = false, owner, people } = props;
  const family = networkFamily(props.analysis);
  const library =
    family === 'classical'
      ? 'scikit-learn'
      : family === 'hybrid'
        ? 'PyTorch and scikit-learn'
        : 'PyTorch';
  const { onShare, onDuplicate, here = null, live = null } = props;
  const canSave = !readOnly && !live;

  return (
    <Group
      justify="space-between"
      wrap="wrap"
      px="sm"
      py={6}
      gap="xs"
      style={{ borderBottom: '1px solid var(--mantine-color-default-border)' }}
    >
      <Group gap="xs" wrap="nowrap" style={{ minWidth: 0, flex: '1 1 auto' }}>
        {!readOnly && (
          <Tooltip label="Layers">
            <ActionIcon
              hiddenFrom="md"
              variant="default"
              size="lg"
              onClick={props.onOpenPalette}
              aria-label="Show layers"
              data-tour="palette-button"
            >
              <IconLayoutSidebar size={18} />
            </ActionIcon>
          </Tooltip>
        )}
        <TextInput
          aria-label="Network name"
          value={name}
          onChange={(event) => onNameChange(event.currentTarget.value)}
          readOnly={readOnly}
          variant={readOnly ? 'filled' : 'default'}
          w={{ base: 150, sm: 210 }}
          // On phones the name takes the free width of its row.
          flex={{ base: '1 1 150px', sm: 'none' }}
          miw={80}
          maw={{ base: 320, sm: 'none' }}
          styles={{ input: { fontWeight: 600 } }}
        />
        <Status {...props} />
        {readOnly && (
          <Tooltip
            label={`${owner ? `Shared with you by ${owner}. ` : ''}You can view it, not change it.`}
            multiline
            w={240}
          >
            <Badge
              color="gray"
              variant="outline"
              tt="none"
              leftSection={<IconEye size={12} />}
              style={{ flexShrink: 0 }}
            >
              <span className={sharingClasses.badgeLabel}>View only</span>
            </Badge>
          </Tooltip>
        )}
        {/* Phones show who has access next to the name; the share button is in the menu. */}
        {people && onShare && !readOnly && (
          <Box hiddenFrom="sm" className={sharingClasses.phoneAvatars}>
            <PeopleAvatars names={people} here={here} max={2} onClick={onShare} />
          </Box>
        )}
      </Group>

      <Group gap="xs" wrap="nowrap" ml="auto">
        {people && onShare && (
          <Box visibleFrom="sm">
            <PeopleAvatars names={people} here={here} max={here ? 5 : 3} onClick={onShare} />
          </Box>
        )}
        {onShare && (
          <Tooltip label="Invite people or share a link">
            <Button
              variant="default"
              leftSection={<IconUsersPlus size={16} />}
              onClick={onShare}
              visibleFrom="sm"
            >
              Share
            </Button>
          </Tooltip>
        )}
        <Menu position="bottom-end" withinPortal>
          <Menu.Target>
            <Button variant="default" rightSection={<IconChevronDown size={14} />}>
              File
            </Button>
          </Menu.Target>
          {/* At most the height of the screen: the list of examples is longer than a phone. */}
          <Menu.Dropdown maw={360} mah="calc(100dvh - 5rem)" style={{ overflowY: 'auto' }}>
            <Menu.Item leftSection={<IconFilePlus size={16} />} onClick={props.onNew}>
              New network
            </Menu.Item>
            <Menu.Item leftSection={<IconFileImport size={16} />} onClick={props.onImport}>
              Open JSON file…
            </Menu.Item>
            <Menu.Item leftSection={<IconFileExport size={16} />} onClick={props.onExportJson}>
              Save as JSON file
            </Menu.Item>
            {!readOnly && (
              <Menu.Item leftSection={<IconLayoutGrid size={16} />} onClick={props.onTidy}>
                Tidy up layout
              </Menu.Item>
            )}
            {canSave && (
              <Menu.Item
                hiddenFrom="sm"
                leftSection={<IconDeviceFloppy size={16} />}
                onClick={props.onSave}
              >
                Save to account
              </Menu.Item>
            )}
            {readOnly && onDuplicate && (
              <Menu.Item
                hiddenFrom="sm"
                leftSection={<IconCopyPlus size={16} />}
                onClick={onDuplicate}
              >
                Duplicate to my networks
              </Menu.Item>
            )}
            {onShare && (
              <Menu.Item
                hiddenFrom="sm"
                leftSection={<IconUsersPlus size={16} />}
                onClick={onShare}
              >
                Share…
              </Menu.Item>
            )}
            <Menu.Item
              hiddenFrom="sm"
              leftSection={<IconCode size={16} />}
              onClick={props.onShowCode}
            >
              View code
            </Menu.Item>
            {props.onShowGuide && (
              <Menu.Item leftSection={<IconHelp size={16} />} onClick={props.onShowGuide}>
                Guide to the builder
              </Menu.Item>
            )}
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

        {live && <LiveBadge indicator={live} />}
        {canSave && (
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
        )}
        {readOnly && onDuplicate && (
          <>
            <Tooltip label="Make a copy that you can change">
              <Button
                variant="default"
                leftSection={<IconCopyPlus size={16} />}
                onClick={onDuplicate}
                visibleFrom="sm"
              >
                Duplicate to my networks
              </Button>
            </Tooltip>
            <Tooltip label="Duplicate to my networks">
              <ActionIcon
                variant="default"
                size="lg"
                onClick={onDuplicate}
                hiddenFrom="sm"
                aria-label="Duplicate to my networks"
              >
                <IconCopyPlus size={18} />
              </ActionIcon>
            </Tooltip>
          </>
        )}
        <Button
          variant="default"
          leftSection={<IconCode size={16} />}
          onClick={props.onShowCode}
          visibleFrom="sm"
          data-tour="code"
        >
          Code
        </Button>
        {!readOnly && <TrainButton run={props.trainingRun} onClick={props.onTrain} />}
        <Tooltip label={`Download the ${library} script (Ctrl+E)`}>
          <Button
            leftSection={<IconDownload size={16} />}
            onClick={props.onExportPython}
            className={classes.exportLabelled}
            data-tour="export"
          >
            Export .py
          </Button>
        </Tooltip>
        <Tooltip label={`Export .py: download the ${library} script`}>
          <ActionIcon
            size="lg"
            onClick={props.onExportPython}
            className={classes.exportIcon}
            aria-label={`Download the ${library} script`}
            data-tour="export-icon"
          >
            <IconDownload size={18} />
          </ActionIcon>
        </Tooltip>
        {props.onShowGuide && (
          <Tooltip label="Guide to the builder">
            <ActionIcon
              variant="subtle"
              color="gray"
              size="lg"
              onClick={props.onShowGuide}
              visibleFrom="sm"
              aria-label="Show the guide"
            >
              <IconHelp size={18} />
            </ActionIcon>
          </Tooltip>
        )}
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
  const label = !run
    ? 'Train this network on the server (Ctrl+Enter)'
    : run.status === 'queued'
      ? `“${run.name}”: ${queueText(run.queue)}`
      : `Training “${run.name}”: ${progressLabel(run)}`;
  return (
    <>
      <Tooltip label={label}>
        <Button
          variant="default"
          leftSection={run ? <Loader size={14} /> : <IconPlayerPlay size={16} />}
          onClick={onClick}
          visibleFrom="sm"
          data-tour="train"
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
          data-tour="train-icon"
        >
          {run ? <Loader size={16} /> : <IconPlayerPlay size={18} />}
        </ActionIcon>
      </Tooltip>
    </>
  );
}
