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
import { DEMO } from '@/demo';
import { LiveBadge } from '@/live/LiveBadge';
import type { LiveIndicator } from '@/live/useLiveBuilder';
import { PeopleAvatars } from '@/sharing/PeopleAvatars';
import sharingClasses from '@/sharing/Sharing.module.css';
import { progressLabel } from '@/training/format';
import { formatCount } from './format';
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
  /** Opens the share dialog; not available in the online demo. */
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
        {/* Phones keep the space for the network name: the check says it is valid. */}
        <Box component="span" visibleFrom="xs">
          Valid ·{' '}
        </Box>
        {formatCount(analysis.total_params)} params
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
  const { name, onNameChange, examples, saving, isSaved, readOnly = false, owner, people } = props;
  const { onShare, onDuplicate, here = null, live = null } = props;
  const canSave = !DEMO && !readOnly && !live;

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
        >
          Code
        </Button>
        {!readOnly && <TrainButton run={props.trainingRun} onClick={props.onTrain} />}
        <Tooltip label="Download the PyTorch script (Ctrl+E)">
          <Button
            leftSection={<IconDownload size={16} />}
            onClick={props.onExportPython}
            className={classes.exportLabelled}
          >
            Export .py
          </Button>
        </Tooltip>
        <Tooltip label="Export .py: download the PyTorch script">
          <ActionIcon
            size="lg"
            onClick={props.onExportPython}
            className={classes.exportIcon}
            aria-label="Download the PyTorch script"
          >
            <IconDownload size={18} />
          </ActionIcon>
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
