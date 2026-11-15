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
import { i18n, useTranslation } from '@/i18n';
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
  const { t } = useTranslation();
  if (empty) {
    return (
      <Badge color="gray" variant="light" tt="none" style={{ flexShrink: 0 }} data-tour="status">
        {t('builder.toolbar.empty')}
      </Badge>
    );
  }
  if (analysisError) {
    return (
      <Tooltip label={analysisError} multiline w={260}>
        <Badge color="red" variant="light" tt="none" style={{ flexShrink: 0 }} data-tour="status">
          {t('builder.toolbar.unreachable')}
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
          {t('builder.toolbar.valid')} ·{' '}
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
      {t('builder.toolbar.problems', { count })}
    </Badge>
  );
}

export function Toolbar(props: ToolbarProps) {
  const { t } = useTranslation();
  const { name, onNameChange, examples, saving, isSaved, readOnly = false, owner, people } = props;
  const family = networkFamily(props.analysis);
  const library =
    family === 'classical'
      ? 'scikit-learn'
      : family === 'hybrid'
        ? t('builder.toolbar.pytorchAndSklearn')
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
          <Tooltip label={t('builder.toolbar.layers')}>
            <ActionIcon
              hiddenFrom="md"
              variant="default"
              size="lg"
              onClick={props.onOpenPalette}
              aria-label={t('builder.toolbar.showLayers')}
              data-tour="palette-button"
            >
              <IconLayoutSidebar size={18} />
            </ActionIcon>
          </Tooltip>
        )}
        <TextInput
          aria-label={t('builder.toolbar.networkName')}
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
            label={owner ? t('builder.toolbar.sharedBy', { owner }) : t('builder.toolbar.viewOnly')}
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
              <span className={sharingClasses.badgeLabel}>
                {i18n.t('ui.builder.Toolbar.viewOnly')}
              </span>
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
          <Tooltip label={t('builder.toolbar.inviteTooltip')}>
            <Button
              variant="default"
              leftSection={<IconUsersPlus size={16} />}
              onClick={onShare}
              visibleFrom="sm"
            >
              {t('builder.toolbar.shareButton')}
            </Button>
          </Tooltip>
        )}
        <Menu position="bottom-end" withinPortal>
          <Menu.Target>
            <Button variant="default" rightSection={<IconChevronDown size={14} />}>
              {t('builder.toolbar.file')}
            </Button>
          </Menu.Target>
          {/* At most the height of the screen: the list of examples is longer than a phone. */}
          <Menu.Dropdown maw={360} mah="calc(100dvh - 5rem)" style={{ overflowY: 'auto' }}>
            <Menu.Item leftSection={<IconFilePlus size={16} />} onClick={props.onNew}>
              {t('builder.toolbar.newNetwork')}
            </Menu.Item>
            <Menu.Item leftSection={<IconFileImport size={16} />} onClick={props.onImport}>
              {t('builder.toolbar.openJson')}
            </Menu.Item>
            <Menu.Item leftSection={<IconFileExport size={16} />} onClick={props.onExportJson}>
              {t('builder.toolbar.saveJson')}
            </Menu.Item>
            {!readOnly && (
              <Menu.Item leftSection={<IconLayoutGrid size={16} />} onClick={props.onTidy}>
                {t('builder.toolbar.tidy')}
              </Menu.Item>
            )}
            {canSave && (
              <Menu.Item
                hiddenFrom="sm"
                leftSection={<IconDeviceFloppy size={16} />}
                onClick={props.onSave}
              >
                {t('builder.toolbar.saveToAccount')}
              </Menu.Item>
            )}
            {readOnly && onDuplicate && (
              <Menu.Item
                hiddenFrom="sm"
                leftSection={<IconCopyPlus size={16} />}
                onClick={onDuplicate}
              >
                {t('builder.toolbar.duplicate')}
              </Menu.Item>
            )}
            {onShare && (
              <Menu.Item
                hiddenFrom="sm"
                leftSection={<IconUsersPlus size={16} />}
                onClick={onShare}
              >
                {t('builder.toolbar.share')}
              </Menu.Item>
            )}
            <Menu.Item
              hiddenFrom="sm"
              leftSection={<IconCode size={16} />}
              onClick={props.onShowCode}
            >
              {t('builder.toolbar.viewCode')}
            </Menu.Item>
            {props.onShowGuide && (
              <Menu.Item leftSection={<IconHelp size={16} />} onClick={props.onShowGuide}>
                {t('builder.toolbar.guide')}
              </Menu.Item>
            )}
            <Menu.Divider />
            <Menu.Label>{t('builder.toolbar.examples')}</Menu.Label>
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
          <Tooltip
            label={isSaved ? t('builder.toolbar.savedTooltip') : t('builder.toolbar.saveTooltip')}
          >
            <ActionIcon
              variant="default"
              size="lg"
              loading={saving}
              onClick={props.onSave}
              visibleFrom="sm"
              aria-label={t('builder.toolbar.saveToAccount')}
            >
              <IconDeviceFloppy size={18} />
            </ActionIcon>
          </Tooltip>
        )}
        {readOnly && onDuplicate && (
          <>
            <Tooltip label={t('builder.toolbar.copyTooltip')}>
              <Button
                variant="default"
                leftSection={<IconCopyPlus size={16} />}
                onClick={onDuplicate}
                visibleFrom="sm"
              >
                {t('builder.toolbar.duplicate')}
              </Button>
            </Tooltip>
            <Tooltip label={t('builder.toolbar.duplicate')}>
              <ActionIcon
                variant="default"
                size="lg"
                onClick={onDuplicate}
                hiddenFrom="sm"
                aria-label={t('builder.toolbar.duplicate')}
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
          {t('builder.toolbar.code')}
        </Button>
        {!readOnly && <TrainButton run={props.trainingRun} onClick={props.onTrain} />}
        <Tooltip label={t('builder.toolbar.downloadShortcut', { library })}>
          <Button
            leftSection={<IconDownload size={16} />}
            onClick={props.onExportPython}
            className={classes.exportLabelled}
            data-tour="export"
          >
            {t('builder.toolbar.export')}
          </Button>
        </Tooltip>
        <Tooltip label={t('builder.toolbar.exportTooltip', { library })}>
          <ActionIcon
            size="lg"
            onClick={props.onExportPython}
            className={classes.exportIcon}
            aria-label={t('builder.toolbar.download', { library })}
            data-tour="export-icon"
          >
            <IconDownload size={18} />
          </ActionIcon>
        </Tooltip>
        {props.onShowGuide && (
          <Tooltip label={t('builder.toolbar.guide')}>
            <ActionIcon
              variant="subtle"
              color="gray"
              size="lg"
              onClick={props.onShowGuide}
              visibleFrom="sm"
              aria-label={t('builder.toolbar.showGuide')}
            >
              <IconHelp size={18} />
            </ActionIcon>
          </Tooltip>
        )}
        <Tooltip label={i18n.t('ui.builder.Toolbar.properties')}>
          <ActionIcon
            hiddenFrom="lg"
            variant="default"
            size="lg"
            onClick={props.onOpenInspector}
            aria-label={t('builder.toolbar.showProperties')}
          >
            <IconAdjustmentsHorizontal size={18} />
          </ActionIcon>
        </Tooltip>
      </Group>
    </Group>
  );
}

function TrainButton({ run, onClick }: { run: RunSummary | null; onClick: () => void }) {
  const { t } = useTranslation();
  const label = !run
    ? t('builder.toolbar.trainTooltip')
    : run.status === 'queued'
      ? `“${run.name}”: ${queueText(run.queue)}`
      : t('builder.toolbar.trainingRun', { name: run.name, progress: progressLabel(run) });
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
          {run ? progressLabel(run) : t('builder.toolbar.train')}
        </Button>
      </Tooltip>
      <Tooltip label={label}>
        <ActionIcon
          variant="default"
          size="lg"
          onClick={onClick}
          hiddenFrom="sm"
          aria-label={
            run
              ? t('builder.toolbar.trainingProgress', { progress: progressLabel(run) })
              : t('builder.toolbar.train')
          }
          data-tour="train-icon"
        >
          {run ? <Loader size={16} /> : <IconPlayerPlay size={18} />}
        </ActionIcon>
      </Tooltip>
    </>
  );
}
