import { useEffect, useState } from 'react';
import {
  IconCheck,
  IconCopy,
  IconDatabase,
  IconLogout,
  IconRefresh,
  IconUsers,
  IconX,
} from '@tabler/icons-react';
import {
  ActionIcon,
  Alert,
  Avatar,
  Button,
  Center,
  CopyButton,
  Divider,
  Group,
  Loader,
  Modal,
  Select,
  Switch,
  Tabs,
  Text,
  TextInput,
  Tooltip,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import {
  api,
  ApiError,
  type MemberRole,
  type Project,
  type ProjectMember,
  type ProjectMembers,
  type ShareLink,
  type UserRef,
} from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { useReturnFocus } from '@/components/useReturnFocus';
import { i18n, useTranslation } from '@/i18n';
import { BadgeSnippet } from './BadgeSnippet';
import { ProjectDataPanel } from './ProjectDataPanel';
import { MEMBER_ROLE_OPTIONS, ROLE_LABELS, shareUrl } from './roles';
import classes from './Sharing.module.css';

/** What the dialog needs to know about the project. */
export type ShareTarget = Pick<Project, 'id' | 'name' | 'role' | 'link_access' | 'share_token'>;

/** The sharing state after a change in the dialog. */
export interface SharingState extends ShareLink {
  members: ProjectMembers;
  shared: boolean;
}

interface ShareDialogProps {
  project: ShareTarget;
  onClose: () => void;
  /** Members or the link changed. */
  onChange?: (state: SharingState) => void;
  /** The current user left the project (and lost access). */
  onLeave?: () => void;
  /** people: members and the link · data: datasets and preparations shared in the project. */
  initialTab?: ShareTab;
}

export type ShareTab = 'people' | 'data';

/** The message of a rejected invitation, shown below the field. */
function inviteError(error: ApiError): string {
  const data = (error.data ?? {}) as Record<string, unknown>;
  for (const key of ['user', 'role', 'detail', 'non_field_errors']) {
    const value = data[key];
    if (Array.isArray(value) && value.length > 0) {
      return value.map(String).join(' ');
    }
    if (typeof value === 'string') {
      return value;
    }
  }
  return error.message;
}

/** Who has access, inviting people (owner) and the view link. Mount it while it is open. */
export function ShareDialog({
  project,
  onClose,
  onChange,
  onLeave,
  initialTab = 'people',
}: ShareDialogProps) {
  // Removed when it closes: the focus goes back to the Share button.
  useReturnFocus();
  useTranslation();
  const { user } = useAuth();
  const [tab, setTab] = useState<ShareTab>(initialTab);
  const isOwner = project.role === 'owner';
  const [members, setMembers] = useState<ProjectMembers | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [link, setLink] = useState<ShareLink>({
    link_access: project.link_access,
    share_token: project.share_token,
  });
  const [invitee, setInvitee] = useState('');
  const [inviteRole, setInviteRole] = useState<MemberRole>('editor');
  const [inviteMessage, setInviteMessage] = useState<string | null>(null);
  const [inviting, setInviting] = useState(false);
  const [busyMember, setBusyMember] = useState<number | null>(null);
  const [linkBusy, setLinkBusy] = useState(false);
  const [confirming, setConfirming] = useState<'regenerate' | 'leave' | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.projects.members
      .list(project.id)
      .then((result) => !cancelled && setMembers(result))
      .catch((error: Error) => !cancelled && setLoadError(error.message));
    return () => {
      cancelled = true;
    };
  }, [project.id]);

  const report = (nextMembers: ProjectMembers, nextLink: ShareLink) =>
    onChange?.({
      ...nextLink,
      members: nextMembers,
      shared: nextMembers.members.length > 0 || nextLink.link_access === 'view',
    });

  const failed = (title: string) => (error: unknown) =>
    notifications.show({ color: 'red', title, message: (error as Error).message });

  const updateMembers = (update: (list: ProjectMember[]) => ProjectMember[]) => {
    if (!members) {
      return;
    }
    const next = { ...members, members: update(members.members) };
    setMembers(next);
    report(next, link);
  };

  const invite = async () => {
    const name = invitee.trim();
    if (!name) {
      setInviteMessage(i18n.t('ui.sharing.ShareDialog.enterAUsernameOrAn'));
      return;
    }
    setInviting(true);
    setInviteMessage(null);
    try {
      const member = await api.projects.members.add(project.id, name, inviteRole);
      updateMembers((list) => [...list.filter((m) => m.id !== member.id), member]);
      setInvitee('');
    } catch (error) {
      if (error instanceof ApiError && error.status === 400) {
        setInviteMessage(inviteError(error));
      } else {
        failed(i18n.t('ui.sharing.ShareDialog.couldNotInvite'))(error);
      }
    } finally {
      setInviting(false);
    }
  };

  const changeRole = async (member: ProjectMember, role: MemberRole) => {
    setBusyMember(member.id);
    try {
      const updated = await api.projects.members.update(project.id, member.id, role);
      updateMembers((list) => list.map((m) => (m.id === updated.id ? updated : m)));
    } catch (error) {
      failed(i18n.t('ui.sharing.ShareDialog.couldNotChangeTheRole'))(error);
    } finally {
      setBusyMember(null);
    }
  };

  const remove = async (member: ProjectMember) => {
    setBusyMember(member.id);
    try {
      await api.projects.members.remove(project.id, member.id);
      updateMembers((list) => list.filter((m) => m.id !== member.id));
    } catch (error) {
      failed(i18n.t('ui.sharing.ShareDialog.couldNotRemove', { username: member.user.username }))(
        error
      );
    } finally {
      setBusyMember(null);
    }
  };

  const changeLink = async (enabled: boolean, regenerate = false) => {
    setLinkBusy(true);
    try {
      const next = await api.projects.setLink(project.id, enabled, regenerate);
      setLink(next);
      setConfirming(null);
      if (members) {
        report(members, next);
      }
      if (regenerate) {
        notifications.show({ message: i18n.t('ui.sharing.ShareDialog.newLinkCreatedTheOld') });
      }
    } catch (error) {
      failed(i18n.t('ui.sharing.ShareDialog.couldNotChangeTheLink'))(error);
    } finally {
      setLinkBusy(false);
    }
  };

  const own = members?.members.find((member) => member.user.id === user?.id) ?? null;

  const leave = async () => {
    setBusyMember(own?.id ?? null);
    try {
      if (own) {
        await api.projects.members.remove(project.id, own.id);
      }
      notifications.show({
        message: i18n.t('ui.sharing.ShareDialog.youLeft', { name: project.name }),
      });
      onLeave?.();
      onClose();
    } catch (error) {
      failed(i18n.t('ui.sharing.ShareDialog.couldNotLeaveTheNetwork'))(error);
      setBusyMember(null);
    }
  };

  const linkOn = link.link_access === 'view';
  const url = link.share_token ? shareUrl(link.share_token) : null;

  return (
    <Modal
      opened
      onClose={onClose}
      title={i18n.t('ui.sharing.ShareDialog.share', { name: project.name })}
      size="lg"
    >
      <Tabs
        value={tab}
        onChange={(value) => value && setTab(value as ShareTab)}
        keepMounted={false}
      >
        <Tabs.List mb="md">
          <Tabs.Tab value="people" leftSection={<IconUsers size={14} />}>
            {i18n.t('ui.sharing.ShareDialog.people')}
          </Tabs.Tab>
          <Tabs.Tab value="data" leftSection={<IconDatabase size={14} />}>
            {i18n.t('ui.sharing.ShareDialog.data')}
          </Tabs.Tab>
        </Tabs.List>
        <Tabs.Panel value="people">
          {isOwner && (
            <form
              className={classes.invite}
              onSubmit={(event) => {
                event.preventDefault();
                void invite();
              }}
            >
              <TextInput
                className={classes.inviteUser}
                aria-label={i18n.t('ui.sharing.ShareDialog.usernameOrEMail')}
                placeholder={i18n.t('ui.sharing.ShareDialog.usernameOrEMail')}
                autoComplete="off"
                autoCapitalize="none"
                spellCheck={false}
                value={invitee}
                error={inviteMessage}
                onChange={(event) => {
                  setInvitee(event.currentTarget.value);
                  setInviteMessage(null);
                }}
              />
              <div className={classes.inviteControls}>
                <Select
                  aria-label={i18n.t('ui.sharing.ShareDialog.roleOfTheInvitedPerson')}
                  data={MEMBER_ROLE_OPTIONS}
                  value={inviteRole}
                  onChange={(value) => value && setInviteRole(value as MemberRole)}
                  allowDeselect={false}
                  comboboxProps={{ withinPortal: true }}
                  w={130}
                />
                <Button type="submit" loading={inviting}>
                  {i18n.t('ui.sharing.ShareDialog.invite')}
                </Button>
              </div>
            </form>
          )}

          <Text fw={600} size="sm" mt={isOwner ? 'lg' : 0} mb={4} id="share-people">
            {i18n.t('ui.sharing.ShareDialog.peopleWithAccess')}
          </Text>
          {loadError && <Alert color="red">{loadError}</Alert>}
          {!members && !loadError && (
            <Center h={80}>
              <Loader size="sm" />
            </Center>
          )}
          {members && (
            <ul className={classes.people} aria-labelledby="share-people">
              <Person user={members.owner} you={members.owner.id === user?.id}>
                <Text size="sm" c="dimmed">
                  {ROLE_LABELS.owner}
                </Text>
              </Person>
              {members.members.map((member) => {
                const name = member.user.username;
                const you = member.user.id === user?.id;
                return (
                  <Person key={member.id} user={member.user} you={you}>
                    {isOwner ? (
                      <>
                        <Select
                          aria-label={i18n.t('ui.sharing.ShareDialog.roleOf', { name })}
                          data={MEMBER_ROLE_OPTIONS}
                          value={member.role}
                          onChange={(value) =>
                            value &&
                            value !== member.role &&
                            void changeRole(member, value as MemberRole)
                          }
                          allowDeselect={false}
                          disabled={busyMember === member.id}
                          comboboxProps={{ withinPortal: true }}
                          size="xs"
                          w={110}
                        />
                        <Tooltip label={i18n.t('ui.sharing.ShareDialog.remove', { name })}>
                          <ActionIcon
                            variant="subtle"
                            color="gray"
                            onClick={() => void remove(member)}
                            disabled={busyMember === member.id}
                            aria-label={i18n.t('ui.sharing.ShareDialog.remove', { name })}
                          >
                            <IconX size={16} />
                          </ActionIcon>
                        </Tooltip>
                      </>
                    ) : (
                      <>
                        <Text size="sm" c="dimmed">
                          {ROLE_LABELS[member.role]}
                        </Text>
                        {you && (
                          <Button
                            size="compact-sm"
                            variant="light"
                            color="red"
                            leftSection={<IconLogout size={14} />}
                            onClick={() => setConfirming('leave')}
                          >
                            {i18n.t('ui.sharing.ShareDialog.leave')}
                          </Button>
                        )}
                      </>
                    )}
                  </Person>
                );
              })}
            </ul>
          )}
          {members && isOwner && members.members.length === 0 && (
            <Text size="sm" c="dimmed" mt={4}>
              {i18n.t('ui.sharing.ShareDialog.onlyYouCanOpenThis')}
            </Text>
          )}
          {confirming === 'leave' && (
            <div className={classes.confirm}>
              <Text size="sm">
                {i18n.t('ui.sharing.ShareDialog.leaveYouLoseAccessTo', { name: project.name })}
              </Text>
              <Group justify="flex-end" gap="xs" mt="sm">
                <Button variant="default" size="xs" onClick={() => setConfirming(null)}>
                  {i18n.t('ui.sharing.ShareDialog.cancel')}
                </Button>
                <Button
                  color="red"
                  size="xs"
                  loading={busyMember !== null && busyMember === own?.id}
                  onClick={() => void leave()}
                >
                  {i18n.t('ui.sharing.ShareDialog.leave')}
                </Button>
              </Group>
            </div>
          )}

          <Divider my="lg" />
          {isOwner ? (
            <>
              <Switch
                label={i18n.t('ui.sharing.ShareDialog.anyoneWithTheLinkCan')}
                description={i18n.t('ui.sharing.ShareDialog.noAccountNeededTheyCan')}
                checked={linkOn}
                disabled={linkBusy}
                onChange={(event) => void changeLink(event.currentTarget.checked)}
              />
              {linkOn && url && (
                <>
                  <div className={classes.link}>
                    <TextInput
                      className={classes.linkUrl}
                      aria-label={i18n.t('ui.sharing.ShareDialog.shareLink')}
                      value={url}
                      readOnly
                      onFocus={(event) => event.currentTarget.select()}
                    />
                    <CopyButton value={url} timeout={2000}>
                      {({ copied, copy }) => (
                        <Button
                          onClick={copy}
                          color={copied ? 'teal' : undefined}
                          leftSection={copied ? <IconCheck size={16} /> : <IconCopy size={16} />}
                        >
                          {copied
                            ? i18n.t('ui.sharing.ShareDialog.copied')
                            : i18n.t('ui.sharing.ShareDialog.copyLink')}
                        </Button>
                      )}
                    </CopyButton>
                  </div>
                  <Button
                    variant="subtle"
                    size="compact-sm"
                    mt="xs"
                    leftSection={<IconRefresh size={14} />}
                    onClick={() => setConfirming('regenerate')}
                    disabled={linkBusy}
                  >
                    {i18n.t('ui.sharing.ShareDialog.createNewLink')}
                  </Button>
                  {confirming === 'regenerate' && (
                    <div className={classes.confirm}>
                      <Text size="sm">{i18n.t('ui.sharing.ShareDialog.createANewLinkThe')}</Text>
                      <Group justify="flex-end" gap="xs" mt="sm">
                        <Button variant="default" size="xs" onClick={() => setConfirming(null)}>
                          {i18n.t('ui.sharing.ShareDialog.cancel')}
                        </Button>
                        <Button
                          size="xs"
                          loading={linkBusy}
                          onClick={() => void changeLink(true, true)}
                        >
                          {i18n.t('ui.sharing.ShareDialog.createNewLink')}
                        </Button>
                      </Group>
                    </div>
                  )}
                  <BadgeSnippet url={url} />
                </>
              )}
            </>
          ) : (
            <Text size="sm" c="dimmed">
              {i18n.t('ui.sharing.ShareDialog.onlyTheOwnerCanChange', {
                value: linkOn
                  ? i18n.t('ui.sharing.ShareDialog.anyoneWithTheLinkCan2')
                  : i18n.t('ui.sharing.ShareDialog.onlyThePeopleAboveCan'),
              })}
            </Text>
          )}
        </Tabs.Panel>
        <Tabs.Panel value="data">
          <ProjectDataPanel project={project} />
        </Tabs.Panel>
      </Tabs>

      <Group justify="flex-end" mt="xl">
        <Button variant="default" onClick={onClose}>
          {i18n.t('ui.sharing.ShareDialog.done')}
        </Button>
      </Group>
    </Modal>
  );
}

function Person({
  user,
  you,
  children,
}: {
  user: UserRef;
  you: boolean;
  children: React.ReactNode;
}) {
  useTranslation();
  return (
    <li className={classes.person} aria-label={user.username}>
      <div className={classes.personName}>
        <Avatar name={user.username} color="initials" size={32} />
        <Text size="sm" fw={500} truncate>
          {user.username}
          {you && (
            <Text span c="dimmed" size="sm">
              {' '}
              {i18n.t('ui.sharing.ShareDialog.you')}
            </Text>
          )}
        </Text>
      </div>
      <div className={classes.personActions}>{children}</div>
    </li>
  );
}
