import { Fragment, useState } from 'react';
import {
  IconActivity,
  IconBrandGithub,
  IconChevronDown,
  IconCoffee,
  IconFileText,
  IconFolder,
  IconId,
  IconInfoCircle,
  IconKey,
  IconLogout,
  IconMenu2,
  IconShieldLock,
  IconTable,
  IconTopologyStar3,
  IconUserCircle,
} from '@tabler/icons-react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import {
  ActionIcon,
  Avatar,
  Badge,
  Box,
  Button,
  Group,
  Menu,
  Tooltip,
  UnstyledButton,
} from '@mantine/core';
import { api, type LegalPageName } from '@/api';
import { useAuth } from '@/auth/AuthContext';
import { relativeTime } from '@/collab/time';
import { DEMO } from '@/demo';
import { LEGAL_LABELS, legalPages, useLegalInfo } from '@/legal/useLegalInfo';
import { REPOSITORY_URL, SUPPORT_URL } from '@/links';
import { ColorSchemeToggle } from './ColorSchemeToggle';
import { Logo } from './Logo';

export const HEADER_HEIGHT = 56;

const LEGAL_ICONS: Record<LegalPageName, React.ReactNode> = {
  imprint: <IconId size={16} />,
  privacy: <IconShieldLock size={16} />,
  terms: <IconFileText size={16} />,
};

export function AppHeader() {
  const { user, ready, logout } = useAuth();
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const legal = legalPages(useLegalInfo());
  const navVariant = (path: string) =>
    pathname === path || pathname.startsWith(`${path}/`) ? 'light' : 'subtle';
  // When an agent last did something: asked for when the account menu opens.
  const [latestActivity, setLatestActivity] = useState<string | null>(null);
  const checkActivity = () => {
    api.agentActivity
      .summary()
      .then((summary) => setLatestActivity(summary.latest_at))
      .catch(() => undefined); // Then the menu shows no time.
  };

  return (
    <Group h="100%" px={{ base: 'xs', xs: 'md' }} gap="xs" justify="space-between" wrap="nowrap">
      <Group gap="lg" wrap="nowrap">
        <UnstyledButton component={Link} to="/" aria-label="Brick DNN home">
          <Logo />
        </UnstyledButton>
        <Group gap={4} visibleFrom="sm">
          <Button component={Link} to="/" variant={navVariant('/')} size="compact-md" color="gray">
            Builder
          </Button>
          <Button
            component={Link}
            to="/data"
            variant={navVariant('/data')}
            size="compact-md"
            color="gray"
          >
            Data
          </Button>
          {DEMO && (
            <Tooltip label="Runs in your browser. Install Brick DNN locally to train networks.">
              <Badge variant="light" color="gray">
                Online demo
              </Badge>
            </Tooltip>
          )}
          {user && (
            <Button
              component={Link}
              to="/projects"
              variant={navVariant('/projects')}
              size="compact-md"
              color="gray"
            >
              My networks
            </Button>
          )}
        </Group>
      </Group>

      <Group gap="xs" wrap="nowrap">
        <PhoneMenu
          loggedIn={Boolean(user)}
          legal={legal}
          isActive={(path) => navVariant(path) === 'light'}
        />
        <Tooltip label="Source code on GitHub">
          <ActionIcon
            component="a"
            href={REPOSITORY_URL}
            target="_blank"
            rel="noreferrer"
            variant="default"
            size="lg"
            aria-label="GitHub repository"
            visibleFrom="sm"
          >
            <IconBrandGithub size={18} />
          </ActionIcon>
        </Tooltip>
        <Tooltip label="Buy me a coffee">
          <ActionIcon
            component="a"
            href={SUPPORT_URL}
            target="_blank"
            rel="noreferrer"
            variant="default"
            size="lg"
            aria-label="Buy me a coffee"
            visibleFrom="sm"
          >
            <IconCoffee size={18} />
          </ActionIcon>
        </Tooltip>
        {legal.length > 0 && (
          <Box visibleFrom="sm">
            <Menu position="bottom-end" withinPortal>
              <Menu.Target>
                <Tooltip label="Imprint and privacy">
                  <ActionIcon variant="default" size="lg" aria-label="Legal information">
                    <IconInfoCircle size={18} />
                  </ActionIcon>
                </Tooltip>
              </Menu.Target>
              <Menu.Dropdown>
                {legal.map((page) => (
                  <Menu.Item
                    key={page}
                    leftSection={LEGAL_ICONS[page]}
                    onClick={() => navigate(`/${page}`)}
                  >
                    {LEGAL_LABELS[page]}
                  </Menu.Item>
                ))}
              </Menu.Dropdown>
            </Menu>
          </Box>
        )}
        <ColorSchemeToggle />
        {!DEMO && ready && !user && (
          <>
            <Button component={Link} to="/login" variant="default">
              Log in
            </Button>
            <Button component={Link} to="/register" visibleFrom="xs">
              Sign up
            </Button>
          </>
        )}
        {user && (
          <Menu position="bottom-end" withinPortal onOpen={checkActivity}>
            <Menu.Target>
              <Button
                variant="default"
                leftSection={
                  <Avatar size={22} color="brick" radius="xl">
                    {user.username.slice(0, 1).toUpperCase()}
                  </Avatar>
                }
                rightSection={<IconChevronDown size={14} />}
                aria-label={user.username}
              >
                {/* Phones show only the avatar. */}
                <Box component="span" visibleFrom="xs">
                  {user.username}
                </Box>
              </Button>
            </Menu.Target>
            <Menu.Dropdown>
              <Menu.Item
                leftSection={<IconFolder size={16} />}
                onClick={() => navigate('/projects')}
              >
                My networks
              </Menu.Item>
              <Menu.Item leftSection={<IconTable size={16} />} onClick={() => navigate('/data')}>
                Data preparation
              </Menu.Item>
              <Menu.Item leftSection={<IconKey size={16} />} onClick={() => navigate('/tokens')}>
                API tokens
              </Menu.Item>
              <Menu.Item
                leftSection={<IconActivity size={16} />}
                rightSection={<RecentActivity at={latestActivity} />}
                onClick={() => navigate('/activity')}
              >
                Agent activity
              </Menu.Item>
              <Menu.Item
                leftSection={<IconUserCircle size={16} />}
                onClick={() => navigate('/account')}
              >
                Account
              </Menu.Item>
              <Menu.Divider />
              <Menu.Item
                leftSection={<IconLogout size={16} />}
                onClick={() => {
                  logout();
                  navigate('/');
                }}
              >
                Log out
              </Menu.Item>
            </Menu.Dropdown>
          </Menu>
        )}
      </Group>
    </Group>
  );
}

const RECENT = 24 * 60 * 60 * 1000;
const JUST_NOW = 10 * 60 * 1000;

/** "5 min ago" next to "Agent activity" when an agent acted today; a dot while it is busy. */
function RecentActivity({ at }: { at: string | null }) {
  if (!at) {
    return null;
  }
  const age = Date.now() - new Date(at).getTime();
  if (age > RECENT) {
    return null;
  }
  return (
    <Group gap={4} wrap="nowrap" c="dimmed" fz="xs">
      {age < JUST_NOW && (
        <Box component="span" w={6} h={6} bg="green" style={{ borderRadius: '50%' }} aria-hidden />
      )}
      {relativeTime(at)}
    </Group>
  );
}

/** Below `sm` the page links and the GitHub and support links move into a menu. */
function PhoneMenu({
  loggedIn,
  legal,
  isActive,
}: {
  loggedIn: boolean;
  legal: LegalPageName[];
  isActive: (path: string) => boolean;
}) {
  const navigate = useNavigate();
  const item = (path: string, label: string, icon: React.ReactNode) => (
    <Menu.Item
      leftSection={icon}
      onClick={() => navigate(path)}
      fw={isActive(path) ? 600 : undefined}
      aria-current={isActive(path) ? 'page' : undefined}
    >
      {label}
    </Menu.Item>
  );

  return (
    <Box hiddenFrom="sm">
      <Menu position="bottom-end" withinPortal>
        <Menu.Target>
          <ActionIcon variant="default" size="lg" aria-label="Pages">
            <IconMenu2 size={18} />
          </ActionIcon>
        </Menu.Target>
        <Menu.Dropdown>
          {DEMO && <Menu.Label>Online demo: runs in your browser</Menu.Label>}
          {item('/', 'Builder', <IconTopologyStar3 size={16} />)}
          {item('/data', 'Data', <IconTable size={16} />)}
          {loggedIn && item('/projects', 'My networks', <IconFolder size={16} />)}
          <Menu.Divider />
          <Menu.Item
            component="a"
            href={REPOSITORY_URL}
            target="_blank"
            rel="noreferrer"
            leftSection={<IconBrandGithub size={16} />}
          >
            Source code on GitHub
          </Menu.Item>
          <Menu.Item
            component="a"
            href={SUPPORT_URL}
            target="_blank"
            rel="noreferrer"
            leftSection={<IconCoffee size={16} />}
          >
            Buy me a coffee
          </Menu.Item>
          {legal.length > 0 && <Menu.Divider />}
          {legal.map((page) => (
            <Fragment key={page}>
              {item(`/${page}`, LEGAL_LABELS[page], LEGAL_ICONS[page])}
            </Fragment>
          ))}
        </Menu.Dropdown>
      </Menu>
    </Box>
  );
}
