import { IconBrandGithub, IconChevronDown, IconFolder, IconLogout } from '@tabler/icons-react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import {
  ActionIcon,
  Avatar,
  Badge,
  Button,
  Group,
  Menu,
  Tooltip,
  UnstyledButton,
} from '@mantine/core';
import { useAuth } from '@/auth/AuthContext';
import { DEMO } from '@/demo';
import { ColorSchemeToggle } from './ColorSchemeToggle';
import { Logo } from './Logo';

export const HEADER_HEIGHT = 56;
export const REPOSITORY_URL = 'https://github.com/alexanderKuehling/brick-dnn';

export function AppHeader() {
  const { user, ready, logout } = useAuth();
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const navVariant = (path: string) => (pathname === path ? 'light' : 'subtle');

  return (
    <Group h="100%" px="md" justify="space-between" wrap="nowrap">
      <Group gap="lg" wrap="nowrap">
        <UnstyledButton component={Link} to="/" aria-label="Brick DNN home">
          <Logo />
        </UnstyledButton>
        <Group gap={4} visibleFrom="sm">
          <Button component={Link} to="/" variant={navVariant('/')} size="compact-md" color="gray">
            Builder
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
        <Tooltip label="Source code on GitHub">
          <ActionIcon
            component="a"
            href={REPOSITORY_URL}
            target="_blank"
            rel="noreferrer"
            variant="default"
            size="lg"
            aria-label="GitHub repository"
          >
            <IconBrandGithub size={18} />
          </ActionIcon>
        </Tooltip>
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
          <Menu position="bottom-end" withinPortal>
            <Menu.Target>
              <Button
                variant="default"
                leftSection={
                  <Avatar size={22} color="brick" radius="xl">
                    {user.username.slice(0, 1).toUpperCase()}
                  </Avatar>
                }
                rightSection={<IconChevronDown size={14} />}
              >
                {user.username}
              </Button>
            </Menu.Target>
            <Menu.Dropdown>
              <Menu.Item
                leftSection={<IconFolder size={16} />}
                onClick={() => navigate('/projects')}
              >
                My networks
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
