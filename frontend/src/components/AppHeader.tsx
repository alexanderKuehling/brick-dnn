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
  IconReceiptRefund,
  IconShieldLock,
  IconTable,
  IconTag,
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
  Indicator,
  Menu,
  Tooltip,
  UnstyledButton,
} from '@mantine/core';
import { api, type LegalPageName } from '@/api';
import { usePendingApprovals } from '@/approvals/pending';
import { useAuth } from '@/auth/AuthContext';
import { useBillingInfo } from '@/billing/useBillingInfo';
import { relativeTime } from '@/collab/time';
import { useTranslation } from '@/i18n';
import { LanguageSwitch } from '@/i18n/LanguageSwitch';
import { legalPages, useLegalInfo } from '@/legal/useLegalInfo';
import { REPOSITORY_URL, SUPPORT_URL } from '@/links';
import { ColorSchemeToggle } from './ColorSchemeToggle';
import { Logo } from './Logo';

export const HEADER_HEIGHT = 56;

const LEGAL_ICONS: Record<LegalPageName, React.ReactNode> = {
  imprint: <IconId size={16} />,
  privacy: <IconShieldLock size={16} />,
  terms: <IconFileText size={16} />,
  refund: <IconReceiptRefund size={16} />,
};

export function AppHeader() {
  const { t } = useTranslation();
  const { user, ready, logout } = useAuth();
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const legal = legalPages(useLegalInfo());
  // Only a server that sells plans has a pricing page.
  const pricing = Boolean(useBillingInfo());
  const navVariant = (path: string) =>
    pathname === path || pathname.startsWith(`${path}/`) ? 'light' : 'subtle';
  // Requests of agents that wait for the user's approval (a badge on the account menu).
  const pending = usePendingApprovals(Boolean(user));
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
        <UnstyledButton component={Link} to="/" aria-label={t('nav.home')}>
          <Logo />
        </UnstyledButton>
        <Group gap={4} visibleFrom="sm">
          <Button component={Link} to="/" variant={navVariant('/')} size="compact-md" color="gray">
            {t('nav.builder')}
          </Button>
          <Button
            component={Link}
            to="/data"
            variant={navVariant('/data')}
            size="compact-md"
            color="gray"
          >
            {t('nav.data')}
          </Button>
          {user && (
            <Button
              component={Link}
              to="/projects"
              variant={navVariant('/projects')}
              size="compact-md"
              color="gray"
            >
              {t('nav.myNetworks')}
            </Button>
          )}
          {pricing && (
            <Button
              component={Link}
              to="/pricing"
              variant={navVariant('/pricing')}
              size="compact-md"
              color="gray"
            >
              {t('nav.pricing')}
            </Button>
          )}
        </Group>
      </Group>

      <Group gap="xs" wrap="nowrap">
        <PhoneMenu
          loggedIn={Boolean(user)}
          pricing={pricing}
          legal={legal}
          isActive={(path) => navVariant(path) === 'light'}
        />
        <Tooltip label={t('nav.sourceCode')}>
          <ActionIcon
            component="a"
            href={REPOSITORY_URL}
            target="_blank"
            rel="noreferrer"
            variant="default"
            size="lg"
            aria-label={t('nav.githubRepository')}
            visibleFrom="sm"
          >
            <IconBrandGithub size={18} />
          </ActionIcon>
        </Tooltip>
        <Tooltip label={t('nav.coffee')}>
          <ActionIcon
            component="a"
            href={SUPPORT_URL}
            target="_blank"
            rel="noreferrer"
            variant="default"
            size="lg"
            aria-label={t('nav.coffee')}
            visibleFrom="sm"
          >
            <IconCoffee size={18} />
          </ActionIcon>
        </Tooltip>
        {legal.length > 0 && (
          <Box visibleFrom="sm">
            <Menu position="bottom-end" withinPortal>
              <Menu.Target>
                <Tooltip label={t('nav.legalTooltip')}>
                  <ActionIcon variant="default" size="lg" aria-label={t('nav.legalInfo')}>
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
                    {t(`legal.labels.${page}`)}
                  </Menu.Item>
                ))}
              </Menu.Dropdown>
            </Menu>
          </Box>
        )}
        <Box visibleFrom="xs">
          <LanguageSwitch />
        </Box>
        <ColorSchemeToggle />
        {ready && !user && (
          <>
            <Button component={Link} to="/login" variant="default">
              {t('nav.logIn')}
            </Button>
            <Button component={Link} to="/register" visibleFrom="xs">
              {t('nav.signUp')}
            </Button>
          </>
        )}
        {user && (
          <Menu position="bottom-end" withinPortal onOpen={checkActivity}>
            <Menu.Target>
              <Button
                variant="default"
                leftSection={
                  <Indicator
                    label={pending}
                    size={16}
                    color="yellow"
                    disabled={pending === 0}
                    offset={2}
                    data-testid="approvals-indicator"
                  >
                    <Avatar size={22} color="brick" radius="xl">
                      {user.username.slice(0, 1).toUpperCase()}
                    </Avatar>
                  </Indicator>
                }
                rightSection={<IconChevronDown size={14} />}
                aria-label={
                  pending > 0
                    ? t('nav.userWaiting', { user: user.username, count: pending })
                    : user.username
                }
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
                {t('nav.myNetworks')}
              </Menu.Item>
              <Menu.Item leftSection={<IconTable size={16} />} onClick={() => navigate('/data')}>
                {t('nav.dataPreparation')}
              </Menu.Item>
              <Menu.Item leftSection={<IconKey size={16} />} onClick={() => navigate('/tokens')}>
                {t('nav.apiTokens')}
              </Menu.Item>
              <Menu.Item
                leftSection={<IconActivity size={16} />}
                rightSection={
                  pending > 0 ? (
                    <Badge size="sm" color="yellow" variant="filled">
                      {t('nav.waiting', { count: pending })}
                    </Badge>
                  ) : (
                    <RecentActivity at={latestActivity} />
                  )
                }
                onClick={() => navigate('/activity')}
              >
                {t('nav.agentActivity')}
              </Menu.Item>
              <Menu.Item
                leftSection={<IconUserCircle size={16} />}
                onClick={() => navigate('/account')}
              >
                {t('nav.account')}
              </Menu.Item>
              <Menu.Divider />
              <Menu.Item
                leftSection={<IconLogout size={16} />}
                onClick={() => {
                  logout();
                  navigate('/');
                }}
              >
                {t('nav.logOut')}
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
  pricing,
  legal,
  isActive,
}: {
  loggedIn: boolean;
  pricing: boolean;
  legal: LegalPageName[];
  isActive: (path: string) => boolean;
}) {
  const { t } = useTranslation();
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
          <ActionIcon variant="default" size="lg" aria-label={t('nav.pages')}>
            <IconMenu2 size={18} />
          </ActionIcon>
        </Menu.Target>
        <Menu.Dropdown>
          {item('/', t('nav.builder'), <IconTopologyStar3 size={16} />)}
          {item('/data', t('nav.data'), <IconTable size={16} />)}
          {loggedIn && item('/projects', t('nav.myNetworks'), <IconFolder size={16} />)}
          {pricing && item('/pricing', t('nav.pricing'), <IconTag size={16} />)}
          <Menu.Divider />
          <Menu.Item
            component="a"
            href={REPOSITORY_URL}
            target="_blank"
            rel="noreferrer"
            leftSection={<IconBrandGithub size={16} />}
          >
            {t('nav.sourceCode')}
          </Menu.Item>
          <Menu.Item
            component="a"
            href={SUPPORT_URL}
            target="_blank"
            rel="noreferrer"
            leftSection={<IconCoffee size={16} />}
          >
            {t('nav.coffee')}
          </Menu.Item>
          <Menu.Divider />
          <Box px="sm" py={6} hiddenFrom="xs">
            <LanguageSwitch />
          </Box>
          {legal.length > 0 && <Menu.Divider />}
          {legal.map((page) => (
            <Fragment key={page}>
              {item(`/${page}`, t(`legal.labels.${page}`), LEGAL_ICONS[page])}
            </Fragment>
          ))}
        </Menu.Dropdown>
      </Menu>
    </Box>
  );
}
