import { IconChartLine } from '@tabler/icons-react';
import { Link, useInRouterContext } from 'react-router-dom';
import { Button } from '@mantine/core';
import { useTranslation } from '@/i18n';

/** The address of the runs page with a network's runs, or the user's own. */
export function runsPageUrl(project: number | null): string {
  return project !== null ? `/runs?project=${project}` : '/runs?mine=true';
}

/** "Compare runs": the entry point from the run history to the runs page. */
export function CompareLink({ project }: { project: number | null }) {
  const { t } = useTranslation();
  const inRouter = useInRouterContext();
  const to = runsPageUrl(project);
  const common = {
    size: 'compact-xs' as const,
    variant: 'light' as const,
    leftSection: <IconChartLine size={14} />,
  };
  return inRouter ? (
    <Button component={Link} to={to} {...common}>
      {t('experiments.history.compare')}
    </Button>
  ) : (
    <Button component="a" href={to} {...common}>
      {t('experiments.history.compare')}
    </Button>
  );
}
