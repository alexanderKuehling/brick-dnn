import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Alert, Button, Center, Container, Loader, Stack, Text, Title } from '@mantine/core';
import { api, ApiError, type ComparisonOptions, type RunRow, type SharedComparison } from '@/api';
import { MadeWith } from '@/components/MadeWith';
import { EDITOR_PATH } from '@/components/RedirectToEditor';
import { useTranslation } from '@/i18n';
import { formatDate } from '@/training/format';
import { columnById, shownColumns, type Column } from './columns';
import { CompareView } from './CompareView';
import { RunsTable } from './RunsTable';
import classes from './Experiments.module.css';

/** Rows sorted by a column in the browser (a shared comparison has all its rows). */
export function sortRows(rows: RunRow[], column: Column | null, order: 'asc' | 'desc'): RunRow[] {
  if (!column) {
    return rows;
  }
  const value = (row: RunRow) =>
    column.id === 'name' ? row.name.toLowerCase() : column.value(row);
  const present = rows.filter((row) => value(row) !== null && value(row) !== undefined);
  const missing = rows.filter((row) => value(row) === null || value(row) === undefined);
  const sign = order === 'asc' ? 1 : -1;
  present.sort((a, b) => {
    const x = value(a)!;
    const y = value(b)!;
    if (typeof x === 'number' && typeof y === 'number') {
      return (x - y) * sign;
    }
    return String(x).localeCompare(String(y)) * sign;
  });
  return [...present, ...missing];
}

/** A comparison of runs shared with a link (`/compare/<token>`): read-only, no login. */
export function SharedComparisonPage() {
  const { t } = useTranslation();
  const { token = '' } = useParams();
  const [data, setData] = useState<SharedComparison | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [options, setOptions] = useState<ComparisonOptions>({});
  const [sort, setSort] = useState<{ key: string; order: 'asc' | 'desc' }>({
    key: '',
    order: 'asc',
  });

  useEffect(() => {
    const controller = new AbortController();
    api.training.comparisons
      .shared(token, controller.signal)
      .then((shared) => {
        setData(shared);
        setOptions(shared.options);
      })
      .catch((err: Error) => {
        if (!controller.signal.aborted) {
          setError(
            err instanceof ApiError && err.status === 404
              ? t('experiments.shared.gone')
              : err.message
          );
        }
      });
    return () => controller.abort();
  }, [token, t]);

  const columns = useMemo(
    () => (data ? shownColumns(data.runs, data.columns.length ? data.columns : null) : []),
    [data]
  );
  if (error) {
    return (
      <Container size="sm" py="xl">
        <Alert color="red" title={t('experiments.shared.title')}>
          {error}
        </Alert>
      </Container>
    );
  }
  if (!data) {
    return (
      <Center h={200}>
        <Loader aria-label={t('experiments.page.loading')} />
      </Center>
    );
  }
  const sortColumn =
    sort.key === 'name'
      ? ({ id: 'name', value: (row: RunRow) => row.name } as unknown as Column)
      : (columns.find((c) => c.sort === sort.key) ?? columnById(sort.key));
  const rows = sortRows(data.runs, sort.key ? sortColumn : null, sort.order);
  return (
    <Container size="xl" py="xl">
      <Stack gap="md">
        <div>
          <Text size="xs" c="dimmed" tt="uppercase" fw={700}>
            {t('experiments.shared.title')}
          </Text>
          <Title order={2}>{data.name}</Title>
          <Text size="sm" c="dimmed">
            {t('experiments.shared.intro', {
              count: data.runs.length,
              date: formatDate(data.created_at),
            })}
          </Text>
        </div>
        {data.runs.length === 0 ? (
          <Text c="dimmed">{t('experiments.shared.noRuns')}</Text>
        ) : (
          <>
            <RunsTable
              label={t('experiments.shared.tableLabel', { name: data.name })}
              rows={rows}
              columns={columns}
              sort={sort.key}
              order={sort.order}
              onSort={(key) =>
                setSort((current) => ({
                  key,
                  order: current.key === key && current.order === 'asc' ? 'desc' : 'asc',
                }))
              }
            />
            <section className={classes.compare} aria-labelledby="shared-compare-title">
              <Title order={3} size="h4" id="shared-compare-title" mb="xs">
                {t('experiments.compare.title')}
              </Title>
              <CompareView
                rows={data.runs}
                curves={data.curves}
                options={options}
                onOptionsChange={setOptions}
              />
            </section>
          </>
        )}
      </Stack>
      {/* For people who have never seen the app: what it is, and the way into the editor. */}
      <MadeWith>
        <Button component={Link} to={EDITOR_PATH}>
          {t('brand.madeWith.buildYourOwn')}
        </Button>
      </MadeWith>
    </Container>
  );
}
