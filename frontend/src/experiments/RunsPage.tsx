import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { IconSearch, IconShare } from '@tabler/icons-react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  Alert,
  Button,
  Center,
  Checkbox,
  Container,
  Group,
  Loader,
  Pagination,
  Select,
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core';
import { useDebouncedValue } from '@mantine/hooks';
import { notifications } from '@mantine/notifications';
import {
  api,
  type ComparisonOptions,
  type RunCurve,
  type RunRow,
  type RunStatus,
  type RunTable,
  type RunTableQuery,
} from '@/api';
import { EDITOR_PATH } from '@/components/RedirectToEditor';
import { useTranslation } from '@/i18n';
import { isActive, STATUS_LABELS } from '@/training/format';
import { AnnotateDialog } from './AnnotateDialog';
import { ColumnPicker } from './ColumnPicker';
import { availableColumns, shownColumns } from './columns';
import { CompareView } from './CompareView';
import { RunsTable } from './RunsTable';
import { ShareComparisonDialog } from './ShareComparisonDialog';
import classes from './Experiments.module.css';

/** How often active runs (their rows and curves) are read again. */
export const POLL_MS = 5000;
const PAGE_SIZE = 50;
const STATUSES: RunStatus[] = ['queued', 'running', 'completed', 'stopped', 'failed'];
const FAMILIES = ['neural', 'classical', 'hybrid', 'ensemble'] as const;

/** The filters, sorting and page of the table, as they are in the address. */
function queryFrom(params: URLSearchParams): RunTableQuery {
  const number = (name: string) => {
    const value = params.get(name);
    return value && /^\d+$/.test(value) ? Number(value) : undefined;
  };
  return {
    project: number('project'),
    dataset: params.get('dataset') || undefined,
    status: (params.get('status') as RunStatus) || undefined,
    family: (params.get('family') as RunTableQuery['family']) || undefined,
    tag: params.get('tag') || undefined,
    starred: params.get('starred') === 'true' || undefined,
    mine: params.get('mine') === 'true' || undefined,
    q: params.get('q') || undefined,
    sort: params.get('sort') || undefined,
    order: (params.get('order') as 'asc' | 'desc') || undefined,
    page: number('page'),
  };
}

/**
 * The runs page (`/runs`): every run of the user and of their shared networks in one table,
 * filtered and sorted on the server, with chosen columns; up to 10 runs selected are compared
 * below it (curves, settings against results, parallel coordinates) and can be shared.
 */
export function RunsPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const query = useMemo(() => queryFrom(params), [params]);
  const [table, setTable] = useState<RunTable | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [chosen, setChosen] = useState<string[] | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  // Rows of selected runs, also when they are on another page.
  const [selectedRows, setSelectedRows] = useState<Record<string, RunRow>>({});
  const [curves, setCurves] = useState<RunCurve[] | null>(null);
  const [curvesError, setCurvesError] = useState<string | null>(null);
  const [options, setOptions] = useState<ComparisonOptions>({});
  const [annotating, setAnnotating] = useState<RunRow | null>(null);
  const [sharing, setSharing] = useState(false);
  const [search, setSearch] = useState(query.q ?? '');
  const [debouncedSearch] = useDebouncedValue(search, 300);
  const [tick, setTick] = useState(0);

  const update = useCallback(
    (change: Record<string, string | null | undefined>, keepPage = false) => {
      setParams(
        (current) => {
          const next = new URLSearchParams(current);
          for (const [key, value] of Object.entries(change)) {
            if (value === null || value === undefined || value === '') {
              next.delete(key);
            } else {
              next.set(key, value);
            }
          }
          if (!keepPage) {
            next.delete('page');
          }
          return next;
        },
        { replace: true }
      );
    },
    [setParams]
  );

  useEffect(() => {
    if ((query.q ?? '') !== debouncedSearch) {
      update({ q: debouncedSearch || null });
    }
  }, [debouncedSearch, query.q, update]);

  // The table: again when the filters change, and every few seconds while runs are active.
  useEffect(() => {
    const controller = new AbortController();
    api.training.runs
      .table({ ...query, page_size: PAGE_SIZE }, controller.signal)
      .then((data) => {
        setTable(data);
        setError(null);
        setSelectedRows((rows) => {
          const next = { ...rows };
          data.results.forEach((row) => {
            if (row.id in next) {
              next[row.id] = row;
            }
          });
          return next;
        });
      })
      .catch((err: Error) => {
        if (!controller.signal.aborted) {
          setError(err.message);
        }
      });
    return () => controller.abort();
  }, [query, tick]);

  useEffect(() => {
    const controller = new AbortController();
    api.training.runs
      .columns(controller.signal)
      .then((data) => setChosen(data.columns))
      .catch(() => undefined);
    return () => controller.abort();
  }, []);

  // The curves of the selected runs.
  const selectedKey = selected.join(',');
  const lastCurves = useRef<string>('');
  useEffect(() => {
    if (selected.length === 0) {
      setCurves(null);
      return;
    }
    const controller = new AbortController();
    if (lastCurves.current !== selectedKey) {
      setCurves(null);
      lastCurves.current = selectedKey;
    }
    api.training.runs
      .curves(selected, controller.signal)
      .then((data) => {
        setCurves(data.runs);
        setCurvesError(null);
      })
      .catch((err: Error) => {
        if (!controller.signal.aborted) {
          setCurvesError(err.message);
        }
      });
    return () => controller.abort();
  }, [selectedKey, tick]);

  const live =
    (table?.results ?? []).some((row) => isActive(row.status)) ||
    Object.values(selectedRows).some((row) => isActive(row.status));
  useEffect(() => {
    if (!live) {
      return;
    }
    const timer = window.setInterval(() => setTick((value) => value + 1), POLL_MS);
    return () => window.clearInterval(timer);
  }, [live]);

  const rows = table?.results ?? [];
  const columns = shownColumns(rows, chosen);
  const available = availableColumns(rows, chosen ?? []);
  const saveColumns = (next: string[] | null) => {
    setChosen(next);
    api.training.runs.setColumns(next).catch((err: Error) =>
      notifications.show({
        color: 'red',
        title: t('experiments.columns.saveFailed'),
        message: err.message,
      })
    );
  };
  const maxSelected = table?.limits.compare ?? 10;
  const toggle = (id: string) => {
    const row = rows.find((r) => r.id === id);
    setSelected((list) => {
      if (list.includes(id)) {
        return list.filter((other) => other !== id);
      }
      return list.length >= maxSelected ? list : [...list, id];
    });
    setSelectedRows((map) => {
      const next = { ...map };
      if (id in next) {
        delete next[id];
      } else if (row) {
        next[id] = row;
      }
      return next;
    });
  };
  const replaceRow = (id: string, change: Partial<RunRow>) => {
    setTable((current) =>
      current
        ? {
            ...current,
            results: current.results.map((row) => (row.id === id ? { ...row, ...change } : row)),
          }
        : current
    );
    setSelectedRows((map) => (id in map ? { ...map, [id]: { ...map[id], ...change } } : map));
  };
  const star = async (row: RunRow) => {
    try {
      const saved = await api.training.runs.annotate(row.id, { starred: !row.starred });
      replaceRow(row.id, { starred: saved.starred });
    } catch (err) {
      notifications.show({ color: 'red', message: (err as Error).message });
    }
  };
  const compared = selected
    .map((id) => selectedRows[id])
    .filter((row): row is RunRow => row !== undefined);
  const sortBy = (sort: string) => {
    const current = table?.sort ?? 'created_at';
    const order = table?.order ?? 'desc';
    const next =
      current === sort
        ? order === 'asc'
          ? 'desc'
          : 'asc'
        : sort.startsWith('metric:')
          ? 'asc'
          : 'desc';
    update({ sort, order: next });
  };
  const facets = table?.facets;

  return (
    <Container size="xl" py="xl">
      <Stack gap="md">
        <Group justify="space-between" align="flex-end" wrap="wrap">
          <div>
            <Title order={2}>{t('experiments.page.title')}</Title>
            <Text size="sm" c="dimmed">
              {t('experiments.page.intro')}
            </Text>
          </div>
          <Group gap="xs">
            <ColumnPicker
              available={available}
              shown={columns.map((c) => c.id)}
              onChange={saveColumns}
              onReset={() => saveColumns(null)}
            />
            <Button
              size="xs"
              leftSection={<IconShare size={14} />}
              disabled={compared.length === 0}
              onClick={() => setSharing(true)}
            >
              {t('experiments.page.share')}
            </Button>
          </Group>
        </Group>

        <div className={classes.filters} role="search" aria-label={t('experiments.filters.label')}>
          <TextInput
            size="xs"
            w={200}
            label={t('experiments.filters.search')}
            leftSection={<IconSearch size={14} />}
            value={search}
            onChange={(event) => setSearch(event.currentTarget.value)}
          />
          <Select
            size="xs"
            w={180}
            label={t('experiments.filters.network')}
            placeholder={t('experiments.filters.allNetworks')}
            data={(facets?.projects ?? []).map((p) => ({ value: String(p.id), label: p.name }))}
            value={query.project ? String(query.project) : null}
            onChange={(value) => update({ project: value })}
            clearable
            searchable
          />
          <Select
            size="xs"
            w={180}
            label={t('experiments.filters.dataset')}
            placeholder={t('experiments.filters.allDatasets')}
            data={(facets?.datasets ?? []).map((d) => ({ value: d.key, label: d.name || d.key }))}
            value={query.dataset ?? null}
            onChange={(value) => update({ dataset: value })}
            clearable
            searchable
          />
          <Select
            size="xs"
            w={160}
            label={t('experiments.filters.type')}
            placeholder={t('experiments.filters.allTypes')}
            data={FAMILIES.map((family) => ({
              value: family,
              label: t(`experiments.family.${family}`),
            }))}
            value={query.family ?? null}
            onChange={(value) => update({ family: value })}
            clearable
          />
          <Select
            size="xs"
            w={140}
            label={t('experiments.filters.status')}
            placeholder={t('experiments.filters.allStatuses')}
            data={STATUSES.map((status) => ({ value: status, label: STATUS_LABELS[status] }))}
            value={query.status ?? null}
            onChange={(value) => update({ status: value })}
            clearable
          />
          <Select
            size="xs"
            w={150}
            label={t('experiments.filters.tag')}
            placeholder={t('experiments.filters.allTags')}
            data={(facets?.tags ?? []).map((tag) => ({
              value: tag.name,
              label: `${tag.name} (${tag.count})`,
            }))}
            value={query.tag ?? null}
            onChange={(value) => update({ tag: value })}
            clearable
            searchable
          />
          <Checkbox
            size="xs"
            mb={6}
            label={t('experiments.filters.starred')}
            checked={query.starred === true}
            onChange={(event) => update({ starred: event.currentTarget.checked ? 'true' : null })}
          />
          <Checkbox
            size="xs"
            mb={6}
            label={t('experiments.filters.mine')}
            checked={query.mine === true}
            onChange={(event) => update({ mine: event.currentTarget.checked ? 'true' : null })}
          />
        </div>

        {error && (
          <Alert color="red">
            {error}{' '}
            <Button
              size="compact-xs"
              variant="light"
              color="red"
              onClick={() => setTick((v) => v + 1)}
            >
              {t('experiments.page.retry')}
            </Button>
          </Alert>
        )}
        {!table && !error ? (
          <Center h={160}>
            <Loader aria-label={t('experiments.page.loading')} />
          </Center>
        ) : table && table.count === 0 ? (
          <Text c="dimmed" size="sm" ta="center" py="xl">
            {Object.values(query).some((v) => v !== undefined)
              ? t('experiments.page.noMatch')
              : t('experiments.page.empty')}
          </Text>
        ) : table ? (
          <>
            <Group justify="space-between" wrap="wrap" gap="xs">
              <Text size="xs" c="dimmed" aria-live="polite">
                {t('experiments.page.count', { count: table.count })}
                {selected.length > 0 &&
                  ` · ${t('experiments.page.selected', { count: selected.length, max: maxSelected })}`}
              </Text>
              {selected.length > 0 && (
                <Button
                  size="compact-xs"
                  variant="subtle"
                  onClick={() => {
                    setSelected([]);
                    setSelectedRows({});
                  }}
                >
                  {t('experiments.page.clearSelection')}
                </Button>
              )}
            </Group>
            <RunsTable
              label={t('experiments.page.tableLabel')}
              rows={rows}
              columns={columns}
              sort={table.sort}
              order={table.order}
              onSort={sortBy}
              selected={selected}
              maxSelected={maxSelected}
              onToggle={toggle}
              onOpen={(row) =>
                navigate(EDITOR_PATH, { state: { openTraining: true, run: row.id } })
              }
              onAnnotate={setAnnotating}
              onStar={(row) => void star(row)}
            />
            {table.pages > 1 && (
              <Group justify="center">
                <Pagination
                  size="sm"
                  total={table.pages}
                  value={table.page}
                  onChange={(page) => update({ page: String(page) }, true)}
                  getItemProps={(page) => ({ 'aria-label': t('experiments.page.page', { page }) })}
                  getControlProps={(control) => ({
                    'aria-label': t(`experiments.page.${control}`),
                  })}
                />
              </Group>
            )}
            {table.truncated && (
              <Text size="xs" c="dimmed">
                {t('experiments.page.truncated')}
              </Text>
            )}
          </>
        ) : null}

        <section className={classes.compare} aria-labelledby="compare-title">
          <Title order={3} size="h4" id="compare-title" mb="xs">
            {t('experiments.compare.title')}
          </Title>
          <CompareView
            rows={compared}
            curves={curves}
            allRows={rows}
            options={options}
            onOptionsChange={setOptions}
            error={curvesError}
          />
        </section>
      </Stack>

      <AnnotateDialog
        row={annotating}
        knownTags={(facets?.tags ?? []).map((tag) => tag.name)}
        maxTags={table?.limits.tags}
        maxNote={table?.limits.note}
        onClose={() => setAnnotating(null)}
        onSaved={(saved) => {
          replaceRow(saved.id, { tags: saved.tags, note: saved.note, starred: saved.starred });
          setTick((value) => value + 1); // the tags of the filter
        }}
      />
      <ShareComparisonDialog
        opened={sharing}
        onClose={() => setSharing(false)}
        rows={compared}
        columns={columns.map((c) => c.id)}
        options={options}
      />
    </Container>
  );
}
