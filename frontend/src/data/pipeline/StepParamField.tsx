import { useId, useState } from 'react';
import { IconCheck, IconPlus, IconTrash } from '@tabler/icons-react';
import {
  ActionIcon,
  Autocomplete,
  Button,
  Chip,
  Group,
  Input,
  MultiSelect,
  NumberInput,
  Select,
  Stack,
  Switch,
  TagsInput,
  Text,
  TextInput,
  type ComboboxItem,
  type ComboboxLikeRenderOptionInput,
} from '@mantine/core';
import type { ColumnInfo, ColumnKind, DataSource, StepParamSpec } from '@/api/types';
import { formatCount } from '../table/format';
import { KindIcon } from '../table/kinds';
import { allowedColumns, columnsProblem, formatCellValue, parseCellValue } from './params';

export interface StepParamFieldProps {
  param: StepParamSpec;
  /** The current value (the default if the step does not set it). */
  value: unknown;
  onChange: (value: unknown) => void;
  /** The columns this step works on (output of the step before it). */
  columns: ColumnInfo[];
  /** All data sources of the user (for `source` / `source_columns`). */
  sources: DataSource[];
  currentSourceId: number;
  /** All params of the step (`source_columns` reads the chosen source). */
  params: Record<string, unknown>;
  /** A problem with this field from the last preview. */
  error?: string | null;
  /**
   * mapping: the keys are column names (renaming) – suggest them and keep the new values as
   * text. Otherwise keys are cell values and new values are parsed like value fields.
   */
  mappingKeysAreColumns?: boolean;
}

const comboboxProps = { withinPortal: true, shadow: 'md' } as const;

/** One form field of a pipeline step, chosen by the param kind of the step catalog. */
export function StepParamField(props: StepParamFieldProps) {
  const { param, value, onChange, sources, currentSourceId, params, error } = props;
  const description = param.help || undefined;

  switch (param.kind) {
    case 'column':
    case 'columns':
      return <ColumnField {...props} />;

    case 'source_columns': {
      const source = sources.find((s) => s.id === params[param.source_param ?? 'source']);
      if (!source) {
        return (
          <MultiSelect
            label={param.label}
            description={description}
            data={[]}
            value={[]}
            placeholder="Choose a table first"
            disabled
            error={error}
          />
        );
      }
      return <ColumnField {...props} columns={source.columns} />;
    }

    case 'source': {
      const others = sources.filter((s) => s.id !== currentSourceId);
      const data = others.map((s) => ({
        value: String(s.id),
        label: `${s.name} (${formatCount(s.n_rows)} rows × ${s.n_columns} columns)`,
      }));
      const chosen = value === null || value === undefined || value === '' ? null : String(value);
      let problem: string | null = null;
      if (chosen !== null && !others.some((s) => String(s.id) === chosen)) {
        problem =
          Number(chosen) === currentSourceId
            ? 'Choose another table than the one you are preparing.'
            : 'This table does not exist (any more).';
        data.push({ value: chosen, label: `Table #${chosen}` });
      }
      return (
        <Select
          label={param.label}
          description={
            others.length === 0
              ? 'Import another CSV file to combine it with this one.'
              : description
          }
          placeholder={others.length === 0 ? 'No other tables' : 'Choose a table'}
          data={data}
          value={chosen}
          disabled={others.length === 0 && chosen === null}
          searchable={others.length > 8}
          allowDeselect={false}
          comboboxProps={comboboxProps}
          error={error || problem}
          onChange={(next) => onChange(next === null ? null : Number(next))}
        />
      );
    }

    case 'select': {
      const options = param.options ?? [];
      const current = value === null || value === undefined ? null : String(value);
      const data = options.map((option) => ({ value: option.value, label: option.label }));
      if (current !== null && !options.some((option) => option.value === current)) {
        data.push({ value: current, label: `${current} (unknown)` });
      }
      return (
        <Select
          label={param.label}
          description={description}
          data={data}
          value={current}
          allowDeselect={param.optional}
          clearable={param.optional}
          placeholder={param.optional ? 'Not set' : undefined}
          comboboxProps={comboboxProps}
          error={error}
          onChange={(next) => {
            if (next !== null || param.optional) {
              onChange(next);
            }
          }}
        />
      );
    }

    case 'multiselect': {
      const selected = Array.isArray(value) ? value.map(String) : [];
      return <ChipsField {...props} selected={selected} />;
    }

    case 'number':
    case 'int':
      return <NumberField {...props} />;

    case 'bool':
      return (
        <Switch
          label={param.label}
          description={description}
          checked={Boolean(value)}
          error={error}
          onChange={(event) => onChange(event.currentTarget.checked)}
        />
      );

    case 'string':
      return (
        <TextInput
          label={param.label}
          description={description}
          value={typeof value === 'string' ? value : value == null ? '' : String(value)}
          placeholder={param.optional ? 'Optional' : undefined}
          styles={
            param.name === 'expression'
              ? { input: { fontFamily: 'var(--mantine-font-family-monospace)' } }
              : undefined
          }
          error={error}
          onChange={(event) => {
            const text = event.currentTarget.value;
            onChange(text === '' && param.optional ? null : text);
          }}
        />
      );

    case 'value':
      return <ValueField {...props} />;

    case 'list':
      return <ListField {...props} />;

    case 'mapping':
      return <MappingField {...props} />;

    default:
      return (
        <Text size="sm" c="red">
          {param.label}: this setting cannot be edited here (unknown kind “{String(param.kind)}”).
        </Text>
      );
  }
}

// --------------------------------------------------------------------------
// Columns
// --------------------------------------------------------------------------

function columnOption(kinds: Map<string, ColumnKind>) {
  return function ColumnOption({ option, checked }: ComboboxLikeRenderOptionInput<ComboboxItem>) {
    const kind = kinds.get(option.value);
    return (
      <Group gap={8} wrap="nowrap" style={{ flex: 1, minWidth: 0 }}>
        {kind ? <KindIcon kind={kind} /> : <span style={{ width: 14 }} />}
        <Text size="sm" truncate style={{ flex: 1 }}>
          {option.label}
        </Text>
        {checked && <IconCheck size={14} aria-hidden />}
      </Group>
    );
  };
}

function ColumnField({ param, value, onChange, columns, error }: StepParamFieldProps) {
  const allowed = allowedColumns(param, columns);
  const kinds = new Map(columns.map((column) => [column.name, column.kind]));
  const data = allowed.map((column) => ({ value: column.name, label: column.name }));
  const renderOption = columnOption(kinds);
  const description = param.help || undefined;
  const nothingFound =
    allowed.length === 0 && columns.length > 0 ? 'No suitable column' : 'No matching column';

  if (param.kind === 'column') {
    const current = typeof value === 'string' && value !== '' ? value : null;
    const problem = current ? columnsProblem(param, [current], columns) : null;
    if (current && !allowed.some((column) => column.name === current)) {
      data.push({ value: current, label: current });
    }
    return (
      <Select
        label={param.label}
        description={description}
        placeholder="Choose a column"
        data={data}
        value={current}
        searchable
        clearable={param.optional}
        allowDeselect={param.optional}
        nothingFoundMessage={nothingFound}
        renderOption={renderOption}
        leftSection={current && kinds.has(current) ? <KindIcon kind={kinds.get(current)!} /> : null}
        comboboxProps={comboboxProps}
        error={error || problem}
        onChange={(next) => {
          if (next !== null || param.optional) {
            onChange(next);
          }
        }}
      />
    );
  }

  const selected = Array.isArray(value) ? value.map(String) : [];
  const problem = selected.length > 0 ? columnsProblem(param, selected, columns) : null;
  return (
    <MultiSelect
      label={param.label}
      description={description}
      placeholder={selected.length === 0 ? (param.empty_means ?? 'Choose columns') : undefined}
      data={data}
      value={selected}
      searchable
      clearable
      nothingFoundMessage={nothingFound}
      renderOption={renderOption}
      comboboxProps={comboboxProps}
      error={error || problem}
      onChange={onChange}
    />
  );
}

// --------------------------------------------------------------------------
// Options as chips
// --------------------------------------------------------------------------

function ChipsField({
  param,
  onChange,
  error,
  selected,
}: StepParamFieldProps & { selected: string[] }) {
  const baseId = useId();
  const options = param.options ?? [];
  return (
    <Input.Wrapper
      label={param.label}
      labelElement="div"
      id={baseId}
      description={param.help || undefined}
      error={error}
    >
      <Chip.Group
        multiple
        value={selected}
        onChange={(next) =>
          // Keep the catalog order, like the server does.
          onChange(options.map((option) => option.value).filter((v) => next.includes(v)))
        }
      >
        <Group gap={6} mt={6} role="group" aria-labelledby={`${baseId}-label`}>
          {options.map((option) => (
            <Chip key={option.value} value={option.value} size="xs" variant="outline">
              {option.label}
            </Chip>
          ))}
        </Group>
      </Chip.Group>
    </Input.Wrapper>
  );
}

// --------------------------------------------------------------------------
// Numbers
// --------------------------------------------------------------------------

function numberProblem(param: StepParamSpec, raw: number | string): string | null {
  if (raw === '') {
    return param.optional ? null : 'Enter a number.';
  }
  if (typeof raw !== 'number' || !Number.isFinite(raw)) {
    return 'Enter a number.';
  }
  if (param.kind === 'int' && !Number.isInteger(raw)) {
    return 'Enter a whole number.';
  }
  if (param.min !== undefined && raw < param.min) {
    return `Must be at least ${param.min}.`;
  }
  if (param.max !== undefined && raw > param.max) {
    return `Must be at most ${param.max}.`;
  }
  return null;
}

/** Keeps unfinished input ("0.", "-") as typed and reports only valid numbers. */
function NumberField({ param, value, onChange, error }: StepParamFieldProps) {
  const external = typeof value === 'number' ? value : '';
  const [draft, setDraft] = useState<number | string>(external);
  const [synced, setSynced] = useState<number | string>(external);
  if (external !== synced) {
    // The value changed outside (another step, undo, reload): show it.
    setSynced(external);
    if (Number(draft) !== external || draft === '') {
      setDraft(external);
    }
  }
  const problem = numberProblem(
    param,
    typeof draft === 'string' && draft !== '' ? Number(draft) : draft
  );
  const isInt = param.kind === 'int';
  const step = param.step ?? (isInt ? 1 : param.max !== undefined && param.max <= 1 ? 0.01 : 0.1);

  return (
    <NumberInput
      label={param.label}
      description={param.help || undefined}
      value={draft}
      min={param.min}
      max={param.max}
      step={step}
      allowDecimal={!isInt}
      placeholder={param.optional ? 'Not set' : undefined}
      error={error || problem}
      onChange={(raw) => {
        setDraft(raw);
        if (raw === '') {
          if (param.optional) {
            onChange(null);
          }
          return;
        }
        if (typeof raw === 'number' && numberProblem(param, raw) === null) {
          onChange(raw);
        }
      }}
    />
  );
}

// --------------------------------------------------------------------------
// Cell values
// --------------------------------------------------------------------------

function ValueField({ param, value, onChange, error }: StepParamFieldProps) {
  const [draft, setDraft] = useState(() => formatCellValue(value));
  const [synced, setSynced] = useState<unknown>(value);
  if (!Object.is(value, synced)) {
    setSynced(value);
    if (!Object.is(parseCellValue(draft), value ?? null)) {
      setDraft(formatCellValue(value));
    }
  }
  const parsed = parseCellValue(draft);
  const missing = parsed === null && !param.optional;
  return (
    <TextInput
      label={param.label}
      description={param.help || undefined}
      value={draft}
      placeholder={param.optional ? 'A number or a text' : undefined}
      rightSection={
        parsed === null ? null : (
          <Text size="xs" c="dimmed" aria-hidden>
            {typeof parsed === 'number' ? 'number' : 'text'}
          </Text>
        )
      }
      rightSectionWidth={56}
      rightSectionPointerEvents="none"
      error={error || (missing ? 'Enter a value.' : null)}
      onChange={(event) => {
        const text = event.currentTarget.value;
        setDraft(text);
        onChange(parseCellValue(text));
      }}
    />
  );
}

function ListField({ param, value, onChange, error }: StepParamFieldProps) {
  const values = Array.isArray(value) ? value : [];
  return (
    <TagsInput
      label={param.label}
      description={param.help || 'Press Enter after each value.'}
      value={values.map(formatCellValue)}
      placeholder={values.length === 0 ? 'Type a value, press Enter' : undefined}
      clearable
      error={error}
      onChange={(tags) =>
        onChange(tags.map(parseCellValue).filter((tag) => tag !== null && tag !== ''))
      }
    />
  );
}

// --------------------------------------------------------------------------
// Mappings
// --------------------------------------------------------------------------

interface MappingRow {
  id: number;
  key: string;
  value: string;
}

function toRows(value: unknown, keysAreColumns: boolean): MappingRow[] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return [];
  }
  return Object.entries(value as Record<string, unknown>).map(([key, next], index) => ({
    id: index,
    key,
    value: keysAreColumns ? (next == null ? '' : String(next)) : formatCellValue(next),
  }));
}

function toMapping(rows: MappingRow[], keysAreColumns: boolean): Record<string, unknown> {
  const mapping: Record<string, unknown> = {};
  for (const row of rows) {
    if (row.key !== '' && !(row.key in mapping)) {
      mapping[row.key] = keysAreColumns ? row.value.trim() : parseCellValue(row.value);
    }
  }
  return mapping;
}

/** Rows of "old → new" pairs; incomplete rows stay in the editor until they have a key. */
function MappingField({
  param,
  value,
  onChange,
  columns,
  error,
  mappingKeysAreColumns = false,
}: StepParamFieldProps) {
  const baseId = useId();
  const [rows, setRows] = useState(() => toRows(value, mappingKeysAreColumns));
  const [synced, setSynced] = useState(() => JSON.stringify(value ?? {}));
  const serialized = JSON.stringify(value ?? {});
  if (serialized !== synced) {
    setSynced(serialized);
    if (JSON.stringify(toMapping(rows, mappingKeysAreColumns)) !== serialized) {
      setRows(toRows(value, mappingKeysAreColumns));
    }
  }

  const update = (next: MappingRow[]) => {
    setRows(next);
    onChange(toMapping(next, mappingKeysAreColumns));
  };
  const nextId = rows.reduce((max, row) => Math.max(max, row.id), -1) + 1;
  const keyLabel = mappingKeysAreColumns ? 'Column' : 'Value';
  const valueLabel = mappingKeysAreColumns ? 'New name' : 'Replace with';
  const suggestions = mappingKeysAreColumns ? columns.map((column) => column.name) : [];

  return (
    <Input.Wrapper
      label={param.label}
      labelElement="div"
      id={baseId}
      description={param.help || undefined}
      error={error}
    >
      <Stack gap={6} mt={6} role="group" aria-labelledby={`${baseId}-label`}>
        {rows.map((row, index) => {
          const duplicate =
            row.key !== '' && rows.findIndex((other) => other.key === row.key) < index;
          const unknownColumn =
            mappingKeysAreColumns && row.key !== '' && !suggestions.includes(row.key);
          const keyProps = {
            'aria-label': `${keyLabel} ${index + 1}`,
            placeholder: keyLabel,
            value: row.key,
            error: duplicate
              ? 'Listed twice'
              : unknownColumn
                ? 'No such column at this step'
                : undefined,
            style: { flex: 1, minWidth: 0 },
          };
          const setKey = (key: string) =>
            update(rows.map((r) => (r.id === row.id ? { ...r, key } : r)));
          return (
            <Group key={row.id} gap={6} wrap="nowrap" align="flex-start">
              {mappingKeysAreColumns ? (
                <Autocomplete
                  {...keyProps}
                  data={suggestions}
                  comboboxProps={comboboxProps}
                  onChange={setKey}
                />
              ) : (
                <TextInput {...keyProps} onChange={(event) => setKey(event.currentTarget.value)} />
              )}
              <Text size="sm" c="dimmed" mt={6} aria-hidden>
                →
              </Text>
              <TextInput
                aria-label={`${valueLabel} ${index + 1}`}
                placeholder={mappingKeysAreColumns ? valueLabel : 'missing'}
                value={row.value}
                style={{ flex: 1, minWidth: 0 }}
                onChange={(event) => {
                  const text = event.currentTarget.value;
                  update(rows.map((r) => (r.id === row.id ? { ...r, value: text } : r)));
                }}
              />
              <ActionIcon
                variant="subtle"
                color="gray"
                mt={4}
                aria-label={`Remove row ${index + 1}`}
                onClick={() => update(rows.filter((r) => r.id !== row.id))}
              >
                <IconTrash size={16} />
              </ActionIcon>
            </Group>
          );
        })}
        <Button
          variant="subtle"
          size="compact-sm"
          leftSection={<IconPlus size={14} />}
          style={{ alignSelf: 'flex-start' }}
          onClick={() => setRows([...rows, { id: nextId, key: '', value: '' }])}
        >
          Add {mappingKeysAreColumns ? 'column' : 'value'}
        </Button>
      </Stack>
    </Input.Wrapper>
  );
}
