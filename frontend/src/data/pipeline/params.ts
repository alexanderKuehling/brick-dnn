/** Param values of pipeline steps: defaults, visibility, cell value parsing, summaries. */
import type {
  CellValue,
  ColumnInfo,
  DataSource,
  StepParamSpec,
  StepReport,
  StepSpec,
} from '@/api/types';

/** The value of a param, or its default if the step does not set it. */
export function paramValue(param: StepParamSpec, params: Record<string, unknown>): unknown {
  return params[param.name] === undefined ? param.default : params[param.name];
}

function sameValue(a: unknown, b: unknown): boolean {
  return a === b || JSON.stringify(a) === JSON.stringify(b);
}

/**
 * Whether a param is shown (and used): `visible_if` compares the current value (or the
 * default) of another param. Hidden params keep their values, so switching back restores
 * them.
 */
export function isParamVisible(
  param: StepParamSpec,
  params: Record<string, unknown>,
  spec: StepSpec
): boolean {
  const condition = param.visible_if;
  if (!condition) {
    return true;
  }
  const controller = spec.params.find((p) => p.name === condition.param);
  const current = controller ? paramValue(controller, params) : params[condition.param];
  return condition.values.some((value) => sameValue(value, current));
}

const NUMBER = /^[+-]?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?$/i;

/**
 * Text typed into a value field → a cell value: "" → null, numeric text → number, text in
 * double quotes → that text (to compare "007" as text), anything else → the trimmed text.
 */
export function parseCellValue(text: string): CellValue {
  const trimmed = text.trim();
  if (trimmed === '') {
    return null;
  }
  if (trimmed.length >= 2 && trimmed.startsWith('"') && trimmed.endsWith('"')) {
    return trimmed.slice(1, -1);
  }
  if (NUMBER.test(trimmed)) {
    const number = Number(trimmed);
    if (Number.isFinite(number)) {
      return number;
    }
  }
  return trimmed;
}

/** A cell value → the text for a value field (inverse of `parseCellValue`). */
export function formatCellValue(value: unknown): string {
  if (value === null || value === undefined) {
    return '';
  }
  if (typeof value === 'string') {
    // Quote text that would otherwise be read back as a number or as empty.
    const reparsed = parseCellValue(value);
    return typeof reparsed === 'string' && reparsed === value ? value : `"${value}"`;
  }
  return String(value);
}

/** Columns a column/columns param may use (filtered by `column_kinds`). */
export function allowedColumns(param: StepParamSpec, columns: ColumnInfo[]): ColumnInfo[] {
  const kinds = param.column_kinds;
  return kinds && kinds.length > 0
    ? columns.filter((column) => kinds.includes(column.kind))
    : columns;
}

const KIND_NAMES: Record<string, string> = {
  numeric: 'numeric',
  boolean: 'boolean',
  datetime: 'date',
  categorical: 'categorical',
  text: 'text',
};

/** A plain explanation why chosen columns cannot be used here, or null. */
export function columnsProblem(
  param: StepParamSpec,
  names: string[],
  columns: ColumnInfo[]
): string | null {
  const kinds = param.column_kinds ?? [];
  const byName = new Map(columns.map((column) => [column.name, column]));
  const missing = names.filter((name) => !byName.has(name));
  if (missing.length > 0) {
    const list = missing.map((name) => `“${name}”`).join(', ');
    return missing.length === 1
      ? `${list} does not exist at this step.`
      : `${list} do not exist at this step.`;
  }
  if (kinds.length > 0) {
    const wrong = names.filter((name) => !kinds.includes(byName.get(name)!.kind));
    if (wrong.length > 0) {
      const wanted = kinds.map((kind) => KIND_NAMES[kind] ?? kind).join(' or ');
      const column = byName.get(wrong[0])!;
      return `“${column.name}” is a ${KIND_NAMES[column.kind] ?? column.kind} column; this needs ${wanted} columns.`;
    }
  }
  return null;
}

/** Display name of a data source id. */
export function sourceLabel(sources: DataSource[], id: unknown): string {
  const source = sources.find((s) => s.id === id);
  return source ? source.name : `Table #${String(id)}`;
}

function optionLabel(param: StepParamSpec, value: unknown): string {
  return param.options?.find((option) => option.value === value)?.label ?? String(value);
}

function shortList(names: string[], max = 3): string {
  return names.length <= max
    ? names.join(', ')
    : `${names.slice(0, max).join(', ')} +${names.length - max}`;
}

/**
 * A one-line summary of the settings for a collapsed step card, e.g.
 * "age, income · Median". Only visible params that carry a choice are included.
 */
export function paramSummary(
  spec: StepSpec,
  params: Record<string, unknown>,
  sources: DataSource[] = []
): string {
  const parts: string[] = [];
  for (const param of spec.params) {
    if (!isParamVisible(param, params, spec)) {
      continue;
    }
    const value = paramValue(param, params);
    switch (param.kind) {
      case 'column':
        if (typeof value === 'string' && value) {
          parts.push(value);
        }
        break;
      case 'columns':
      case 'source_columns':
        if (Array.isArray(value) && value.length > 0) {
          parts.push(shortList(value.map(String)));
        } else if (param.empty_means) {
          parts.push(param.empty_means);
        }
        break;
      case 'source':
        if (value !== null && value !== undefined) {
          parts.push(sourceLabel(sources, value));
        }
        break;
      case 'select':
        if (value !== null && value !== undefined) {
          parts.push(optionLabel(param, value));
        }
        break;
      case 'multiselect':
        if (Array.isArray(value) && value.length > 0) {
          parts.push(shortList(value.map((v) => optionLabel(param, v))));
        }
        break;
      case 'mapping':
        if (value && typeof value === 'object') {
          const entries = Object.entries(value as Record<string, unknown>);
          if (entries.length > 0) {
            parts.push(
              shortList(
                entries.map(([from, to]) => `${from} → ${formatCellValue(to) || '∅'}`),
                2
              )
            );
          }
        }
        break;
      default:
        break;
    }
  }
  return parts.join(' · ');
}

/**
 * Step errors name the field they are about ("Columns: 'city' is a categorical column …").
 * Returns the param the error belongs to and the message without the label.
 */
export function errorField(
  spec: StepSpec,
  report: StepReport | undefined
): { param: string; message: string } | null {
  const error = report?.status === 'error' ? report.error : null;
  if (!error) {
    return null;
  }
  // Longest label first, so "Key columns here" wins over "Key columns".
  const params = [...spec.params].sort((a, b) => b.label.length - a.label.length);
  for (const param of params) {
    const prefix = `${param.label}: `;
    if (error.startsWith(prefix)) {
      return { param: param.name, message: error.slice(prefix.length) };
    }
  }
  return null;
}
