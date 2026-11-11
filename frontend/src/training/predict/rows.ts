/**
 * Rows for predictions: the values of the form, pasted or uploaded CSV text, the results as
 * a CSV file, and the messages of the server per row and column.
 */
import type { Prediction, PredictionCell, PredictionResult } from '@/api';

/** A column of the prediction form: a feature (prepared) or a column of the source (raw). */
export interface InputColumn {
  name: string;
  kind: 'numeric' | 'boolean' | 'datetime' | 'categorical' | 'text';
  categories?: string[];
  min?: number | string | null;
  max?: number | string | null;
  integer?: boolean;
}

/** The text of a cell for the form ('' for a missing value). */
export function cellText(value: PredictionCell | undefined): string {
  if (value === null || value === undefined) {
    return '';
  }
  return typeof value === 'boolean' ? String(value) : String(value);
}

/** A form value as sent to the server: numbers for prepared columns, text otherwise. */
export function cellValue(text: string, column: InputColumn, prepared: boolean): PredictionCell {
  const trimmed = text.trim();
  if (trimmed === '') {
    return null;
  }
  if (prepared || column.kind === 'numeric') {
    const number = Number(trimmed);
    // Text that is no plain number (e.g. "1.234,5") goes as it is: the server reads it like
    // the import did, or says why it cannot.
    return Number.isFinite(number) ? number : trimmed;
  }
  if (column.kind === 'boolean') {
    return trimmed === 'true' ? true : trimmed === 'false' ? false : trimmed;
  }
  return trimmed;
}

// ---------------------------------------------------------------------------- CSV

function detectDelimiter(line: string): string {
  const counts = [',', ';', '\t', '|'].map((d) => [d, line.split(d).length - 1] as const);
  counts.sort((a, b) => b[1] - a[1]);
  return counts[0][1] > 0 ? counts[0][0] : ',';
}

/** Records of CSV text (quotes as in RFC 4180), the delimiter detected from the first line. */
export function parseCsv(text: string): string[][] {
  // Without a byte order mark.
  const clean = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
  const firstLine = clean.split(/\r?\n/, 1)[0] ?? '';
  const delimiter = detectDelimiter(firstLine);
  const records: string[][] = [];
  let record: string[] = [];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < clean.length; i += 1) {
    const char = clean[i];
    if (quoted) {
      if (char === '"' && clean[i + 1] === '"') {
        cell += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        cell += char;
      }
    } else if (char === '"' && cell === '') {
      quoted = true;
    } else if (char === delimiter) {
      record.push(cell);
      cell = '';
    } else if (char === '\n' || char === '\r') {
      if (char === '\r' && clean[i + 1] === '\n') {
        i += 1;
      }
      record.push(cell);
      records.push(record);
      record = [];
      cell = '';
    } else {
      cell += char;
    }
  }
  if (cell !== '' || record.length > 0) {
    record.push(cell);
    records.push(record);
  }
  return records.filter((r) => r.some((c) => c.trim() !== ''));
}

export interface CsvRows {
  /** One object per row, by column name (only the columns of the form). */
  rows: Record<string, string>[];
  /** The target column's values, if the file has it (to compare with the predictions). */
  actual: string[] | null;
  /** Columns of the file that the model does not take (ignored). */
  ignored: string[];
}

/**
 * The rows of CSV text with a header line. Columns that the form does not have are ignored,
 * missing ones are an error. Throws an Error with a message for the user.
 */
export function csvRows(
  text: string,
  columns: InputColumn[],
  options: { maxRows: number; target?: string | null; required?: string[] }
): CsvRows {
  const records = parseCsv(text);
  if (records.length < 2) {
    throw new Error('Paste a header line with the column names and at least one row.');
  }
  const header = records[0].map((h) => h.trim());
  const names = new Set(columns.map((c) => c.name));
  const required = options.required ?? columns.map((c) => c.name);
  const missing = required.filter((name) => !header.includes(name));
  if (missing.length > 0) {
    const shown = missing.slice(0, 5).join(', ') + (missing.length > 5 ? ' …' : '');
    throw new Error(
      `The header line lacks ${missing.length === 1 ? 'the column' : 'columns'}: ${shown}.`
    );
  }
  const body = records.slice(1);
  if (body.length > options.maxRows) {
    throw new Error(`${body.length} rows are too many: at most ${options.maxRows} at a time.`);
  }
  const targetAt = options.target ? header.indexOf(options.target) : -1;
  const rows = body.map((record) => {
    const row: Record<string, string> = {};
    header.forEach((name, i) => {
      if (names.has(name)) {
        row[name] = (record[i] ?? '').trim();
      }
    });
    return row;
  });
  const actual = targetAt >= 0 ? body.map((record) => (record[targetAt] ?? '').trim()) : null;
  return {
    rows,
    // An empty target column (rows to predict) is no comparison.
    actual: actual && actual.some((value) => value !== '') ? actual : null,
    ignored: header.filter((name, i) => !names.has(name) && i !== targetAt && name !== ''),
  };
}

function csvCell(value: unknown): string {
  const text = value === null || value === undefined ? '' : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** The inputs and their predictions as a CSV file. */
export function resultsCsv(
  columns: string[],
  rows: Record<string, unknown>[],
  predictions: Prediction[],
  task: PredictionResult['task'],
  target?: string | null
): string {
  const name = target || 'prediction';
  const extra =
    task === 'classification'
      ? [`predicted_${name}`, 'probability']
      : [
          `predicted_${name}`,
          ...(predictions.some((p) => typeof p.std === 'number') ? ['std'] : []),
        ];
  const lines = [[...columns, ...extra].map(csvCell).join(',')];
  rows.forEach((row, i) => {
    const p = predictions[i] ?? {};
    const values =
      task === 'classification'
        ? [p.class ?? '', p.probability ?? '']
        : [p.value ?? '', ...(extra.length > 1 ? [p.std ?? ''] : [])];
    lines.push([...columns.map((c) => row[c]), ...values].map(csvCell).join(','));
  });
  return `${lines.join('\n')}\n`;
}

// ---------------------------------------------------------------------------- messages

export interface RowMessage {
  /** 1-based row number, null for messages about all rows. */
  row: number | null;
  column: string | null;
  text: string;
}

/** "Row 3, 'temp': 'hot' is not a number." -> row 3, column temp ("The row" is row 1). */
export function parseMessage(text: string): RowMessage {
  const match = /^(?:Row (\d+)|The row)(?:, '((?:[^']|'(?!:))*)')?:\s*(.*)$/s.exec(text);
  if (!match) {
    return { row: null, column: null, text };
  }
  return { row: match[1] ? Number(match[1]) : 1, column: match[2] ?? null, text: match[3] };
}

/**
 * Messages of one request of several (rows from `offset + 1` on) with the numbers of the
 * whole file: "Row 2" of the third request of 64 rows is "Row 130".
 */
export function shiftMessage(text: string, offset: number): string {
  if (offset === 0) {
    return text;
  }
  const message = parseMessage(text);
  if (message.row === null) {
    return text;
  }
  const column = message.column !== null ? `, '${message.column}'` : '';
  return `Row ${message.row + offset}${column}: ${message.text}`;
}

/** `items` in pieces of at most `size`. */
export function chunks<T>(items: T[], size: number): T[][] {
  const result: T[][] = [];
  for (let i = 0; i < items.length; i += Math.max(1, size)) {
    result.push(items.slice(i, i + Math.max(1, size)));
  }
  return result;
}

/** "92.4 %" */
export function formatProbability(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '–';
  }
  return `${(value * 100).toFixed(1)} %`;
}

/** A predicted value with up to 4 significant digits after the point (no exponent). */
export function formatValue(value: Prediction['value']): string {
  if (Array.isArray(value)) {
    return value.map((v) => formatValue(v)).join(', ');
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '–';
  }
  const abs = Math.abs(value);
  const digits = abs >= 1000 ? 1 : abs >= 1 ? 3 : 4;
  return Number(value.toFixed(digits)).toLocaleString('en-US', { maximumFractionDigits: digits });
}
