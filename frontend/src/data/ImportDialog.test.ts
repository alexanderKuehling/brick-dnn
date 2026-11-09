import { describe, expect, it } from 'vitest';
import type { ImportReport } from '@/api';
import {
  acceptedFiles,
  DEFAULT_FILE_TYPES,
  describeReport,
  fileProblem,
  fileStem,
  linkProblem,
} from './ImportDialog';

const CSV: ImportReport = {
  delimiter: ';',
  decimal: ',',
  header: true,
  encoding: 'utf-8',
  rows: 10,
  columns: 3,
  warnings: [],
};

describe('describeReport', () => {
  it('describes CSV files, also those imported before other formats existed', () => {
    expect(describeReport(CSV)).toBe('semicolon-separated · decimal comma · UTF-8 · header row');
    expect(describeReport({ ...CSV, format: 'csv', compression: 'gzip', header: false })).toBe(
      'semicolon-separated · decimal comma · UTF-8 · gzip-packed · no header row'
    );
  });

  it('describes workbooks, JSON and binary formats', () => {
    const binary = { ...CSV, delimiter: null, encoding: null, decimal: '.' };
    expect(describeReport({ ...binary, format: 'xlsx', sheet: 'Q1', sheets: ['Q1', 'Q2'] })).toBe(
      'Excel workbook, sheet “Q1” · header row'
    );
    expect(describeReport({ ...binary, format: 'json', json_path: ['result', 'items'] })).toBe(
      'JSON, the list under “result.items”'
    );
    expect(describeReport({ ...binary, format: 'jsonl' })).toBe('JSON Lines');
    expect(
      describeReport({ ...binary, format: 'parquet', compression: 'zip', member: 'events.parquet' })
    ).toBe('Parquet · “events.parquet” from a ZIP file');
  });
});

describe('files and links', () => {
  const file = (name: string, size = 10) => new File(['x'.repeat(size)], name);

  it('accepts the file types of the server', () => {
    for (const name of ['a.csv', 'b.XLSX', 'c.json', 'd.parquet', 'e.csv.gz', 'f.zip', 'g.ods']) {
      expect(fileProblem(file(name), 10)).toBeNull();
    }
    expect(fileProblem(file('photo.png'), 10)).toMatch(/“photo.png” cannot be imported/);
    expect(fileProblem(file('data.parquet'), 10, ['.csv', '.xlsx'])).toMatch(/\(.csv, .xlsx\)/);
    expect(fileProblem(file('big.csv', 2 * 1024 * 1024), 1)).toMatch(/up to 1 MB/);
    expect(acceptedFiles(DEFAULT_FILE_TYPES)).toContain('.xlsx,');
    expect(acceptedFiles(['.csv'])).toContain('text/csv');
  });

  it('names tables after their file', () => {
    expect(fileStem('Umsatz 2024.xlsx')).toBe('Umsatz 2024');
    expect(fileStem('orders.csv.gz')).toBe('orders');
    expect(fileStem('archive.zip')).toBe('archive');
    expect(fileStem('README')).toBe('README');
  });

  it('checks links', () => {
    expect(linkProblem('https://docs.google.com/spreadsheets/d/abc/edit#gid=0')).toBeNull();
    expect(linkProblem(' http://example.org/data.csv ')).toBeNull();
    expect(linkProblem('example.org/data.csv')).toMatch(/https:\/\//);
    expect(linkProblem('ftp://example.org/data.csv')).not.toBeNull();
    expect(linkProblem('https://')).not.toBeNull();
  });
});
