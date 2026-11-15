import type { data as en } from '../en/data';
import type { Messages } from '../types';

export const data: Messages<typeof en> = {
  tableSize: '{{rows}} × {{columns}}',
  rows_one: '{{value}} Zeile',
  rows_other: '{{value}} Zeilen',
  columns_one: '{{value}} Spalte',
  columns_other: '{{value}} Spalten',
  steps: {
    none: 'keine Schritte',
    count_one: '{{count}} Schritt',
    count_other: '{{count}} Schritte',
  },
};
