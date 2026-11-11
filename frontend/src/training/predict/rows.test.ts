import { fitDrawing, grayOf, imageInput, rasterize, resizeArea } from './pixels';
import {
  cellValue,
  chunks,
  csvRows,
  parseCsv,
  parseMessage,
  resultsCsv,
  shiftMessage,
  type InputColumn,
} from './rows';

const COLUMNS: InputColumn[] = [
  { name: 'date', kind: 'datetime' },
  { name: 'store', kind: 'categorical', categories: ['Berlin', 'Köln'] },
  { name: 'temp', kind: 'numeric' },
  { name: 'promo', kind: 'boolean' },
];

describe('rows for predictions', () => {
  it('turns form values into cells', () => {
    expect(cellValue(' 3.5 ', COLUMNS[2], false)).toBe(3.5);
    // Text the server reads like the import (decimal comma) goes as it is.
    expect(cellValue('3,5', COLUMNS[2], false)).toBe('3,5');
    expect(cellValue('', COLUMNS[2], false)).toBeNull();
    expect(cellValue('true', COLUMNS[3], false)).toBe(true);
    expect(cellValue('Berlin', COLUMNS[1], false)).toBe('Berlin');
    expect(cellValue('12', COLUMNS[1], true)).toBe(12); // prepared columns are numbers
  });

  it('reads CSV text with quotes and either delimiter', () => {
    expect(parseCsv('﻿a;b\n"x;1";"say ""hi"""\r\n\n2;3')).toEqual([
      ['a', 'b'],
      ['x;1', 'say "hi"'],
      ['2', '3'],
    ]);
    const text =
      'date,store,temp,promo,sales,extra\n2024-05-01,Berlin,3.5,true,120,x\n2024-05-02,Köln,,false,99,y\n';
    const rows = csvRows(text, COLUMNS, { maxRows: 10, target: 'sales' });
    expect(rows.rows).toEqual([
      { date: '2024-05-01', store: 'Berlin', temp: '3.5', promo: 'true' },
      { date: '2024-05-02', store: 'Köln', temp: '', promo: 'false' },
    ]);
    expect(rows.actual).toEqual(['120', '99']);
    expect(rows.ignored).toEqual(['extra']);
    expect(() => csvRows('date,store\n1,2', COLUMNS, { maxRows: 10 })).toThrow(
      'The header line lacks columns: temp, promo.'
    );
    expect(() => csvRows(text, COLUMNS, { maxRows: 1 })).toThrow('2 rows are too many: at most 1');
    expect(() => csvRows('date', COLUMNS, { maxRows: 1 })).toThrow('Paste a header line');
  });

  it('writes the results as CSV', () => {
    const text = resultsCsv(
      ['store', 'note'],
      [
        { store: 'Berlin', note: 'a, "b"' },
        { store: 'Köln', note: '' },
      ],
      [
        { class: 'yes', probability: 0.9 },
        { class: 'no', probability: 0.6 },
      ],
      'classification',
      'churn'
    );
    expect(text).toBe(
      'store,note,predicted_churn,probability\nBerlin,"a, ""b""",yes,0.9\nKöln,,no,0.6\n'
    );
    const values = resultsCsv(['x'], [{ x: 1 }], [{ value: 3.5, std: 0.2 }], 'regression', 'y');
    expect(values).toBe('x,predicted_y,std\n1,3.5,0.2\n');
  });

  it('reads and renumbers the messages of the server', () => {
    expect(parseMessage("Row 3, 'temp': 'hot' is not a number.")).toEqual({
      row: 3,
      column: 'temp',
      text: "'hot' is not a number.",
    });
    expect(parseMessage("The row, 'store': 'Bonn' does not occur")).toMatchObject({
      row: 1,
      column: 'store',
    });
    expect(parseMessage('Row 2: no match in stores.')).toMatchObject({ row: 2, column: null });
    expect(parseMessage('Something else.')).toMatchObject({ row: null });
    expect(shiftMessage("Row 3, 'temp': bad.", 64)).toBe("Row 67, 'temp': bad.");
    expect(shiftMessage('The row: bad.', 128)).toBe('Row 129: bad.');
    expect(shiftMessage('Other.', 64)).toBe('Other.');
    expect(chunks([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]]);
  });
});

describe('pixels for predictions', () => {
  it('turns RGBA pixels into channels × height × width like Pillow', () => {
    const rgba = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255];
    expect(grayOf(255, 255, 255)).toBe(255);
    expect(imageInput(rgba, 2, 2, 1)).toEqual([
      [
        [76, 149],
        [29, 255],
      ],
    ]);
    expect(imageInput(rgba, 2, 2, 3)).toEqual([
      [
        [255, 0],
        [0, 255],
      ],
      [
        [0, 255],
        [0, 255],
      ],
      [
        [0, 0],
        [255, 255],
      ],
    ]);
  });

  it('resizes by averaging', () => {
    const source = new Float32Array([0, 100, 200, 100]);
    expect(Array.from(resizeArea(source, 2, 2, 1, 1))).toEqual([100]);
    expect(Array.from(resizeArea(source, 2, 2, 4, 4)).slice(0, 4)).toEqual([0, 0, 100, 100]);
  });

  it('centres a drawn digit like MNIST', () => {
    expect(fitDrawing(new Float32Array(112 * 112), 112, 28, 28)).toBeNull();
    // A vertical line in the top left corner of the pad.
    const pixels = rasterize(
      [
        [
          { x: 0.1, y: 0.05 },
          { x: 0.1, y: 0.4 },
        ],
      ],
      112
    );
    const image = fitDrawing(pixels, 112, 28, 28)!;
    expect(image).toHaveLength(28);
    expect(image[0]).toHaveLength(28);
    const lit = image
      .flatMap((row, y) => row.map((v, x) => ({ x, y, v })))
      .filter((p) => p.v > 128);
    // Scaled to 20 pixels high and moved to the centre.
    const ys = lit.map((p) => p.y);
    const xs = lit.map((p) => p.x);
    expect(Math.max(...ys) - Math.min(...ys)).toBeGreaterThanOrEqual(17);
    expect(Math.abs((Math.max(...xs) + Math.min(...xs)) / 2 - 13.5)).toBeLessThanOrEqual(1.5);
    expect(Math.abs((Math.max(...ys) + Math.min(...ys)) / 2 - 13.5)).toBeLessThanOrEqual(1.5);
    expect(image.flat().every((v) => Number.isInteger(v) && v >= 0 && v <= 255)).toBe(true);
  });
});
