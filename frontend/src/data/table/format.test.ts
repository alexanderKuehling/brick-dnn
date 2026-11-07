import {
  formatCell,
  formatCount,
  formatDateTime,
  formatNumber,
  formatShare,
  sampleCaption,
} from './format';

const NBSP = ' ';

describe('formatNumber', () => {
  it('uses thousands separators and keeps integers whole', () => {
    expect(formatNumber(950000)).toBe('950,000');
    expect(formatNumber(-1234)).toBe('-1,234');
    expect(formatNumber(0)).toBe('0');
  });

  it('shows six significant digits for fractions, more for large numbers', () => {
    expect(formatNumber(0.123456789)).toBe('0.123457');
    expect(formatNumber(3.14159265)).toBe('3.14159');
    expect(formatNumber(38400.5)).toBe('38,400.5');
    expect(formatNumber(1234567.891)).toBe('1,234,567.89');
  });

  it('uses exponents for tiny and huge values', () => {
    expect(formatNumber(1.5e-7)).toBe('1.5e-7');
    expect(formatNumber(2e20)).toBe('2e+20');
  });
});

describe('formatDateTime', () => {
  it('shows midnight as a date and other times with the time of day', () => {
    expect(formatDateTime(Date.UTC(2021, 2, 5))).toBe('2021-03-05');
    expect(formatDateTime(Date.UTC(2021, 2, 5, 14, 30))).toBe('2021-03-05 14:30:00');
    expect(formatDateTime(Date.UTC(2021, 2, 5, 14, 30, 0, 250))).toBe('2021-03-05 14:30:00.250');
  });
});

describe('formatCell', () => {
  it('formats by kind and aligns numbers right', () => {
    expect(formatCell(1234.5, 'numeric')).toEqual({
      text: '1,234.5',
      missing: false,
      align: 'right',
    });
    expect(formatCell(Date.UTC(2022, 0, 1), 'datetime').text).toBe('2022-01-01');
    expect(formatCell(1, 'boolean')).toEqual({ text: 'true', missing: false, align: 'center' });
    expect(formatCell(0, 'boolean').text).toBe('false');
    expect(formatCell('Berlin', 'categorical')).toEqual({
      text: 'Berlin',
      missing: false,
      align: 'left',
    });
  });

  it('marks missing values', () => {
    expect(formatCell(null, 'numeric')).toEqual({ text: 'missing', missing: true, align: 'right' });
    expect(formatCell(Number.NaN, 'numeric').missing).toBe(true);
    expect(formatCell(undefined, 'text').missing).toBe(true);
  });
});

describe('formatShare', () => {
  it('rounds to one decimal below 10 % and whole numbers above', () => {
    expect(formatShare(0, 100)).toBe(`0${NBSP}%`);
    expect(formatShare(1, 10000)).toBe(`<0.1${NBSP}%`);
    expect(formatShare(42, 1000)).toBe(`4.2${NBSP}%`);
    expect(formatShare(120, 1200)).toBe(`10${NBSP}%`);
    expect(formatShare(571, 1000)).toBe(`57${NBSP}%`);
    expect(formatShare(9999, 10000)).toBe(`>99.9${NBSP}%`);
    expect(formatShare(10, 10)).toBe(`100${NBSP}%`);
  });
});

describe('sampleCaption', () => {
  it('says whether the sample is complete', () => {
    expect(sampleCaption(5000, 1234567)).toBe('Showing 5,000 random rows of 1,234,567');
    expect(sampleCaption(1200, 1200)).toBe('Showing all 1,200 rows');
    expect(sampleCaption(1, 1)).toBe('Showing all 1 row');
    expect(formatCount(1234567)).toBe('1,234,567');
  });
});
