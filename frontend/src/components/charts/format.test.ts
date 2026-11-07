import {
  dateUnitForValues,
  formatBytes,
  formatCoefficient,
  formatCompact,
  formatCount,
  formatDate,
  formatDateTime,
  formatDuration,
  formatNumber,
  formatPercent,
  timeUnitFor,
} from './format';

describe('format', () => {
  it('formats counts with thousands separators', () => {
    expect(formatCount(1234567)).toBe('1,234,567');
    expect(formatCount(null)).toBe('–');
  });

  it('formats measured values with four significant digits', () => {
    expect(formatNumber(41200)).toBe('41,200');
    expect(formatNumber(1234567.8)).toBe('1,234,568');
    expect(formatNumber(0.41834)).toBe('0.4183');
    expect(formatNumber(-0)).toBe('0');
    expect(formatNumber(0.000015)).toBe('1.5e-5');
    expect(formatNumber(3.2e18)).toBe('3.2e+18');
    expect(formatNumber(NaN)).toBe('–');
    expect(formatCompact(12900)).toBe('12.9K');
    expect(formatCompact(950)).toBe('950');
  });

  it('formats shares so that they never round to 0 or 100 %', () => {
    expect(formatPercent(0.5)).toBe('50%');
    expect(formatPercent(0.042)).toBe('4.2%');
    expect(formatPercent(0.05)).toBe('5%');
    expect(formatPercent(0.0004)).toBe('<0.1%');
    expect(formatPercent(0.9996)).toBe('>99.9%');
    expect(formatPercent(0.995)).toBe('99.5%');
    expect(formatPercent(0)).toBe('0%');
    expect(formatPercent(1)).toBe('100%');
    expect(formatPercent(undefined)).toBe('–');
  });

  it('formats coefficients with a real minus sign', () => {
    expect(formatCoefficient(-0.4213)).toBe('−0.42');
    expect(formatCoefficient(-0.001)).toBe('0.00');
    expect(formatCoefficient(1)).toBe('1.00');
    expect(formatCoefficient(null)).toBe('–');
  });

  it('formats bytes', () => {
    expect(formatBytes(312_000)).toBe('305 KB');
    expect(formatBytes(512)).toBe('512 bytes');
  });

  it('formats timestamps in UTC at the precision asked for', () => {
    const ms = Date.UTC(2021, 2, 4, 13, 5, 9);
    expect(formatDate(ms, 'year')).toBe('2021');
    expect(formatDate(ms, 'month')).toBe('Mar 2021');
    expect(formatDate(ms, 'day')).toBe('2021-03-04');
    expect(formatDate(ms, 'minute')).toBe('2021-03-04 13:05');
    expect(formatDate(ms, 'second')).toBe('2021-03-04 13:05:09');
    expect(formatDate(null)).toBe('–');
    expect(timeUnitFor(400 * 86_400_000)).toBe('year');
    expect(timeUnitFor(3_600_000)).toBe('minute');
  });

  it('formats timestamps in full for tooltips and labels', () => {
    expect(formatDateTime(Date.UTC(2021, 2, 4))).toBe('2021-03-04');
    expect(formatDateTime(Date.UTC(2021, 2, 4, 13, 5, 9))).toBe('2021-03-04 13:05:09');
    expect(formatDateTime(Date.UTC(2021, 2, 4, 0, 0, 0, 250))).toBe('2021-03-04 00:00:00.250');
    expect(formatDateTime(Date.UTC(1969, 11, 31, 23))).toBe('1969-12-31 23:00:00');
    expect(formatDateTime(null)).toBe('–');
  });

  it('finds the precision datetime values need', () => {
    expect(dateUnitForValues([Date.UTC(2021, 0, 1), Date.UTC(2021, 5, 3)])).toBe('day');
    expect(dateUnitForValues([Date.UTC(2021, 0, 1, 12)])).toBe('minute');
    expect(dateUnitForValues([Date.UTC(2021, 0, 1, 12, 0, 5)])).toBe('second');
  });

  it('formats durations', () => {
    expect(formatDuration(3 * 86_400_000)).toBe('3 days');
    expect(formatDuration(Date.UTC(2024, 0, 1) - Date.UTC(2021, 0, 1))).toBe('3 years');
    expect(formatDuration(5 * 3_600_000)).toBe('5 hours');
    expect(formatDuration(500)).toBe('500 ms');
  });
});
