import { dayKey, dayLabel, groupByDay, relativeTime, timeInDay } from './time';

const NOW = new Date(2026, 10, 9, 15, 0); // Monday, 9 November 2026, 15:00 local time
const at = (days: number, hours: number, minutes = 0) =>
  new Date(2026, 10, 9 - days, hours, minutes).toISOString();

describe('relativeTime', () => {
  it('is relative within a day', () => {
    expect(relativeTime(at(0, 14, 59), NOW)).toBe('1 min ago');
    expect(relativeTime(new Date(NOW.getTime() - 20_000).toISOString(), NOW)).toBe('just now');
    expect(relativeTime(at(0, 14, 15), NOW)).toBe('45 min ago');
    expect(relativeTime(at(0, 12, 0), NOW)).toBe('3 h ago');
    expect(relativeTime(at(1, 16, 0), NOW)).toBe('23 h ago');
  });

  it('shows the date for older times, with the year if it is another one', () => {
    expect(relativeTime(at(3, 9, 30), NOW)).toContain('Nov');
    expect(relativeTime(at(3, 9, 30), NOW)).not.toContain('2026');
    expect(relativeTime(new Date(2025, 4, 2, 10).toISOString(), NOW)).toContain('2025');
  });

  it('treats times slightly in the future as now', () => {
    expect(relativeTime(new Date(NOW.getTime() + 5_000).toISOString(), NOW)).toBe('just now');
  });
});

describe('timeInDay', () => {
  it('is relative within a day, then the time of day', () => {
    expect(timeInDay(at(0, 14, 50), NOW)).toBe('10 min ago');
    expect(timeInDay(at(2, 9, 5), NOW)).toMatch(/9.05|09.05/);
  });
});

describe('days', () => {
  it('names today and yesterday', () => {
    expect(dayLabel(new Date(at(0, 1)), NOW)).toBe('Today');
    expect(dayLabel(new Date(at(1, 23)), NOW)).toBe('Yesterday');
    expect(dayLabel(new Date(at(2, 12)), NOW)).toContain('November');
    expect(dayLabel(new Date(2025, 0, 3), NOW)).toContain('2025');
  });

  it('groups consecutive items by local day', () => {
    const items = [at(0, 14), at(0, 9), at(1, 18), at(5, 10), at(5, 8)];
    const groups = groupByDay(items, (item) => item, NOW);
    expect(groups.map((group) => group.label.split(',')[0])).toEqual([
      'Today',
      'Yesterday',
      'Wednesday',
    ]);
    expect(groups.map((group) => group.items.length)).toEqual([2, 1, 2]);
    expect(groups[0].key).toBe(dayKey(NOW));
  });
});
