import { budgetText, estimateText } from './CostEstimate';
import { formatCpu, formatMoney, formatResetDate, usageColor, usagePercent } from './format';

test('formatCpu', () => {
  expect(formatCpu(0.0123)).toBe('0.01 CPU min');
  expect(formatCpu(2.5)).toBe('2.5 CPU min');
  expect(formatCpu(12.34)).toBe('12.3 CPU min');
  expect(formatCpu(1234.4)).toBe('1,234 CPU min');
  expect(formatCpu(3, '')).toBe('3');
  expect(formatCpu(null)).toBe('–');
});

test('formatMoney', () => {
  expect(formatMoney(0.27, 'EUR')).toBe('€0.27');
  expect(formatMoney(0.0042, 'USD')).toBe('$0.0042');
  expect(formatMoney(1, null)).toBe('');
  expect(formatMoney(null, 'EUR')).toBe('');
});

test('reset dates are UTC', () => {
  expect(formatResetDate('2026-12-01T00:00:00Z')).toBe('1 December 2026');
  expect(formatResetDate('2027-01-01T00:00:00Z')).toBe('1 January 2027');
});

test('usage bars', () => {
  expect(usagePercent(75, 300)).toBe(25);
  expect(usagePercent(400, 300)).toBe(100);
  expect(usagePercent(0, null)).toBe(0);
  expect(usageColor(25)).toBe('brick');
  expect(usageColor(80)).toBe('yellow');
  expect(usageColor(100)).toBe('red');
});

test('estimate and budget texts', () => {
  expect(estimateText({ estimate: null })).toBeNull();
  expect(
    estimateText({
      estimate: {
        cpu_minutes: 12,
        low_cpu_minutes: 4,
        high_cpu_minutes: 36,
        cost: null,
        currency: null,
        basis: '',
        note: '',
        details: {},
      },
    })
  ).toBe('≈ 12 CPU min (4–36)');
  const budget = {
    allowed: true,
    reason: null,
    warning: null,
    run_limit_cpu_minutes: 50,
    limit_reason: 'budget' as const,
    remaining_cpu_minutes: 50,
    resets_at: '2026-12-01T00:00:00Z',
  };
  expect(budgetText({ budget })).toBe(
    'It may use up to 50 CPU min (what is left of your budget). 50 CPU min left this month, until 1 December 2026.'
  );
  // Without any limit there is nothing to say; a refused start is explained elsewhere.
  expect(
    budgetText({
      budget: { ...budget, run_limit_cpu_minutes: null, remaining_cpu_minutes: null },
    })
  ).toBeNull();
  expect(budgetText({ budget: { ...budget, allowed: false } })).toBeNull();
});
