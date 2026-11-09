import type { AgentAction } from '@/api';
import {
  actionLabel,
  argumentText,
  formatDuration,
  groupEntries,
  summaryParts,
  unnamedTargets,
} from './format';

function action(id: number, changes: Partial<AgentAction> = {}): AgentAction {
  return {
    id,
    created_at: '2026-11-09T14:00:00Z',
    source: 'mcp',
    agent: { kind: 'token', id: 1, key: 'token:1', name: 'Laptop', prefix: 'bdnn_x' },
    user_agent: '',
    tool: 'get_training_run',
    method: '',
    writes: false,
    outcome: 'ok',
    error: '',
    status_code: null,
    duration_ms: 10,
    summary: 'Checked training “GP 1”: running',
    arguments: {},
    targets: [{ kind: 'run', id: 'r1', name: 'GP 1' }],
    project: null,
    ...changes,
  };
}

test('formatDuration', () => {
  expect(formatDuration(42.4)).toBe('42 ms');
  expect(formatDuration(1234)).toBe('1.2 s');
  expect(formatDuration(42_000)).toBe('42 s');
  expect(formatDuration(125_000)).toBe('2 min 5 s');
  expect(formatDuration(120_000)).toBe('2 min');
});

test('actionLabel', () => {
  expect(actionLabel({ source: 'mcp', tool: 'save_project', method: '' })).toBe('save_project');
  expect(actionLabel({ source: 'api', tool: 'project-detail', method: 'PATCH' })).toBe(
    'PATCH project-detail'
  );
});

test('argumentText', () => {
  expect(argumentText('48 KB, 801 lines')).toBe('48 KB, 801 lines');
  expect(argumentText({ epochs: 20, learning_rate: 0.001, shuffle: true })).toBe(
    'epochs: 20, learning_rate: 0.001, shuffle: yes'
  );
  expect(argumentText([3, 28, 28])).toBe('[3, 28, 28]');
  expect(argumentText(null)).toBe('–');
});

test('summaryParts marks the names of what was touched', () => {
  const targets = [
    { kind: 'run' as const, id: 'r1', name: 'GP 1' },
    { kind: 'dataset' as const, id: 3, name: 'houses.csv' },
    { kind: 'project' as const, id: 12 },
  ];
  expect(summaryParts('Started training “GP 1” on dataset “houses.csv” (#3)', targets)).toEqual([
    'Started training ',
    { target: targets[0], text: '“GP 1”' },
    ' on dataset ',
    { target: targets[1], text: '“houses.csv”' },
    ' (#3)',
  ]);
  expect(summaryParts('Listed the saved networks', targets)).toEqual(['Listed the saved networks']);
  // Only the first mention becomes a link.
  const same = [{ kind: 'project' as const, id: 1, name: 'A' }];
  expect(summaryParts('Copied “A” to “A”', same)).toEqual([
    'Copied ',
    { target: same[0], text: '“A”' },
    ' to “A”',
  ]);
  expect(unnamedTargets('Stopped training “GP 1”', targets)).toEqual([targets[1]]);
});

test('groupEntries groups runs of the same read', () => {
  const groups = groupEntries([
    action(6),
    action(5),
    action(4, { outcome: 'error', error: 'Gone' }),
    action(3),
    action(2, { agent: { kind: 'app', id: 7, key: 'app:7', name: 'Claude', prefix: null } }),
    action(1, { tool: 'save_project', writes: true }),
    action(0, { tool: 'save_project', writes: true }),
  ]);
  expect(groups.map((group) => [group.key, group.entries.map((entry) => entry.id)])).toEqual([
    [5, [6, 5]],
    [4, [4]],
    [3, [3]],
    [2, [2]],
    [1, [1]],
    [0, [0]],
  ]);
  const differentRuns = groupEntries([
    action(2),
    action(1, { targets: [{ kind: 'run', id: 'r2', name: 'GP 2' }] }),
  ]);
  expect(differentRuns).toHaveLength(2);
});
