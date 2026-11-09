import { version } from '@/collab/fixtures';
import { layerCount, versionAuthor, versionTitle } from './labels';

describe('versionTitle', () => {
  it('names named versions by their name', () => {
    expect(versionTitle(version({ id: 1, kind: 'named', label: 'Baseline' }), 'bob')).toBe(
      'Baseline'
    );
  });

  it('says what was restored', () => {
    const entry = version({ id: 2, kind: 'restore', label: 'Restored Baseline' });
    expect(versionTitle(entry, 'bob')).toBe('Restored Baseline');
    expect(versionTitle({ ...entry, label: '' }, 'bob')).toBe('Restored version 2');
  });

  it('names automatic entries after who edited', () => {
    expect(versionTitle(version({ id: 3, created_by: 'bob' }), 'alice')).toBe('Edited by bob');
    expect(versionTitle(version({ id: 3, created_by: 'bob' }), 'bob')).toBe('Edited by you');
    expect(versionTitle(version({ id: 3, created_by: null }), 'bob')).toBe('Edited');
  });
});

describe('versionAuthor', () => {
  it('is only shown where the title does not say it', () => {
    expect(versionAuthor(version({ id: 1, created_by: 'bob' }), 'alice')).toBeNull();
    expect(versionAuthor(version({ id: 1, kind: 'named', created_by: 'bob' }), 'alice')).toBe(
      'bob'
    );
    expect(versionAuthor(version({ id: 1, kind: 'named', created_by: 'bob' }), 'bob')).toBe('you');
  });
});

it('counts layers', () => {
  expect(layerCount(1)).toBe('1 layer');
  expect(layerCount(0)).toBe('0 layers');
  expect(layerCount(12)).toBe('12 layers');
});
