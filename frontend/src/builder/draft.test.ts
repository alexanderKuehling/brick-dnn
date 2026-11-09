import { beforeEach } from 'vitest';
import { loadDraft, saveDraft } from './draft';

const DOC = { version: 1 as const, nodes: [], edges: [] };

beforeEach(() => localStorage.clear());

describe('draft', () => {
  it('keeps the version and role of the open project', () => {
    saveDraft({ doc: DOC, name: 'Net', projectId: 4, version: 7, role: 'viewer', saved: true });
    expect(loadDraft()).toEqual({
      doc: DOC,
      name: 'Net',
      projectId: 4,
      version: 7,
      role: 'viewer',
      saved: true,
    });
  });

  it('reads drafts from before sharing existed', () => {
    localStorage.setItem('brickdnn:draft', JSON.stringify({ doc: DOC, name: 'Old', projectId: 4 }));
    expect(loadDraft()).toEqual({
      doc: DOC,
      name: 'Old',
      projectId: 4,
      version: null,
      role: null,
      saved: false,
    });
  });

  it('ignores project details without a project and unknown roles', () => {
    saveDraft({ doc: DOC, name: 'New', projectId: null, version: 3, role: 'owner', saved: true });
    expect(loadDraft()).toMatchObject({ projectId: null, version: null, role: null, saved: false });
    localStorage.setItem(
      'brickdnn:draft',
      JSON.stringify({ doc: DOC, name: 'X', projectId: 2, version: 'x', role: 'admin' })
    );
    expect(loadDraft()).toMatchObject({ version: null, role: null });
  });
});
