import type { GraphDocument } from '@/api/types';

/** The builder's working copy is kept in localStorage so a reload never loses work. */
export interface Draft {
  doc: GraphDocument;
  name: string;
  projectId: number | null;
}

const DRAFT_KEY = 'brickdnn:draft';

export function loadDraft(): Draft | null {
  try {
    const raw = localStorage.getItem(DRAFT_KEY);
    if (!raw) {
      return null;
    }
    const draft = JSON.parse(raw) as Draft;
    if (!draft?.doc || !Array.isArray(draft.doc.nodes) || !Array.isArray(draft.doc.edges)) {
      return null;
    }
    return {
      doc: draft.doc,
      name: draft.name || 'Untitled network',
      projectId: draft.projectId ?? null,
    };
  } catch {
    return null;
  }
}

export function saveDraft(draft: Draft): void {
  try {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
  } catch {
    // Storage full or unavailable: the draft simply is not persisted.
  }
}
