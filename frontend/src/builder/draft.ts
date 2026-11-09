import type { GraphDocument, Project, ProjectRole } from '@/api/types';

/** The builder's working copy is kept in localStorage so a reload never loses work. */
export interface Draft {
  doc: GraphDocument;
  name: string;
  projectId: number | null;
  /** The version of the saved project the working copy is based on (null: unknown). */
  version?: number | null;
  /** The user's role in the saved project (null: unknown, e.g. drafts from older versions). */
  role?: ProjectRole | null;
  /** The working copy had no unsaved changes when it was stored. */
  saved?: boolean;
}

const DRAFT_KEY = 'brickdnn:draft';
const ROLES: readonly string[] = ['owner', 'editor', 'viewer'];

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
    const projectId = draft.projectId ?? null;
    return {
      doc: draft.doc,
      name: draft.name || 'Untitled network',
      projectId,
      version: projectId !== null && typeof draft.version === 'number' ? draft.version : null,
      role:
        projectId !== null && ROLES.includes(draft.role as string)
          ? (draft.role as ProjectRole)
          : null,
      saved: projectId !== null && draft.saved === true,
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

/** The draft for opening a saved project as it is on the server. */
export function projectDraft(project: Project): Draft {
  return {
    doc: project.graph,
    name: project.name,
    projectId: project.id,
    version: project.version,
    role: project.role,
    saved: true,
  };
}
