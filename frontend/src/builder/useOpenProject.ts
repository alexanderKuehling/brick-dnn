import { useCallback, useEffect, useState } from 'react';
import { api, type Project, type ProjectMembers, type ProjectRole } from '@/api';
import type { SharingState } from '@/sharing/ShareDialog';

/** A project as the server describes it, without the graph (the canvas holds that). */
export type ProjectInfo = Omit<Project, 'graph'>;

/** The saved project that is open in the builder: the canvas is a working copy of it. */
export interface OpenProject {
  id: number;
  /** The version the working copy is based on, sent with saves. null: unknown, saves overwrite. */
  version: number | null;
  /** null until the server has answered (e.g. drafts from before sharing existed). */
  role: ProjectRole | null;
  /** Owner and sharing state from the server; null until it has answered. */
  info: ProjectInfo | null;
}

export function openProjectOf(project: Project): OpenProject {
  const { graph: _graph, ...info } = project;
  return { id: project.id, version: project.version, role: project.role, info };
}

/**
 * The builder's project, role and version, kept apart from the graph so that collaboration
 * features can build on it. Also loads who has access to a shared project (for the toolbar).
 */
export function useOpenProject(initial: OpenProject | null) {
  const [project, setProject] = useState<OpenProject | null>(initial);
  const [members, setMembers] = useState<{ projectId: number; data: ProjectMembers } | null>(null);

  const attach = useCallback((source: Project) => setProject(openProjectOf(source)), []);
  const detach = useCallback(() => setProject(null), []);

  const id = project?.id ?? null;
  const shared = Boolean(project?.info?.shared);
  useEffect(() => {
    if (id === null || !shared) {
      setMembers(null);
      return undefined;
    }
    if (members?.projectId === id) {
      return undefined; // Known already (e.g. just changed in the share dialog).
    }
    let cancelled = false;
    api.projects.members
      .list(id)
      .then((data) => !cancelled && setMembers({ projectId: id, data }))
      .catch(() => undefined); // Only the avatars are missing then.
    return () => {
      cancelled = true;
    };
  }, [id, shared]);

  /** Takes over changes made in the share dialog. */
  const applySharing = useCallback(
    (state: SharingState) => {
      if (id === null) {
        return;
      }
      setMembers({ projectId: id, data: state.members });
      const { shared: isShared, link_access, share_token } = state;
      setProject((current) =>
        current?.id === id && current.info
          ? { ...current, info: { ...current.info, shared: isShared, link_access, share_token } }
          : current
      );
    },
    [id]
  );

  return {
    project,
    setProject,
    attach,
    detach,
    applySharing,
    /** Owner and members of a shared project, null otherwise (or while loading). */
    members: members && members.projectId === id ? members.data : null,
    /** Viewers can look at the network but not change it. */
    readOnly: project?.role === 'viewer',
  };
}
