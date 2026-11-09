import { createContext, useContext } from 'react';
import type { DataPrep, ProjectRole } from '@/api';

/** What the user may do with the open preparation (it may be shared with them). */
export interface PrepAccess {
  /** Viewers of a shared preparation look at it: no changes, no training datasets. */
  readOnly: boolean;
  /** The owner's username if the preparation belongs to someone else. */
  sharedBy: string | null;
}

export const FULL_ACCESS: PrepAccess = { readOnly: false, sharedBy: null };

/** Read by the editor and the result views, which hide their editing actions for viewers. */
export const PrepAccessContext = createContext<PrepAccess>(FULL_ACCESS);

export function usePrepAccess(): PrepAccess {
  return useContext(PrepAccessContext);
}

/** Older servers do not send `role`: their preparations are all the user's own. */
export function prepRole(prep: Pick<DataPrep, 'role'>): ProjectRole {
  return prep.role ?? 'owner';
}

export function accessOf(prep: Pick<DataPrep, 'role' | 'owner'> | null): PrepAccess {
  if (!prep) {
    return FULL_ACCESS;
  }
  const role = prepRole(prep);
  return { readOnly: role === 'viewer', sharedBy: role === 'owner' ? null : prep.owner };
}
