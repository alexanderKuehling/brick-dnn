/** Test data for history and comments. */
import type { ProjectComment, ProjectVersion, UserRef } from '@/api/types';

export const ALICE: UserRef = { id: 1, username: 'alice' };
export const BOB: UserRef = { id: 2, username: 'bob' };
export const CAROL: UserRef = { id: 3, username: 'carol' };

/** An ISO time `minutes` before now. */
export function minutesAgo(minutes: number, now = Date.now()): string {
  return new Date(now - minutes * 60_000).toISOString();
}

export function comment(
  over: Partial<ProjectComment> & Pick<ProjectComment, 'id'>
): ProjectComment {
  return {
    parent: null,
    node_id: '',
    text: `Comment ${over.id}`,
    author: ALICE,
    resolved: false,
    resolved_by: null,
    created_at: minutesAgo(30),
    updated_at: minutesAgo(30),
    ...over,
  };
}

export function version(
  over: Partial<ProjectVersion> & Pick<ProjectVersion, 'id'>
): ProjectVersion {
  return {
    version: over.id,
    kind: 'auto',
    label: '',
    name: 'Shared MLP',
    num_layers: 2,
    created_by: 'alice',
    created_at: minutesAgo(5),
    updated_at: minutesAgo(5),
    ...over,
  };
}
