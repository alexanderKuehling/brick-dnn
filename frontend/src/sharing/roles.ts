import { api, type MemberRole, type ProjectRole } from '@/api';

export const ROLE_LABELS: Record<ProjectRole, string> = {
  owner: 'Owner',
  editor: 'Can edit',
  viewer: 'Can view',
};

export const MEMBER_ROLE_OPTIONS: { value: MemberRole; label: string }[] = [
  { value: 'editor', label: ROLE_LABELS.editor },
  { value: 'viewer', label: ROLE_LABELS.viewer },
];

/** The public page of a share link. */
export function shareUrl(token: string): string {
  return new URL(`/share/${encodeURIComponent(token)}`, window.location.origin).href;
}

/** Remove the current user from a project shared with them. */
export async function leaveProject(projectId: number, userId: number): Promise<void> {
  const { members } = await api.projects.members.list(projectId);
  const own = members.find((member) => member.user.id === userId);
  if (!own) {
    return; // Removed already.
  }
  await api.projects.members.remove(projectId, own.id);
}
