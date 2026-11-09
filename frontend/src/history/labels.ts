import type { ProjectVersion } from '@/api/types';

/**
 * What an entry is called in the history: its name ("Baseline"), "Restored …" or, for
 * automatic entries, who changed the network ("Edited by bob", "Edited by you").
 */
export function versionTitle(entry: ProjectVersion, me: string | null): string {
  if (entry.kind === 'named' && entry.label) {
    return entry.label;
  }
  if (entry.kind === 'restore') {
    return entry.label || `Restored version ${entry.version}`;
  }
  if (!entry.created_by) {
    return 'Edited';
  }
  return `Edited by ${entry.created_by === me ? 'you' : entry.created_by}`;
}

/** Shown before the title, for screen readers and as a tooltip of the icon. */
export const KIND_LABELS: Record<ProjectVersion['kind'], string> = {
  auto: 'Automatic version',
  named: 'Named version',
  restore: 'Restored version',
};

export function layerCount(count: number): string {
  return count === 1 ? '1 layer' : `${count} layers`;
}

/** Who made a named or restored version (automatic ones say it in the title). */
export function versionAuthor(entry: ProjectVersion, me: string | null): string | null {
  if (entry.kind === 'auto' || !entry.created_by) {
    return null;
  }
  return entry.created_by === me ? 'you' : entry.created_by;
}
