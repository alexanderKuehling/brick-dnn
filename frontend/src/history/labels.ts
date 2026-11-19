import type { ProjectVersion } from '@/api/types';
import { i18n } from '@/i18n';

/**
 * What an entry is called in the history: its name ("Baseline"), "Restored …" or, for
 * automatic entries, who changed the network ("Edited by bob", "Edited by you").
 */
export function versionTitle(entry: ProjectVersion, me: string | null): string {
  if (entry.kind === 'named' && entry.label) {
    return entry.label;
  }
  if (entry.kind === 'restore') {
    return entry.label || i18n.t('ui.history.labels.restoredVersion', { version: entry.version });
  }
  if (!entry.created_by) {
    return i18n.t('ui.history.labels.edited');
  }
  return i18n.t('ui.history.labels.editedBy', {
    value: entry.created_by === me ? i18n.t('ui.history.labels.you') : entry.created_by,
  });
}

/** Shown before the title, for screen readers and as a tooltip of the icon. */
export const KIND_LABELS: Record<ProjectVersion['kind'], string> = {
  get auto() {
    return i18n.t('ui.history.labels.automaticVersion');
  },
  get named() {
    return i18n.t('ui.history.labels.namedVersion');
  },
  get restore() {
    return i18n.t('ui.history.labels.restoredVersion2');
  },
};

export function layerCount(count: number): string {
  return i18n.t('ui.history.labels.layers', { count });
}

/** Who made a named or restored version (automatic ones say it in the title). */
export function versionAuthor(entry: ProjectVersion, me: string | null): string | null {
  if (entry.kind === 'auto' || !entry.created_by) {
    return null;
  }
  return entry.created_by === me ? 'you' : entry.created_by;
}
