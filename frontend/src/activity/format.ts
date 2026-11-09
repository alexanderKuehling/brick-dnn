import type { AgentAction, AgentArgument, AgentRef, AgentTarget, AgentTargetKind } from '@/api';

export const TARGET_LABELS: Record<AgentTargetKind, string> = {
  project: 'Network',
  run: 'Training',
  dataset: 'Dataset',
  data_source: 'Data file',
  data_prep: 'Data preparation',
};

/** "42 ms", "1.3 s", "2 min 5 s". */
export function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${Math.max(0, Math.round(ms))} ms`;
  }
  if (ms < 60_000) {
    return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)} s`;
  }
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return seconds ? `${minutes} min ${seconds} s` : `${minutes} min`;
}

/** The tool of an MCP call ("save_project") or the method and route of a REST request. */
export function actionLabel(entry: Pick<AgentAction, 'source' | 'tool' | 'method'>): string {
  return entry.source === 'api' && entry.method ? `${entry.method} ${entry.tool}` : entry.tool;
}

export function agentLabel(agent: Pick<AgentRef, 'kind' | 'name'>): string {
  return `${agent.kind === 'app' ? 'App' : 'Token'} “${agent.name}”`;
}

function scalarText(value: string | number | boolean | null): string {
  if (value === null) {
    return '–';
  }
  if (typeof value === 'boolean') {
    return value ? 'yes' : 'no';
  }
  return String(value);
}

/** A summarised argument as text: settings as "epochs: 20, learning_rate: 0.001". */
export function argumentText(value: AgentArgument): string {
  if (Array.isArray(value)) {
    return `[${value.map(scalarText).join(', ')}]`;
  }
  if (value !== null && typeof value === 'object') {
    const parts = Object.entries(value).map(([key, item]) => `${key}: ${argumentText(item)}`);
    return parts.length ? parts.join(', ') : '{}';
  }
  return scalarText(value);
}

export type SummaryPart = string | { target: AgentTarget; text: string };

/**
 * The summary with the names of the targets marked, so that they can become links:
 * 'Saved network “House prices” (#12)' → ['Saved network ', {“House prices”}, ' (#12)'].
 */
export function summaryParts(summary: string, targets: AgentTarget[]): SummaryPart[] {
  let parts: SummaryPart[] = [summary];
  for (const target of targets) {
    if (!target.name) {
      continue;
    }
    const quoted = `“${target.name}”`;
    let found = false;
    parts = parts.flatMap((part) => {
      if (found || typeof part !== 'string') {
        return [part];
      }
      const index = part.indexOf(quoted);
      if (index < 0) {
        return [part];
      }
      found = true;
      // The quotes belong to the link, so that a line never breaks between them and the name.
      const before = part.slice(0, index);
      const after = part.slice(index + quoted.length);
      return [before, { target, text: quoted }, after].filter((p) => p !== '');
    });
  }
  return parts;
}

/** Targets the summary does not name (shown as links next to it). */
export function unnamedTargets(summary: string, targets: AgentTarget[]): AgentTarget[] {
  return targets.filter((target) => target.name && !summary.includes(`“${target.name}”`));
}

/** Consecutive entries shown as one row: e.g. an assistant checking a training every second. */
export interface EntryGroup {
  /** Stable while new entries join at the top: the id of the oldest one. */
  key: number;
  /** Newest first. */
  entries: AgentAction[];
}

function sameTargets(a: AgentTarget[], b: AgentTarget[]): boolean {
  return a.length === b.length && a.every((t, i) => t.kind === b[i].kind && t.id === b[i].id);
}

/**
 * Groups runs of the same read-only call of the same agent on the same things that worked
 * (polling a training, listing networks again). Changes and errors always stay single.
 */
export function canGroup(a: AgentAction, b: AgentAction): boolean {
  return (
    !a.writes &&
    !b.writes &&
    a.outcome === 'ok' &&
    b.outcome === 'ok' &&
    a.agent.key === b.agent.key &&
    a.source === b.source &&
    a.tool === b.tool &&
    a.method === b.method &&
    sameTargets(a.targets, b.targets)
  );
}

export function groupEntries(entries: AgentAction[]): EntryGroup[] {
  const groups: EntryGroup[] = [];
  for (const entry of entries) {
    const last = groups[groups.length - 1];
    if (last && canGroup(last.entries[last.entries.length - 1], entry)) {
      last.entries.push(entry);
      last.key = entry.id;
    } else {
      groups.push({ key: entry.id, entries: [entry] });
    }
  }
  return groups;
}
