/** Times in the history and in comments: relative while recent, else date and time. */

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/** The local calendar day, e.g. "2026-11-09" (to group by). */
export function dayKey(time: Date): string {
  const month = String(time.getMonth() + 1).padStart(2, '0');
  const day = String(time.getDate()).padStart(2, '0');
  return `${time.getFullYear()}-${month}-${day}`;
}

function startOfDay(time: Date): number {
  return new Date(time.getFullYear(), time.getMonth(), time.getDate()).getTime();
}

/** "14:20" (in the user's locale). */
export function clockTime(iso: string): string {
  return new Date(iso).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

/** The full date and time, e.g. for tooltips. */
export function fullDateTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'full', timeStyle: 'short' });
}

/** "just now", "5 min ago" or "3 h ago" within a day; null for older times. */
function ago(iso: string, now: Date): string | null {
  const elapsed = now.getTime() - new Date(iso).getTime();
  if (elapsed < MINUTE) {
    return 'just now'; // Also slightly in the future (clocks differ).
  }
  if (elapsed < HOUR) {
    return `${Math.floor(elapsed / MINUTE)} min ago`;
  }
  if (elapsed < DAY) {
    return `${Math.floor(elapsed / HOUR)} h ago`;
  }
  return null;
}

/** Relative within a day, then the date: "5 min ago", "Nov 3, 14:20", "Nov 3, 2025". */
export function relativeTime(iso: string, now: Date = new Date()): string {
  const recent = ago(iso, now);
  if (recent) {
    return recent;
  }
  const time = new Date(iso);
  if (time.getFullYear() !== now.getFullYear()) {
    return time.toLocaleDateString(undefined, { dateStyle: 'medium' });
  }
  const date = time.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  return `${date}, ${clockTime(iso)}`;
}

/** For lists grouped by day: relative within a day, else the time of day ("14:20"). */
export function timeInDay(iso: string, now: Date = new Date()): string {
  return ago(iso, now) ?? clockTime(iso);
}

/** "Today", "Yesterday", "Monday, November 3" or "November 3, 2025". */
export function dayLabel(time: Date, now: Date = new Date()): string {
  const days = Math.round((startOfDay(now) - startOfDay(time)) / DAY);
  if (days === 0) {
    return 'Today';
  }
  if (days === 1) {
    return 'Yesterday';
  }
  if (time.getFullYear() === now.getFullYear()) {
    return time.toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' });
  }
  return time.toLocaleDateString(undefined, { dateStyle: 'long' });
}

export interface DayGroup<T> {
  key: string;
  label: string;
  items: T[];
}

/** Consecutive items of the same local day, in the given order (e.g. newest first). */
export function groupByDay<T>(
  items: T[],
  timeOf: (item: T) => string,
  now: Date = new Date()
): DayGroup<T>[] {
  const groups: DayGroup<T>[] = [];
  for (const item of items) {
    const time = new Date(timeOf(item));
    const key = dayKey(time);
    const last = groups[groups.length - 1];
    if (last?.key === key) {
      last.items.push(item);
    } else {
      groups.push({ key, label: dayLabel(time, now), items: [item] });
    }
  }
  return groups;
}
