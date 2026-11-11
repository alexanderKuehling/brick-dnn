import type { ApprovalEstimate, ApprovalRule } from '@/api';
import { formatCpu, formatMoney } from '@/usage/format';

/** "Asks before trainings above 15 CPU min and publishing" or "Never asks". */
export function ruleSummary(rule: ApprovalRule): string {
  const parts: string[] = [];
  if (rule.training_cpu_minutes === 0) {
    parts.push('every training');
  } else if (rule.training_cpu_minutes !== null) {
    parts.push(`trainings above ${formatCpu(rule.training_cpu_minutes)}`);
  }
  if (rule.publishing) {
    parts.push('publishing');
  }
  if (rule.deleting) {
    parts.push('deleting');
  }
  if (parts.length === 0) {
    return 'Never asks';
  }
  const last = parts.pop();
  return `Asks before ${parts.length ? `${parts.join(', ')} and ${last}` : last}`;
}

/** "42 CPU min (14–126) · €0.70": the estimate of a training that waits. */
export function estimateText(estimate: ApprovalEstimate | null): string {
  if (!estimate || estimate.cpu_minutes === null) {
    return 'Cannot be estimated';
  }
  let text = formatCpu(estimate.cpu_minutes);
  if (estimate.low_cpu_minutes !== null && estimate.high_cpu_minutes !== null) {
    text += ` (${formatCpu(estimate.low_cpu_minutes, '')}–${formatCpu(estimate.high_cpu_minutes, '')})`;
  }
  const money = formatMoney(estimate.cost, estimate.currency);
  return money ? `${text} · ${money}` : text;
}

/** "in 23 h", "in 40 min" or "now": until a request expires. */
export function expiresIn(iso: string, now: Date = new Date()): string {
  const minutes = Math.round((new Date(iso).getTime() - now.getTime()) / 60_000);
  if (minutes <= 0) {
    return 'now';
  }
  if (minutes < 60) {
    return `in ${minutes} min`;
  }
  return `in ${Math.round(minutes / 60)} h`;
}
